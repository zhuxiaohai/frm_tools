import numpy as np
import pandas as pd
from scipy import special, optimize
import matplotlib.pyplot as plt
from operator import gt, lt

import optuna
with optuna._imports.try_import() as _imports:
    import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMPruningCallback

from lightgbm.basic import _ConfigAliases, _log_info, _log_warning
from lightgbm.callback import EarlyStopException, _format_eval_result

from sklearn.model_selection import PredefinedSplit,  StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


def train_val_score(train_score, val_score, w):
    output_scores = val_score - abs(train_score - val_score) * w
    return output_scores


def custom_f1_score(y_pred, y_true_dmatrix):
    y_true = y_true_dmatrix.get_label()
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return 'f1', f1, True


def custom_ks_score(y_pred, y_true_dmatrix):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    y_true = y_true_dmatrix.get_label()
    ks = ks_stats(y_true, y_pred)
    return 'ks', ks, True


def regularize_metric(eval_metric):
    """
    :param eval_metric: list of str, each str should be a built-in metric of sklearn
    """
    eval_metric_list = []
    feval = None
    for metric in eval_metric:
        if metric == 'f1':
            # For custom function, you can specify the optimization direction in lgb api,
            # but for built in metrics, lgb will automatically decide
            # according to the metric type, e.g., the smaller the better for error
            # but the bigger the better for auc, etc.
            feval = custom_f1_score
        elif metric == 'accuracy':
            # It is calculated as #(wrong cases)/#(all cases).
            # The evaluation will regard the instances
            # with prediction value larger than 0.5 as positive instances, i.e., 1 instances)
            eval_metric_list.append('error')
        elif metric == 'roc_auc':
            eval_metric_list.append('auc')
        elif metric == 'ks':
            # same logic as f1
            feval = custom_ks_score
        else:
            raise RuntimeError('not-supported metric: {}'.format(metric))
    return feval, eval_metric_list


def make_train_val(x_train, y_train, eval_set, cv, random_state):
    """
    :param x_train: array-like(dim=2), Feature matrix
    :param y_train: array-like(dim=1), Label
    :param eval_set: a list of (X, y) tuples, only one tuple is supported as of now
                     if this is set, cv will always be 1
    :param cv: int, number of folds for stratified cross_validation
    :param random_state: int
    """
    if eval_set is not None:
        print('using self-defined eval-set')
        assert len(eval_set) == 1
        cv = 1
        if type(x_train) is pd.core.frame.DataFrame:
            x_train_val = pd.concat([x_train, eval_set[0][0]], axis=0)
            y_train_val = pd.concat([y_train, eval_set[0][1]], axis=0)
        else:
            x_train_val = np.concatenate((x_train, eval_set[0][0]), axis=0)
            y_train_val = np.concatenate((y_train, eval_set[0][1]), axis=0)
        # initialize all indices to 0 except the section of training
        # to -1, which means this part will not be in validation.
        # So only one fold is made
        test_fold = np.zeros(x_train_val.shape[0])
        test_fold[:x_train.shape[0]] = -1
        ps = PredefinedSplit(test_fold=test_fold)
        folds = []
        for train_indices_array, val_indices_array in ps.split():
            folds.append((train_indices_array.tolist(),
                          val_indices_array.tolist()))
    else:
        print('using cv {}'.format(cv))
        x_train_val = x_train
        y_train_val = y_train
        # folds = None
        skf = StratifiedKFold(n_splits=cv, shuffle=True,
                              random_state=random_state)
        folds = []
        for train_indices_array, val_indices_array in skf.split(
                x_train_val, y_train_val):
            folds.append((train_indices_array.tolist(),
                          val_indices_array.tolist()))
    return x_train_val, y_train_val, cv, folds


def my_early_stopping(stopping_rounds, w=0.2, verbose=True, period=1, show_stdv=True):
    """Create a callback that activates early stopping for cv.

    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.

    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    w : float
       Coefficient to balance train and val scores
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    period : int, optional (default=1)
        The period to print the evaluation results.
    show_stdv : bool, optional (default=True)
        Whether to show stdv (if provided).

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    # always use the last metric for early stopping
    last_metric_only = True
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]
    last_metric = ['']
    best_balance_score = [0.0]
    best_balance_iter = [0]
    best_balance_score_list = [None]

    def _init(env):
        enabled[0] = not any(env.params.get(boost_alias, "") == 'dart' for boost_alias
                             in _ConfigAliases.get("boosting"))
        if not enabled[0]:
            _log_warning('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')
        if env.evaluation_result_list[0][0] != "cv_agg":
            raise ValueError('the early stopping is only customized for cv')

        if verbose:
            _log_info("Training until validation scores don't improve for {} rounds".format(stopping_rounds))

        # # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        last_metric[0] = env.evaluation_result_list[-1][1].split(" ")[-1]
        best_balance_score[0] = float('-inf') if env.evaluation_result_list[-1][3] else float('inf')
        best_balance_iter[0] = 0
        best_balance_score_list[0] = None
        if w > 0:
            if env.evaluation_result_list[0][1].split(" ")[0] != 'train':
                raise ValueError('train data must be available to balance train and val')
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _final_iteration_check(env, eval_name_splitted, i):
        if env.iteration == env.end_iteration - 1:
            if verbose:
                _log_info('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                # if last_metric_only:
                #     _log_info("Evaluated only: {}".format(eval_name_splitted[-1]))
            raise EarlyStopException(best_iter[i], best_score_list[i])

    def _fetch_balance_train_score(env):
        for i in range(len(env.evaluation_result_list)):
            data_type, metric_name = env.evaluation_result_list[i][1].split(" ")
            if (data_type == 'train') and (metric_name == last_metric[0]):
                return env.evaluation_result_list[i][2]

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        log_flag = verbose and env.evaluation_result_list and ((env.iteration + 1) % period == 0)
        if log_flag:
            log_result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
        # get the train score of the last metric before hand
        if w > 0:
            balance_train_score = _fetch_balance_train_score(env)
        for i in range(len(env.evaluation_result_list)):
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            score = env.evaluation_result_list[i][2]
            # record best score as of now for whatever metric and dataset
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list           
            if (eval_name_splitted[0] == 'train') or (eval_name_splitted[-1] != last_metric[0]):
                _final_iteration_check(env, eval_name_splitted, i)
                continue
            assert (eval_name_splitted[0] != 'train') and (eval_name_splitted[-1] == last_metric[0])
            # the codes below will be executed only when dataset is not train and metric is last_metric
            if w > 0:
                balance_score = train_val_score(balance_train_score, score, w)
                if log_flag:
                    log_result += '\t%s\'s %s: %g' % ('balance', eval_name_splitted[-1], balance_score)
            else:
                balance_score = score
            if cmp_op[i](balance_score, best_balance_score[0]):
                best_balance_score[0] = balance_score
                best_balance_iter[0] = env.iteration
                best_balance_score_list[0] = env.evaluation_result_list
            if env.iteration - best_balance_iter[0] >= stopping_rounds:
                if log_flag:
                    _log_info('[%d]\t%s' % (env.iteration + 1, log_result))
                if verbose:
                    _log_info('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_balance_iter[0] + 1, '\t'.join([_format_eval_result(x) for x in best_balance_score_list[0]])))
                    if last_metric_only:
                        _log_info("Evaluated only: {}".format(eval_name_splitted[-1]))
                raise EarlyStopException(best_balance_iter[0], best_balance_score[0])
            _final_iteration_check(env, eval_name_splitted, i)
        if log_flag:
            _log_info('[%d]\t%s' % (env.iteration + 1, log_result))
    _callback.order = 30
    return _callback


class FocalLoss:
    def __init__(self, gamma, alpha=None):
        # 使用FocalLoss只需要设定以上两个参数,如果alpha=None,默认取值为1
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        # alpha 参数, 根据FL的定义函数,正样本权重为self.alpha,负样本权重为1 - self.alpha
        if (self.alpha is None) or (self.alpha == 1):
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        # pt和p的关系
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        # 即FL的计算公式
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        # 一阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        # 二阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        # 样本初始值寻找过程
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def get_loss(self, preds, train_data):
        # preds: The predicted values. Predicted values are returned before any transformation,
        # e.g. they are raw margin instead of probability of positive class for binary task.
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def eval_focal_loss(self, preds, train_data):
        # preds: If custom fobj is specified, predicted values are returned before any transformation,
        # e.g. they are raw margin instead of probability of positive class for binary task in this case.
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better


class Objective(object):
    def __init__(self, tuning_param_dict, train_set, monitor, coef_train_val_disparity, callbacks, **kwargs):
        self.tuning_param_dict = tuning_param_dict
        self.train_set = train_set
        self.monitor = monitor
        self.coef_train_val_disparity = coef_train_val_disparity
        self.callbacks = callbacks
        self.kwargs = kwargs

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('boosting')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['boosting'] = eval('trial.suggest_' + suggest_type)('boosting', **suggest_param)
        elif param is not None:
            trial_param_dict['boosting'] = param
        booster = trial_param_dict.get('boosting')

        if self.tuning_param_dict.get('fobj', None) is not None:
            fobj_class = self.tuning_param_dict.get('fobj')[0]
            fobj_trial_param_dict = {}
            for key, param in self.tuning_param_dict.get('fobj')[1].items():
                if isinstance(param, tuple):
                    suggest_type = param[0]
                    suggest_param = param[1]
                    fobj_trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
                else:
                    fobj_trial_param_dict[key] = param
            fobj_instance = fobj_class(**fobj_trial_param_dict)
            fobj = fobj_instance.get_loss
            self.train_set.set_init_score(np.full_like(self.train_set.get_label(),
                                          fobj_instance.init_score(self.train_set.get_label()),
                                          dtype=float))

        for key, param in self.tuning_param_dict.items():
            if (key == 'boosting') or (key == 'fobj'):
                continue
            if (booster is None) or (booster != 'dart'):
                if key in ['uniform_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            else:
                trial_param_dict[key] = param

        callbacks = [callback_class(**callback_param) for callback_class, callback_param in self.callbacks]
        callbacks.append(LightGBMPruningCallback(trial, 'valid '+self.monitor))
        # when folds is defined in xgb.cv, nfold will not be used
        if self.tuning_param_dict.get('fobj', None) is None:
            cvresult = lgb.cv(params=trial_param_dict,
                              train_set=self.train_set,
                              eval_train_metric=True,
                              callbacks=callbacks,
                              **self.kwargs
                              )
        else:
            cvresult = lgb.cv(params=trial_param_dict,
                              train_set=self.train_set,
                              fobj=fobj,
                              eval_train_metric=True,
                              callbacks=callbacks,
                              **self.kwargs
                              )

        n_iterations = len(cvresult['valid ' + self.monitor + '-mean'])
        trial.set_user_attr("n_iterations", n_iterations)

        val_score = cvresult['valid ' + self.monitor + '-mean'][-1]
        train_score = cvresult['train ' + self.monitor + '-mean'][-1]
        trial.set_user_attr("val_score_{}".format(self.monitor), val_score)
        trial.set_user_attr("train_score_{}".format(self.monitor), train_score)
        if self.coef_train_val_disparity > 0:
            best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        else:
            best_score = val_score
        return best_score


class OptunaSearchLGB(object):
    def __init__(self, monitor, optimization_direction='maximize', coef_train_val_disparity=0.2, callbacks=[],
                 n_startup_trials=20, n_warmup_steps=20, interval_steps=1,
                 pruning_percentile=75, maximum_time=60*10, n_trials=100, random_state=2, optuna_verbosity=1):
        self.monitor = monitor
        self.optimization_direction = optimization_direction
        self.coef_train_val_disparity = coef_train_val_disparity
        self.callbacks = callbacks
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.pruning_percentile = pruning_percentile
        self.maximum_time = maximum_time
        self.n_trials = n_trials
        self.random_state = random_state
        self.optuna_verbosity = optuna_verbosity
        self.study = None
        self.dynamic_params = {}
        self.static_params = {}
        self.fobj_dynamic_params = {}
        self.fobj_static_params = {}

    def get_params(self):
        """
        how to use the best params returned by this function:
        train_param = instance.get_params()
        model = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
        test_probability_1d_array = model.predict(test_dmatrix)
        """
        if self.study:
            best_trial = self.study.best_trial
            best_param = best_trial.params

            output_param = {key: best_param[key] for key in best_param.keys() if key in self.dynamic_params}
            output_param.update(self.static_params)

            output_fobj_param = {key: best_param[key] for key in best_param.keys() if key in self.fobj_dynamic_params}
            output_fobj_param.update(self.fobj_static_params)

            return output_param, output_fobj_param, best_trial.user_attrs
        else:
            return None

    def plot_optimization(self):
        if self.study:
            return optuna.visualization.plot_optimization_history(self.study)

    def plot_score(self):
        if self.study:
            trial_df = self.study.trials_dataframe()
            _, ax1 = plt.subplots()
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_train_score,
                     label='train')
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_val_score,
                     label='val')
            plt.legend()
            plt.show()

    def plot_importance(self, names=None):
        if self.study:
            return optuna.visualization.plot_param_importances(self.study, params=names)

    def search(self, params, train_set, **kwargs):
        for key, param in params.items():
            if key == 'fobj':
                for fobj_key, fobj_param in param[1].items():
                    if not isinstance(fobj_param, tuple):
                        self.fobj_static_params[fobj_key] = fobj_param
                    else:
                        self.fobj_dynamic_params[fobj_key] = fobj_param
            else:
                if not isinstance(param, tuple):
                    self.static_params[key] = param
                else:
                    self.dynamic_params[key] = param

        kwargs.pop('callbacks', None)
        kwargs.pop('eval_train_metric', None)
        objective = Objective(params, train_set, self.monitor, self.coef_train_val_disparity, self.callbacks, **kwargs)
        if self.optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=self.pruning_percentile,
                                                 n_warmup_steps=self.n_warmup_steps,
                                                 interval_steps=self.interval_steps,
                                                 n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=self.maximum_time, n_trials=self.n_trials, n_jobs=1)
        self.study = study
        print("Number of finished trials: ", len(study.trials))
