import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import gt, lt

import optuna
with optuna._imports.try_import() as _imports:
    import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMPruningCallback

from lightgbm.basic import _ConfigAliases, _log_info, _log_warning
from lightgbm.callback import EarlyStopException, _format_eval_result

from sklearn.model_selection import PredefinedSplit,  StratifiedKFold
from sklearn.metrics import f1_score, roc_curve

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


class Objective(object):
    def __init__(self,
                 dtrain, cv, folds, metric, optimization_direction,
                 feval, eval_metric_list, coef_train_val_disparity,
                 tuning_param_dict, max_boosting_rounds, early_stop,
                 random_state):
        """ objective will be called once in a trial
        dtrain: xgbdmatrix for train_val
        cv: int, number of stratified folds
        folds:  of list of k tuples (in ,out) for cv, where 'in' is a list of indices into dtrain
                    for training in the ith cv and 'out' is a list of indices into dtrain
                    for validation in the ith cv
                when folds is defined, cv will not be used
        metric: str, the metric to optimize
        optimization_direction: str, 'maximize' or 'minimize'
        feval: callable, custom metric fuction for monitor
        eval_metric_list: a list of str, metrics for monitor
        coef_train_val_disparity: float, coefficient for train_val balance
        tuning_param_dict: dict
        max_boosting_rounds: int, max number of trees in a trial of a set of parameters
        early_stop: int, number of rounds(trees) for early stopping
        random_state: int
        """
        self.dtrain = dtrain
        self.cv = cv
        self.folds = folds
        self.feval = feval
        self.eval_metric_list = eval_metric_list
        self.coef_train_val_disparity = coef_train_val_disparity
        self.metric = 'auc' if metric == 'roc_auc' else metric
        self.max_boosting_rounds = max_boosting_rounds
        self.early_stop_rounds = early_stop
        self.random_state = random_state
        self.maximize = True if optimization_direction == 'maximize' else False
        self.tuning_param_dict = tuning_param_dict
        self.dynamic_params = []
        self.static_params = []

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('boosting')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['boosting'] = eval('trial.suggest_' + suggest_type)('boosting', **suggest_param)
            self.dynamic_params.append('boosting')
        elif param is not None:
            trial_param_dict['boosting'] = param
            self.static_params.append('boosting')
        booster = trial_param_dict.get('boosting')
        for key, param in self.tuning_param_dict.items():
            if key == 'boosting':
                continue
            if (booster is None) or (booster != 'dart'):
                if key in ['uniform_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
                self.dynamic_params.append(key)
            else:
                trial_param_dict[key] = param
                self.static_params.append(key)

        earlystopping = my_early_stopping(self.early_stop_rounds,
                                          self.coef_train_val_disparity,
                                          verbose=True if self.tuning_param_dict.get('verbosity', 0) > 0 else False)
        pruning = LightGBMPruningCallback(trial, 'valid '+self.metric)

        # when folds is defined in xgb.cv, nfold will not be used
        cvresult = lgb.cv(params=trial_param_dict,
                          train_set=self.dtrain,
                          num_boost_round=self.max_boosting_rounds,
                          folds=self.folds,
                          nfold=self.cv,
                          metrics=self.eval_metric_list,
                          feval=self.feval,
                          seed=self.random_state,
                          eval_train_metric=True,
                          shuffle=False,
                          stratified=False,
                          callbacks=[pruning, earlystopping],
                          )
        n_iterations = len(cvresult['valid ' + self.metric + '-mean'])
        val_score = cvresult['valid ' + self.metric + '-mean'][-1]
        train_score = cvresult['train ' + self.metric + '-mean'][-1]
        trial.set_user_attr("n_iterations", n_iterations)
        trial.set_user_attr("val_score", val_score)
        trial.set_user_attr("train_score", train_score)
        if self.coef_train_val_disparity > 0:
            best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        else:
            best_score = val_score
        return best_score


class OptunaSearchLGB(object):
    def __init__(self):
        self.study = None
        self.static_param = {}
        self.dynamic_param = {}

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
            best_param.update(best_trial.user_attrs)
            best_param.update(self.static_param)
            return best_param
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

    def search(self, x_train, y_train, tuning_param_dict, cv=3,
               eval_set=None, eval_metric=['roc_auc', 'ks'], optimization_direction='maximize',
               coef_train_val_disparity=0.2, maximum_boosting_rounds=1000, early_stopping_rounds=10,
               n_startup_trials=20, n_warmup_steps=20, interval_steps=1, pruning_percentile=75,
               maximum_time=60*10, n_trials=100, random_state=2, optuna_verbosity=1):
        for key, param in tuning_param_dict.items():
            if not isinstance(param, tuple):
                self.static_param[key] = param
            else:
                self.dynamic_param[key] = param

        feval, eval_metric_list = regularize_metric(eval_metric)
        x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train,
                                                             eval_set, cv,
                                                             random_state)
        xgb_dmatrix = lgb.Dataset(x_train_val, label=y_train_val)
        objective = Objective(dtrain=xgb_dmatrix,
                              cv=cv,
                              folds=folds,
                              metric=eval_metric[-1],
                              optimization_direction=optimization_direction,
                              feval=feval,
                              eval_metric_list=eval_metric_list,
                              coef_train_val_disparity=coef_train_val_disparity,
                              tuning_param_dict=tuning_param_dict,
                              max_boosting_rounds=maximum_boosting_rounds,
                              early_stop=early_stopping_rounds,
                              random_state=random_state)
        if optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=pruning_percentile,
                                                 n_warmup_steps=n_warmup_steps,
                                                 interval_steps=interval_steps,
                                                 n_startup_trials=n_startup_trials)
        study = optuna.create_study(direction=optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=maximum_time, n_trials=n_trials, n_jobs=1)
        self.study = study
        print("Number of finished trials: ", len(study.trials))


def test_optuna():
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    op = OptunaSearchLGB()
    tuning_param_dict = {'objective': 'binary',
                         'verbosity': -1,
                         'seed': 2,
                         'deterministic': True,
                         'boosting': ('categorical', {'choices': ['gbdt', 'dart']}),
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'min_gain_to_split': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 0.95, 'q': 0.05}),
                         'subsample_freq': ('int', {'low': 1, 'high': 10}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'uniform_drop': ('categorical', {'choices': [True, False]})}
    op.search(x_train, y_train, tuning_param_dict, cv=1, coef_train_val_disparity=0,
              eval_set=[(x_valid, y_valid)], optuna_verbosity=1, early_stopping_rounds=10,
              n_warmup_steps=5, n_trials=20)
    train_dmatrix = lgb.Dataset(x_train, label=y_train)
    test_dmatrix = lgb.Dataset(x_valid, label=y_valid)
    train_param = op.get_params()
    print(train_param)
    afsxc = lgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'],
                      valid_sets=[train_dmatrix, test_dmatrix], valid_names=['train', 'val'], feval=custom_ks_score)
    train_pred = afsxc.predict(x_train)
    print("train_data_ks", ks_stats(y_train, train_pred))
    test_pred = afsxc.predict(x_valid)
    print("test_data_ks", ks_stats(y_valid, test_pred))
    op.plot_optimization().show()
    op.plot_importance(list(op.study.trials[0].params.keys())).show()
    op.plot_score()


if __name__ == '__main__':
    test_optuna()
