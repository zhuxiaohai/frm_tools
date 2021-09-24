import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
with optuna._imports.try_import() as _imports:
    import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.callback import TrainingCallback
from xgboost.core import XGBoostError, Booster
from xgboost import rabit
from sklearn.model_selection import GridSearchCV, PredefinedSplit,  StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, make_scorer
from bayes_opt import BayesianOptimization
from .utils import cal_ks, cal_psi_score
import warnings
warnings.filterwarnings('ignore')


def train_val_score(train_score, val_score, w):
    output_scores = val_score - abs(train_score - val_score) * w
    return output_scores


def custom_f1_score(y_pred, y_true_dmatrix):
    """
    the signature is func(y_predicted, DMatrix_y_true) where DMatrix_y_true is
    a DMatrix object such that you may need to call the get_label method.
    It must return a (str, value) pair where the str is a name for the evaluation
    and value is the value of the evaluation function.
    The callable function is always minimized.
    :param y_pred: np.array, probability score predicted by the xgbclassifier
    :param y_true_dmatrix: xgb DMatrix, true label, with positive instances as 1
    """
    y_true = y_true_dmatrix.get_label()
    y_hat = np.zeros_like(y_pred)
    y_hat[y_pred > 0.5] = 1
    f1 = f1_score(y_true, y_hat)
    return 'f1', f1


def custom_ks_score(y_pred, y_true_dmatrix):
    def ks_stats(y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value
    y_true = y_true_dmatrix.get_label()
    ks = ks_stats(y_true, y_pred)
    return 'ks', ks


def regularize_metric(eval_metric):
    """
    :param eval_metric: list of str, each str should be a built-in metric of sklearn
    """
    eval_metric_list = []
    feval = None
    for metric in eval_metric:
        if metric == 'f1':
            # when this happens, feval will always be used for early stopping.
            # For custom function, you can specify the optimization direction in xgb api,
            # but for built in metrics, xgb will automatically decide
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


def select_with_feature_importance(train_data,  test_data, var_names, target,
                                   random_param, oot_data=None, step=0.8,
                                   eval_metric=['roc_auc', 'ks'], maximize=True,
                                   train_early_stopping=10, importance_type='gain',
                                   max_num_vars=50, verbose_train=True, random_state=27):
    feval, eval_metric_list = regularize_metric(eval_metric)
    kept_var_names = var_names
    count = 0
    result = pd.DataFrame()
    np.random.seed(random_state)
    while True:
        current_param = {}
        for key, value in random_param.items():
            current_param[key] = np.random.choice(value, 1, replace=False)[0]
        current_param['eval_metric'] = eval_metric_list
        print('starting iter: ', count)
        # 模型训练
        train_dmatrix = xgb.DMatrix(train_data[kept_var_names], train_data[target])
        test_dmatrix = xgb.DMatrix(test_data[kept_var_names], test_data[target])
        evals = [(train_dmatrix, 'train'), (test_dmatrix, 'test')]
        if oot_data is not None:
            oot_dmatrix = xgb.DMatrix(oot_data[kept_var_names], oot_data[target])
            evals.insert(1, (oot_dmatrix, 'oot'))
        model = xgb.train(current_param, train_dmatrix,
                          num_boost_round=1000,
                          evals=evals,
                          feval=feval,
                          maximize=maximize,
                          verbose_eval=verbose_train,
                          early_stopping_rounds=train_early_stopping)
        # 训练集KS
        train_prediction = model.predict(train_dmatrix, ntree_limit=model.best_ntree_limit)
        trainks = cal_ks(train_prediction, train_data[target])[0]
        dic = {"trainks": float(trainks)}
        # 测试集KS和PSI
        test_prediction = model.predict(test_dmatrix, ntree_limit=model.best_ntree_limit)
        testks = cal_ks(test_prediction, test_data[target])[0]
        testpsi = cal_psi_score(test_prediction, train_prediction)
        dic.update({"testks": float(testks), "testpsi": float(testpsi)})
        # 跨时间验证集KS和PSI
        if oot_data is not None:
            oot_prediction = model.predict(oot_dmatrix, ntree_limit=model.best_ntree_limit)
            ootks = cal_ks(oot_prediction, oot_data[target])[0]
            ootpsi = cal_psi_score(oot_prediction, train_prediction)
            dic.update({"ootks": float(ootks), "ootpsi": float(ootpsi)})
        result = result.append(pd.DataFrame(dic, index=[len(kept_var_names)]))
        # 得到特征重要性
        scores_dict = model.get_score(importance_type=importance_type)
        for var in kept_var_names:
            if var not in scores_dict:
                scores_dict[var] = 0
        scores = pd.Series(scores_dict)
        scores = scores.sort_values(ascending=False) / scores.sum()
        print("current result: ", dic)
        if scores.shape[0] >= max_num_vars:
            num_kept_vars = int(step * scores.shape[0])
            kept_var_names = list(scores.iloc[:num_kept_vars].index)
            print('deleting features: {} --> {}'.format(scores.shape[0], len(kept_var_names)))
            count += 1
        else:
            print("over")
            break
        if count > 1:
            legend_names = []
            for col in result.columns:
                if col.find('psi') < 0:
                    legend_names.append(col)
                    plt.plot(result.index, result[col])
            plt.legend(legend_names)
            plt.show()
            plt.pause(0.2)

    return scores


class MyEvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    rank : int
        Which worker should be used for printing the result.
    period : int
        How many epoches between printing.
    w: float
        coefficient to balance between train and eval
    show_stdv : bool
        Used in cv to show standard deviation.  Users should not specify it.
    '''
    def __init__(self, rank=0, period=1, w=0.2, show_stdv=False):
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        self.w = w
        self.metric = None
        assert period > 0
        # last error message, useful when early stopping and period are used together.
        self._latest = None
        super().__init__()

    def _fmt_metric(self, data, metric, score, std):
        if std is not None and self.show_stdv:
            msg = '\t{0}:{1:.5f}+{2:.5f}'.format(data + '-' + metric, score, std)
        else:
            msg = '\t{0}:{1:.5f}'.format(data + '-' + metric, score)
        return msg

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False
        warn = 'Must have 2 datasets for early stopping, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, warn

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]
            print('Using {} for train_val balance'.format(self.metric))

        msg = '[{}]'.format(epoch)
        if rabit.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                        stdv = None
                    msg += self._fmt_metric(data, metric_name, score, stdv)
            data_names = list(evals_log.keys())
            if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
                train_score, _ = evals_log[data_names[0]][self.metric][-1]
                val_score, _ = evals_log[data_names[1]][self.metric][-1]
            else:
                train_score = evals_log[data_names[0]][self.metric][-1]
                val_score = evals_log[data_names[1]][self.metric][-1]
            score = train_val_score(train_score, val_score, self.w)
            msg += '\t total-{0}: {1:.5f}'.format(self.metric, score)
            msg += '\n'

            if (epoch % self.period) == 0 or self.period == 1:
                rabit.tracker_print(msg)
                self._latest = None
            else:
                # There is skipped message
                self._latest = msg
        return False

    def after_training(self, model):
        if rabit.get_rank() == self.printer_rank and self._latest is not None:
            rabit.tracker_print(self._latest)
        return model


class MyEarlyStopping(TrainingCallback):
    ''' Callback function for xgb cv early stopping
    Parameters
    ----------
    rounds : int
        Early stopping rounds.
    w ： float
        Coefficient to balance the performance between train and val
    maximize : bool
        Whether to maximize evaluation metric.  None means auto (discouraged).
    mode: str
        'cv' or 'train'
    save_best : bool
        Whether training should return the best model or the last model.
    '''

    def __init__(self, rounds, w=0.2, maximize=None, mode='cv', save_best=False):
        self.metric = None
        self.rounds = rounds
        self.w = w
        self.save_best = save_best if mode == 'train' else False
        if maximize:
            self.improve_op = lambda x, y: x > y
        else:
            self.improve_op = lambda x, y: x < y
        self.stopping_history = {}
        self.current_rounds = 0
        self.best_scores = {}
        super().__init__()

    def _update_rounds(self, score, model: Booster, epoch):
        if not self.stopping_history:
            # First round
            self.current_rounds = 0
            self.stopping_history = [score]
            self.best_scores = [score]
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        elif not self.improve_op(score, self.best_scores[-1]):
            # Not improved
            self.stopping_history.append(score)
            self.current_rounds += 1
        else:
            # Improved
            self.stopping_history.append(score)
            self.best_scores.append(score)
            record = self.stopping_history[-1]
            model.set_attr(best_score=str(record), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(self, model: Booster, epoch, evals_log):
        msg = 'Must have 2 datasets for early stopping, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, msg

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]
            print('Using {} for train_val balance'.format(self.metric))

        data_names = list(evals_log.keys())
        if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
            train_score, _ = evals_log[data_names[0]][self.metric][-1]
            val_score, _ = evals_log[data_names[1]][self.metric][-1]
        else:
            train_score = evals_log[data_names[0]][self.metric][-1]
            val_score = evals_log[data_names[1]][self.metric][-1]
        score = train_val_score(train_score, val_score, self.w)
        return self._update_rounds(score, model, epoch)

    def after_training(self, model: Booster):
        try:
            if self.save_best:
                model = model[: int(model.attr('best_iteration')) + 1]
        except XGBoostError as e:
            raise XGBoostError('`save_best` is not applicable to current booster') from e
        return model


class XGBoostPruningCallback(TrainingCallback):
    def __init__(self, trial: optuna.trial.Trial) -> None:
        _imports.check()
        self._trial = trial
        self.metric = None
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        msg = 'Must have 2 datasets for pruning, the first for train and the other for eval.'
        assert len(evals_log.keys()) == 2, msg

        if not self.metric:
            data_name = list(evals_log.keys())[-1]
            data_log = evals_log[data_name]
            self.metric = list(data_log.keys())[-1]

        data_names = list(evals_log.keys())
        if isinstance(evals_log[data_names[0]][self.metric][-1], tuple):
            val_score, _ = evals_log[data_names[1]][self.metric][-1]
        else:
            val_score = evals_log[data_names[1]][self.metric][-1]
        current_score = val_score
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at iteration {}.".format(epoch)
            raise optuna.TrialPruned(message)


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

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('booster')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['booster'] = eval('trial.suggest_' + suggest_type)('booster', **suggest_param)
        elif param is not None:
            trial_param_dict['booster'] = param
        booster = trial_param_dict.get('booster')
        for key, param in self.tuning_param_dict.items():
            if key == 'booster':
                continue
            if (booster is None) or booster == 'gbtree':
                if key in ['sample_type', 'normalize_type', 'one_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            else:
                trial_param_dict[key] = param

        monitor = MyEvaluationMonitor(rank=0,
                                      period=1,
                                      w=self.coef_train_val_disparity,
                                      show_stdv=True)
        earlystopping = MyEarlyStopping(rounds=self.early_stop_rounds,
                                        w=self.coef_train_val_disparity,
                                        maximize=self.maximize,
                                        save_best=False)
        pruning = XGBoostPruningCallback(trial)

        # when folds is defined in xgb.cv, nfold will not be used
        cvresult = xgb.cv(params=trial_param_dict,
                          dtrain=self.dtrain,
                          num_boost_round=self.max_boosting_rounds,
                          early_stopping_rounds=self.early_stop_rounds,
                          nfold=self.cv,
                          folds=self.folds,
                          feval=self.feval,
                          metrics=self.eval_metric_list,
                          maximize=self.maximize,
                          seed=self.random_state,
                          as_pandas=True,
                          callbacks=[monitor, pruning, earlystopping]
                          if self.tuning_param_dict.get('verbosity', 0) > 0 else [pruning, earlystopping])
        train_score = cvresult['train-' + self.metric + '-mean'].iloc[-1]
        val_score = cvresult['test-' + self.metric + '-mean'].iloc[-1]
        best_score = train_val_score(train_score, val_score, self.coef_train_val_disparity)
        num_parallel_tree = trial_param_dict.get('num_parallel_tree', 1)
        trial.set_user_attr("n_iterations", cvresult.shape[0])
        trial.set_user_attr("n_estimators", cvresult.shape[0]*num_parallel_tree)
        trial.set_user_attr("train_score", train_score)
        trial.set_user_attr("val_score", val_score)
        return best_score


class OptunaSearchXGB(object):
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
               maximum_time=60*10, n_trials=100, n_jobs=-1, random_state=2, optuna_verbosity=1):
        for key, param in tuning_param_dict.items():
            if not isinstance(param, tuple):
                self.static_param[key] = param
            else:
                self.dynamic_param[key] = param

        feval, eval_metric_list = regularize_metric(eval_metric)
        x_train_val, y_train_val, cv, folds = make_train_val(x_train, y_train,
                                                             eval_set, cv,
                                                             random_state)
        xgb_dmatrix = xgb.DMatrix(x_train_val, label=y_train_val)
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
        study.optimize(objective, timeout=maximum_time, n_trials=n_trials, n_jobs=n_jobs)
        self.study = study
        print("Number of finished trials: ", len(study.trials))


class GridSearchXGB(object):
    def __init__(self, alg):
        """
        :param alg: XGBClassifier object
          please note that the alg that is used to initialize the class instance will be changed
          after gridsearch because python uses reference of address like that in C++
        """
        self._alg = alg

    def get_params(self):
        return self._alg.get_params()

    def set_params(self, **setting_params):
        self._alg.set_params(**setting_params)

    def predict(self, x):
        """
        :param x: array-like(dim=2), Feature matrix
        """
        # np.array
        # defaults to best_ntree_limit if early stopping is available,
        # otherwise 0 (use all trees).
        prediction = self._alg.predict(x)
        return prediction

    def predict_proba(self, x):
        """
        :param x: array-like(dim=2), Feature matrix
        """
        # np.array
        # defaults to best_ntree_limit if early stopping is available,
        # otherwise 0 (use all trees).
        prediction = self._alg.predict_proba(x)
        return prediction

    def predict_proba_with_best_threshold(self, x, y, metric_list=['f1']):
        """
        tune threshold with best threshold
        :param x: array-like(dim=2), Feature matrix
        :param y: array-like(dim=1), Label
        :param metric_list: a list of metrics
        """
        max_score_list = []
        best_t_list = []
        for metric in metric_list:
            # auc, ks这种只需要有排序性，计算过程是不需要阈值的
            assert metric in ['f1', 'accuracy']
            max_score_list.append([metric, -1])
            best_t_list.append([metric, -1])
        y_prediction = self.predict_proba(x)[:, 1]
        for t in np.arange(0, 1, 0.05):
            y_hat = np.zeros_like(y_prediction)
            y_hat[y_prediction >= t] = 1
            for i in range(len(metric_list)):
                max_score = max_score_list[i][1]
                metric = max_score_list[i][0]
                if metric == 'f1':
                    score = f1_score(y, y_hat)
                elif metric == 'accuracy':
                    score = accuracy_score(y, y_hat)
                else:
                    raise RuntimeError('not-supported metric: {}'.format(metric))
                if score > max_score:
                    max_score_list[i][1] = score
                    best_t_list[i][1] = t

        return y_prediction, max_score_list, best_t_list

    def fit(self, x_train, y_train, param_dict=None):
        if param_dict is not None:
            self._alg = XGBClassifier(**param_dict)
        self._alg.fit(x_train, y_train)

    def _custom_f1_score(self, y_pred, y_true_dmatrix):
        """
        the signature is func(y_predicted, DMatrix_y_true) where DMatrix_y_true is
        a DMatrix object such that you may need to call the get_label method.
        It must return a (str, value) pair where the str is a name for the evaluation
        and value is the value of the evaluation function.
        The callable function is always minimized.
        :param y_pred: np.array, probability score predicted by the xgbclassifier
        :param y_true_dmatrix: xgb DMatrix, true label, with positive instances as 1
        """
        y_true = y_true_dmatrix.get_label()
        y_hat = np.zeros_like(y_pred)
        y_hat[y_pred > 0.5] = 1
        f1 = f1_score(y_true, y_hat)
        return 'f1_err', 1 - f1

    def _ks_stats(self, y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value

    def _custom_ks_score(self, y_pred, y_true_dmatrix):
        y_true = y_true_dmatrix.get_label()
        ks = self._ks_stats(y_true, y_pred)
        return 'ks-err', 1 - ks

    def _regularize_metric(self, eval_metric, early_stopping_rounds):
        """
        :param eval_metric: list of str, each str should be a built-in metric of sklearn
        :param early_stopping_rounds: int or None
        """
        feval = None
        if early_stopping_rounds is not None:
            eval_metric_list = []
            for metric in eval_metric:
                if metric == 'f1':
                    # when this happens, feval will always be used for early stopping.
                    # For custom function/metrics not like 'auc***', the smaller the returned value the better,
                    # but for built in metrics, the cv package will automatically decide
                    # according to the metric type, e.g., the smaller the better for error
                    # but the bigger the better for auc, etc.
                    feval = self._custom_f1_score
                elif metric == 'accuracy':
                    # It is calculated as #(wrong cases)/#(all cases).
                    # The evaluation will regard the instances
                    # with prediction value larger than 0.5 as positive instances, i.e., 1 instances)
                    eval_metric_list.append('error')
                elif metric == 'roc_auc':
                    eval_metric_list.append('auc')
                elif metric == 'ks':
                    # same logic as f1
                    feval = self._custom_ks_score
                else:
                    raise RuntimeError('not-supported metric: {}'.format(metric))
        else:
            eval_metric_list = {}
            # 指定needs_threshold的话，则gridsearchcv或者cross_validate里面的estimator的predict_proba
            # 会被调用，将概率结果传给make_scorer里面的score_function, 计算结果返回一个float
            # 没有指定任何东西的话，则gridsearchcv或者cross_validate里面的estimator的predict
            # 会被调用，返回一个预测类别标签传给make_scorer里面的score_function, 计算结果返回一个float
            for metric in eval_metric:
                if metric == 'f1':
                    eval_metric_list['f1'] = make_scorer(f1_score)
                elif metric == 'accuracy':
                    eval_metric_list['accuracy'] = make_scorer(accuracy_score)
                elif metric == 'roc_auc':
                    eval_metric_list['roc_auc'] = make_scorer(roc_auc_score, needs_threshold=True)
                elif metric == 'ks':
                    eval_metric_list['ks'] = make_scorer(self._ks_stats, needs_threshold=True)
                else:
                    raise RuntimeError('not-supported metric: {}'.format(metric))
        return feval, eval_metric_list

    def _regularize_xgb_params(self, **params_to_optimize):
        regularized_dict = {}
        for key, value in self._alg.get_params().items():
            if key in params_to_optimize.keys():
                if key in ['max_depth', 'n_estimators', 'min_child_weight',
                           'max_delta_step', 'num_parallel_tree']:
                    regularized_dict[key] = int(params_to_optimize[key])
                else:
                    regularized_dict[key] = params_to_optimize[key]
            else:
                regularized_dict[key] = value
        return regularized_dict

    def _make_train_val(self, x_train, y_train, eval_set, cv):
        """
        :param x_train: array-like(dim=2), Feature matrix
        :param y_train: array-like(dim=1), Label
        :param eval_set: a list of (X, y) tuples, only one tuple is supported as of now
        :param cv: int, number of folds for cross_validation
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
                                  random_state=self._alg.get_params()['random_state'])
            folds = []
            for train_indices_array, val_indices_array in skf.split(
                    x_train_val, y_train_val):
                folds.append((train_indices_array.tolist(),
                              val_indices_array.tolist()))
        return x_train_val, y_train_val, cv, folds

    def search_with_bayesian(self, x_train, y_train, tuning_dict,
                             eval_set=None, eval_metric=['f1'], cv=3, verbose=True,
                             n_gs_jobs=1,
                             init_points=5, n_iter=25, acq='ucb', kappa=2.576,
                             xi=0.0, **gp_params):
        """
        :param x_train: array-like(dim=2), Feature matrix
        :param y_train: array-like(dim=1), Label
        :param tuning_dict: dict with keys being params in XGBClassifier,
         and values being a list of lowerbound and upperbound
        :param eval_set: a list of (X, y) tuples, only one tuple is supported as of now
        :param eval_metric: a list of str, but only the last will be used
        :param cv: int, number of folds for cross_validation
        :param verbose: bool
        :param n_gs_jobs: int, number of jobs for cross_validation
        :param init_points: int, number of initial points for bayesian optimization
        :param n_iter: int, number of other points for bayesian optimization besides init_points
        :param acq: 'str', acquisition function used in exporation stage of bayesian,
          must be in ['ucb', 'ei', 'poi']
        :param kappa: param used in acquisition function. As of 'ucb' function, for example,
          the bigger kappas is, the more likely our bayesian is gonna search the unknown spaces
        :param xi: param used in acquisition function
        :param gp_params: param used in gaussian process
        """
        feval, eval_metric_list = self._regularize_metric(eval_metric, None)
        x_train_val, y_train_val, cv, folds = self._make_train_val(x_train, y_train, eval_set, cv)

        # use closure to pass in x_train_val and y_train_val, scoring and n_jobs, cv
        # because the params of the objective function can only be the params to optimize
        def rf_cv(**params_to_optimize):
            xgb_param = self._regularize_xgb_params(**params_to_optimize)
            # cross_val_score returns an array with each element being the val score
            # on each fold
            val = cross_val_score(XGBClassifier(**xgb_param),
                                  x_train_val, y_train_val,
                                  scoring=eval_metric_list[eval_metric[-1]],
                                  n_jobs=n_gs_jobs,
                                  cv=folds).mean()
            return val
        rf_bo = BayesianOptimization(f=rf_cv, pbounds=tuning_dict,
                                     random_state=self._alg.get_params()['random_state'],
                                     verbose=2 if verbose else 0)
        rf_bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq,
                       kappa=kappa, xi=xi, **gp_params)
        optimized_xgb_params = self._regularize_xgb_params(**rf_bo.max['params'])
        # rf_bo.maximize will not get self._alg fitted
        # please note that the alg that is used to initialize the class instance is also changed
        # because python uses reference of address like that in C++
        self._alg.set_params(**optimized_xgb_params)
        print('best tuning: {} with val score {}'.format(rf_bo.max['params'],
                                                         rf_bo.max['target']))
        print('current model: {}'.format(self._alg.get_params()))
        return self._alg.get_params()

    def search(self, x_train, y_train, tuning_dict, eval_set=None, eval_metric=['f1'], cv=3,
               n_gs_jobs=1, early_stopping_rounds=None, coef_train_val_disparity=0.2, verbose=True):
        """
        :param x_train: array-like(dim=2), Feature matrix
        :param y_train: array-like(dim=1), Labels
        :param tuning_dict: dict with keys being params in XGBClassifier,
         and values being a list of grid point to probe
        :param eval_set: a list of (X, y) tuples, only one tuple is supported as of now
        :param eval_metric: a list of str.
        :param cv: int, number of folds for cross_validation
        :param n_gs_jobs: int, number of jobs used in grid searching
        :param early_stopping_rounds: int, Validation metric needs to improve at least once
         in every 'early_stopping_rounds' round(s) to continue training.
         Note that we use this parameter for the search of best number of estimators so all the other
         parameters will be kept unchanged
         If there’s more than one metric in eval_metric, the last metric will be used for early stopping
         except one case, which is f1 or ks are/is in eval_metric list. In that case,
         f1 or ks will always be used for early stopping depending on which one comes latter
        :param coef_train_val_disparity: float, coef to consider the disparity of train and val
        :param verbose: bool, If verbose, prints the evaluation metric measured on the validation set.
        """
        feval, eval_metric_list = self._regularize_metric(eval_metric, early_stopping_rounds)
        if early_stopping_rounds is not None:
            print('warning: only the number of estimators will be increased '
                  'until early stopping criteria'
                  'is met and the tuning grid will not be used')
            x_train_val, y_train_val, cv, folds = self._make_train_val(x_train, y_train, eval_set, cv)
            xgb_dmatrix = xgb.DMatrix(x_train_val, label=y_train_val)
            xgb_param = self._alg.get_xgb_params()
            # when folds is defined in xgb.cv, nfold will not be used
            cvresult = xgb.cv(xgb_param, xgb_dmatrix,
                              num_boost_round=self._alg.get_params()['n_estimators'],
                              nfold=cv,
                              metrics=eval_metric_list,
                              feval=feval,
                              folds=folds,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=True if verbose else None,
                              seed=self._alg.get_params()['random_state'])
            # returns the best n_estimators
            # please note that the alg that is used to initialize the class instance is also changed
            # because python uses reference of address like that in C++
            self._alg.set_params(n_estimators=cvresult.shape[0])
            print('best n_estimators: {} with val score {}'.format(self._alg.get_params()['n_estimators'],
                                                                   cvresult.iloc[-1, :][-2]))
            print('current model: {}'.format(self._alg.get_params()))
            return self._alg.get_params()
        else:
            x_train_val, y_train_val, cv, folds = self._make_train_val(x_train, y_train, eval_set, cv)
            grid_search_params = {'estimator': self._alg,
                                  'param_grid': tuning_dict,
                                  'scoring': eval_metric_list,
                                  'cv': folds,
                                  'verbose': 2 if verbose else 0,
                                  'refit': eval_metric[-1],
                                  'n_jobs': n_gs_jobs,
                                  'return_train_score': True}
            grsearch = GridSearchCV(**grid_search_params)
            grsearch.fit(x_train_val, y_train_val)
            # after grsearch-fitting, the estimator in GridSearchCV will not get fitted
            # please note that the alg that is used to initialize the class instance is also changed
            # because python uses reference of address like that in C++
            if coef_train_val_disparity > 0:
                best_param, best_score, cv_results = self.postprocess_cv_results(
                    grsearch.cv_results_, cv, eval_metric[-1], coef_train_val_disparity)
            else:
                best_param = grsearch.best_params_
                best_score = grsearch.best_score_
                cv_results = grsearch.cv_results_
            self._alg.set_params(**best_param)
            print('best tuning: {} with val score {}'.format(best_param,
                                                             best_score))
            print('current model: {}'.format(self._alg.get_params()))
            return best_param, cv_results

    def postprocess_cv_results(self, cv_results, cv, metric, coef_train_val_disparity=0.2):
        num_param_combo = len(cv_results['params'])
        score_array = np.array([None] * num_param_combo)
        for i in range(num_param_combo):
            train_scores = np.array([None]*cv)
            val_scores = np.array([None]*cv)
            for j in range(cv):
                train_scores[j] = cv_results['split' + str(j) + '_train_' + metric][i]
                val_scores[j] = cv_results['split' + str(j) + '_test_' + metric][i]
            score_array[i] = self.train_val_score(train_scores, val_scores, coef_train_val_disparity)
        cv_results.update({'combined_scores': score_array})
        best_index = score_array.argmax()
        best_param = cv_results['params'][best_index]
        best_score = score_array[best_index]
        return best_param, best_score, cv_results

    def train_val_score(self, train_scores, val_scores, w=0.2):
        """
        Parameters
        ----------
        train_scores: np.array
        val_scores: np.array
        w:float
        Returns: float
        -------
        """
        output_scores = val_scores - abs(train_scores - val_scores) * w
        return output_scores.mean()

    def bidirectional_search(self, x_train, y_train, cv, tuning_dict,
                             eval_set=None, eval_metric=['f1'],
                             n_gs_jobs=1, coef_train_val_disparity=0.2, verbose=True):
        """
        x_train: pd.DataFrame, dim=2
        y_train: pd.Series, dim=1
        cv:  int
        tuning_dict: dict with keys being the feature to be tuned and values being the list of steps, e.g., [-1, -0.5, 1]
        eval_set:  list of (X, y) tuples, only one tuple is supported as of now
        eval_metric: a list of str.
        n_gs_jobs: int, number of jobs to run in parallel
        coef_train_val_disparity: float, coefficient to consider to disparity between train and cal scores
        verbose: Bool, if to print the cv process
        """
        _, eval_metric_list = self._regularize_metric(eval_metric, None)
        x_train_val, y_train_val, cv, folds = self._make_train_val(x_train, y_train, eval_set, cv)
        params = self._alg.get_params()
        cv_results = cross_validate(self._alg, x_train_val, y_train_val,
                                    cv=folds, return_train_score=True,
                                    scoring=eval_metric_list, n_jobs=n_gs_jobs,
                                    verbose=2 if verbose else 0)
        if coef_train_val_disparity > 0:
            train_scores = cv_results['train_' + eval_metric[-1]]
            val_scores = cv_results['test_' + eval_metric[-1]]
            best_score = self.train_val_score(train_scores, val_scores)
        else:
            val_scores = cv_results['test_' + eval_metric[-1]]
            best_score = val_scores.mean()
        best_param_record = {}
        for key in tuning_dict.keys():
            best_param_record[key] = params[key]
        print('original params {} with val score {}'.format(best_param_record, best_score))
        print('stepwise searching begins...')
        for (param, steps) in tuning_dict.items():
            print('param: {}, original value: {}'.format(param, params[param]))
            for step in steps:
                print('step ', step)
                while True:
                    if params[param] + step > 0:
                        params[param] += step
                        print('-->: ', params[param])
                        self._alg.set_params(**params)
                        cv_results = cross_validate(self._alg, x_train_val, y_train_val,
                                                    cv=folds, return_train_score=True,
                                                    scoring=eval_metric_list, n_jobs=n_gs_jobs,
                                                    verbose=2 if verbose else 0)
                        if coef_train_val_disparity > 0:
                            train_scores = cv_results['train_'+eval_metric[-1]]
                            val_scores = cv_results['test_'+eval_metric[-1]]
                            output_score = self.train_val_score(train_scores, val_scores, coef_train_val_disparity)
                        else:
                            val_scores = cv_results['test_' + eval_metric[-1]]
                            output_score = val_scores.mean()
                        if best_score < output_score:
                            best_score = output_score
                            best_param_record[param] = params[param]
                            print('better val score {}'.format(best_score))
                        else:
                            params[param] -= step
                            break
                    else:
                        break
        self._alg.set_params(**params)
        print('best tuning: {} with val score {}'.format(best_param_record, best_score))

        return params


def test_tunning_grid():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...dddd')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        reg_lambda=1,
                        reg_alpha=0,
                        n_jobs=4,
                        scale_pos_weight=1,
                        random_state=27)
    tuning_grid = {'gamma': [0.1, 0.2]}
    gs = GridSearchXGB(alg)
    best_param = gs.search(x_train, y_train, tuning_grid)
    print('search result')
    print(best_param)
    print('current result')
    print(gs.get_params())


def test_early_stopping():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        reg_lambda=1,
                        reg_alpha=0,
                        n_jobs=4,
                        scale_pos_weight=1,
                        random_state=27)
    tuning_grid = {'gamma': [0.1, 0.2]}
    gs = GridSearchXGB(alg)
    best_param = gs.search(x_train, y_train, tuning_grid,
                           [(x_test, y_test)], ['f1', 'roc_auc'],
                           early_stopping_rounds=3)
    print('search result')
    print(best_param)
    print('recheck current result')
    print(gs.get_params())


def test_consecutive_tuning_and_predict():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        reg_lambda=1,
                        reg_alpha=0,
                        n_jobs=1,
                        scale_pos_weight=1,
                        random_state=27)
    tuning_grid = {'max_depth': [2, 10]}
    gs = GridSearchXGB(alg)
    best_param = gs.search(x_train, y_train, tuning_grid,
                           eval_set=[(x_test, y_test)], eval_metric=['ks', 'f1'],
                           early_stopping_rounds=3)
    print('after n_estimators search is completed...')
    print(gs.get_params())
    best_param = gs.search(x_train, y_train, tuning_grid,
                           eval_set=[(x_test, y_test)], eval_metric=['ks', 'f1'])
    print('after gamma_search is completed...')
    print(gs.get_params())
    print('returned best param:')
    print(best_param[0])
    print('cv process')
    print(best_param[1])
    gs.fit(x_train, y_train)
    print('model after fit...')
    print(gs.get_params())
    prediction, best_score_list, best_threshold_list = \
        gs.predict_proba_with_best_threshold(x_test, y_test, ['f1', 'accuracy'])
    print(best_score_list)
    print(best_threshold_list)


def test_bidirectional():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        reg_lambda=1,
                        reg_alpha=0,
                        n_jobs=1,
                        scale_pos_weight=1,
                        random_state=27)
    tuning_dict = {'max_depth': [-2, -1, 1, 2], 'gamma': [-1, 1]}
    gs = GridSearchXGB(alg)
    best_param = gs.bidirectional_search(x_train, y_train, eval_metric=['ks', 'f1'], cv=3,
                                         tuning_dict=tuning_dict, verbose=False, coef_train_val_disparity=0)
    print('after searching, the model is...')
    print(gs.get_params())
    print('returned best param:')
    print(best_param)


def test_bayesian():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        reg_lambda=1,
                        reg_alpha=0,
                        n_jobs=4,
                        scale_pos_weight=1,
                        random_state=27)
    tuning_grid = {'learning_rate': [0.11, 0.3], 'n_estimators': [10, 200]}
    gs = GridSearchXGB(alg)
    best_param = gs.search_with_bayesian(x_train, y_train, tuning_dict=tuning_grid,
                                         eval_set=[(x_test, y_test)], eval_metric=['f1', 'roc_auc'],
                                         init_points=1, n_iter=3)
    print('after bayesian search is completed...')
    print(gs.get_params())


def test_select_features():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    # 产生随机分类数据集，10个特征， 2个类别
    x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=2)
    train_cols_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    pdf = pd.DataFrame(x, columns=train_cols_list)
    pdf['label'] = y
    x_train, x_test, y_train, y_test = train_test_split(pdf.loc[:, train_cols_list],
                                                        pdf.loc[:, 'label'],
                                                        random_state=1)
    # params = {'random_state': 2, 'objective': 'binary:logistic', 'verbosity': 0, 'scale_pos_weight': 1, 'max_depth': 6, 'reg_lambda': 17,
    #  'reg_alpha': 15, 'gamma': 5, 'min_child_weight': 25, 'base_score': 0.6, 'colsample_bytree': 0.8999999999999999,
    #  'colsample_bylevel': 0.7, 'colsample_bynode': 0.5, 'subsample': 0.6, 'eta': 0.26, 'tree_method': 'exact', 'eval_metric': 'auc'
    #  }
    # alg = XGBClassifier(**params)
    # alg.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=10)
    # print(pd.Series(alg.feature_importances_, index=x_train.columns))
    # a = pd.Series(alg.get_booster().get_score(importance_type='gain'))
    # a = a / a.sum()
    # print(a)
    # train_dmatrix = xgb.DMatrix(x_train, y_train)
    # test_dmatrix = xgb.DMatrix(x_test, y_test)
    # evals = [(train_dmatrix, 'train'), (test_dmatrix, 'test')]
    # model = xgb.train(params, train_dmatrix,
    #                   num_boost_round=1000,
    #                   evals=evals,
    #                   maximize=True,
    #                   verbose_eval=True,
    #                   early_stopping_rounds=10)
    # a = pd.Series(model.get_score(importance_type='gain'))
    # a = a / a.sum()
    # print(a)
    random_param = {'seed': [2],
                    'objective': ['binary:logistic'],
                    'verbosity': [0],
                    'scale_pos_weight': np.arange(1, 3),
                    'max_depth': np.arange(2, 11),
                    'reg_lambda': np.arange(1, 21),
                    'reg_alpha': np.arange(1, 21),
                    'gamma': np.arange(0, 6),
                    'min_child_weight': np.arange(1, 31),
                    'base_score': np.arange(0.5, 1, 0.1),
                    'colsample_bytree': np.arange(0.5, 1, 0.1),
                    'colsample_bylevel': np.arange(0.5, 1, 0.1),
                    'colsample_bynode': np.arange(0.5, 1, 0.1),
                    'subsample': np.arange(0.5, 1, 0.1),
                    'eta': np.arange(0.02, 0.3, 0.02),
                    'tree_method': ['auto', 'exact', 'approx', 'hist']}
    scores = select_with_feature_importance(pd.concat([x_train, y_train], axis=1),
                                            pd.concat([x_test, y_test], axis=1),
                                            train_cols_list,
                                            'label',
                                            random_param,
                                            max_num_vars=5)
    print(scores)


def test_optuna():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    op = OptunaSearchXGB()
    tuning_param_dict = {'objective': 'binary:logistic',
                         'verbosity': 0,
                         'seed': 2,
                         'num_parallel_tree': ('int', {'low': 1, 'high': 4}),
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'gamma': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'base_score': ('discrete_uniform', {'low': 0.5, 'high': 0.9, 'q': 0.1}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bylevel': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'eta': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01}),
                         'rate_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'skip_drop': ('float', {'low': 1e-8, 'high': 1.0, 'log': True}),
                         'tree_method': ('categorical', {'choices': ['auto', 'exact', 'approx', 'hist']}),
                         'booster': ('categorical', {'choices': ['gbtree', 'dart']}),
                         'sample_type': ('categorical', {'choices': ['uniform', 'weighted']}),
                         'normalize_type': ('categorical', {'choices': ['tree', 'forest']})}
    op.search(x_train, y_train, tuning_param_dict, cv=1, coef_train_val_disparity=0.4,
              eval_set=[(x_valid, y_valid)], optuna_verbosity=1, early_stopping_rounds=30,
              n_warmup_steps=10)
    train_dmatrix = xgb.DMatrix(x_train, y_train)
    test_dmatrix = xgb.DMatrix(x_valid, y_valid)
    train_param = op.get_params()
    print(train_param)
    afsxc = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
    train_pred = afsxc.predict(train_dmatrix)
    print("train_data_ks", cal_ks(train_pred, y_train))
    test_pred = afsxc.predict(test_dmatrix)
    print("test_data_ks", cal_ks(test_pred, y_valid))
    op.plot_optimization().show()
    op.plot_importance(list(op.dynamic_param.keys())).show()
    op.plot_score()


if __name__ == '__main__':
    test_optuna()
