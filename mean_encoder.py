import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, PredefinedSplit, cross_validate
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, make_scorer
from itertools import product
from xgboost import XGBClassifier


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification',
                 prior_weight_func=lambda x: 0, random_state=27):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
 
        :param n_splits: the number of splits used in mean encoding
 
        :param target_type: str, 'regression' or 'classification'
 
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        如果不考虑先验，可以用 prior_weight_func = lambda x: 0

        :param random_state: int
        """
 
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
        self.random_state = random_state
 
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None
 
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
 
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func, fill_method):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
 
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            # classification
            X_train['pred_temp'] = (y_train == target).astype(int)
        else:
            nf_name = '{}_pred'.format(variable)
            # regression
            X_train['pred_temp'] = y_train
        total_mean = X_train['pred_temp'].mean()
 
        # groupby是不会考虑nan, none的，因此得到的x_train的各个类别对应的mean和size是不统计nan， none的
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg([('mean', 'mean'), ('beta', 'size')])
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * total_mean + (1 - col_avg_y['beta']) * col_avg_y['mean']
        if fill_method == 'min':
            prior = col_avg_y['mean'].min()
        elif fill_method == 'max':
            prior = col_avg_y['mean'].max()
        elif fill_method == 'mean':
            prior = total_mean
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        # 由于col_avf_y不包含nan和none的计算结果，因此对于x_train和x_test原本的缺失值，
        # 连表后在nf_name列中也不会有对应的计算结果，需要自己填充缺失值
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
 
        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y, fill_method='mean'):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :param fill_method: str, mean, min or max, 对于缺失值、训练集未出现的取值，在验证集和测试集应该如何填充，
           mean就是用训练集的平均值，max和min就是用训练集各水平的平均值里面的最大和最小值
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            skf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
 
        if self.target_type == 'classification':
            self.target_values = list(sorted(set(y)))[1:]
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind],
                        variable, target, self.prior_weight_func, fill_method)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind],
                        variable, None, self.prior_weight_func, fill_method)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
 
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
 
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                print(variable, target)
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    # X_new[[variable]]里面的缺失值和训练集不含的类别取值，col_avg_y都是没有计算这些的，因此
                    # X_new[[variable]]与col_acg_y连表后，X_new的缺失值等在nf_name列中都没有对应的计算结果，需要用fillna填充
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(
                        prior, inplace=False)[nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(
                        prior, inplace=False)[nf_name]
                X_new[nf_name] /= self.n_splits
 
        return X_new


class MeanEncoderTuner(object):
    def __init__(self, categorical_features, model, n_splits=5,
                 target_type='classification', prior_weight_func=None, random_state=27):
        self._me = MeanEncoder(categorical_features, n_splits,
                               target_type, prior_weight_func, random_state)
        self._alg = model
        self.categorical_features = categorical_features

    def _ks_stats(self, y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
        ks_value = max(tpr - fpr)
        return ks_value

    def _regularize_metric(self, eval_metric):
        """
        :param eval_metric: list of str, each str should be a built-in metric of sklearn
        :param early_stopping_rounds: int or None
        """
        eval_metric_list = {}
        for metric in eval_metric:
            if metric == 'f1':
                eval_metric_list['f1'] = make_scorer(f1_score)
            elif metric == 'accuracy':
                eval_metric_list['accuracy'] = make_scorer(accuracy_score)
            elif metric == 'roc_auc':
                eval_metric_list['roc_auc'] = make_scorer(roc_auc_score,
                                                          needs_threshold=True)
            elif metric == 'ks':
                eval_metric_list['ks'] = make_scorer(self._ks_stats,
                                                     needs_threshold=True)
            else:
                raise RuntimeError('not-supported metric: {}'.format(metric))
        return eval_metric_list

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
            skf = StratifiedKFold(n_splits=cv, shuffle=True,
                                  random_state=self.random_state)
            folds = []
            for train_indices_array, val_indices_array in skf.split(
                    x_train_val, y_train_val):
                folds.append((train_indices_array.tolist(),
                              val_indices_array.tolist()))
        return x_train_val, y_train_val, cv, folds

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

    def grid_search(self, x_train, y_train, cv,
                    params={'k': 2, 'f': 1}, tuning_dict={'k': [-2, 2], 'f': [-0.5, 0.5]},
                    n_gs_jobs=1, coef_train_val_disparity=0.2, eval_set=None, eval_metric=['f1'], verbose=True):
        eval_metric_list = self._regularize_metric(eval_metric)
        x_train_val, y_train_val, cv, folds = self._make_train_val(x_train, y_train, eval_set, cv)
        x_train_val_new = self._me.fit_transform(x_train_val, y_train_val)
        new_cols = list(set(x_train_val_new.columns) - set(self.categorical_features))
        cv_results = cross_validate(self._alg, x_train_val_new[new_cols], y_train_val,
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
        print('original params {} with val score {}'.format(params, best_score))
        print('stepwise searching begins...')
        for (param, steps) in tuning_dict.items():
            print('param: {}, original value: {}'.format(param, params[param]))
            for step in steps:
                print('step ', step)
                while True:
                    if params[param] + step > 0:
                        params[param] += step
                        print('-->: ', params[param])
                        x_train_val_new = self._me.fit_transform(x_train_val, y_train_val)
                        new_cols = list(set(x_train_val_new.columns) - set(self.categorical_features))
                        cv_results = cross_validate(self._alg, x_train_val_new[new_cols], y_train_val,
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
                            print('better val score {}'.format(best_score))
                        else:
                            params[param] -= step
                            break
                    else:
                        break
        print('best tuning: {} with val score {}'.format(params, best_score))
        return params


def test_grid_search():
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    train_cols_list = ['x281', 'x291', 'x1423', 'y']
    pdf = pd.read_excel('题目四.xlsx')[train_cols_list]
    x_train, x_test, y_train, y_test = train_test_split(pdf[train_cols_list[:-1]],
                                                        pdf['y'],
                                                        random_state=1)
    alg = XGBClassifier(learning_rate=0.1,
                        n_estimators=100,
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
    met = MeanEncoderTuner(categorical_features=['x281'], model=alg, n_splits=5,
                           target_type='classification', prior_weight_func=None, random_state=27)
    params = met.grid_search(x_train, y_train, 3,
                             params={'k': 2, 'f': 1}, tuning_dict={'k': [-2, 2], 'f': [-0.5, 0.5]},
                             n_gs_jobs=1, coef_train_val_disparity=0.2,
                             eval_set=[(x_test, y_test)], eval_metric=['ks'], verbose=False)


def test_fit_transform():
    from sklearn.model_selection import train_test_split
    print('preparing data...')
    train_cols_list = ['x281', 'x291', 'x1423', 'y']
    pdf = pd.read_excel('题目四.xlsx')[train_cols_list]
    x_train, x_test, y_train, y_test = train_test_split(pdf[train_cols_list[:-1]],
                                                        pdf['y'],
                                                        random_state=1)
    print(x_train.head())
    me = MeanEncoder(categorical_features=['x281'], n_splits=5,
                     target_type='classification', prior_weight_func=None, random_state=27)
    x_train = me.fit_transform(x_train, y_train)
    x_test = me.transform(x_test)
    print(x_train.head())
    print(me.transform(x_train).head())



if __name__ == '__main__':
    test_fit_transform()