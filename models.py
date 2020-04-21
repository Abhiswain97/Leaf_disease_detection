from sklearn import linear_model, ensemble, model_selection, svm
from skopt import BayesSearchCV
import xgboost as xgb


class ClassificationModels:
    def __init__(self):
        self.models = {
            "logistic_regression_cv": self._logistic_regression_cv,
            "random_forest_cv": self._random_forest_cv,
            "svc": self._svc,
            "xgboost_cv": self._xgboost_cv
        }

    def __call__(self, model, X_train, y_train, is_binary=True):
        if model not in self.models:
            raise Exception("Model not implemented")
        else:
            if is_binary:
                name = 'binary'
                objective = 'binary:logistic'
                eval_metric = 'auc'
                scoring = 'roc_auc'
                m_name = 'roc_auc'
                if model == 'xgboost_cv':
                    return self._xgboost_cv(X_train, y_train, objective, scoring, eval_metric, m_name, name)
                elif model == 'random_forest_cv':
                    return self._random_forest_cv(X_train, y_train, scoring, m_name, name)

            elif not is_binary:
                name = 'multiclass'
                objective = 'multi:softmax'
                eval_metric = 'mlogloss'
                scoring = metrics.make_scorer(metrics.f1_score, average='weighted')
                m_name = 'f1'
                if model == 'xgboost_cv':
                    return self._xgboost_cv(X_train, y_train, objective, scoring, eval_metric, m_name, name)
                elif model == 'random_forest_cv':
                    return self._random_forest_cv(X_train, y_train, scoring, m_name, name)
            else:
                return self.models[model](X_train, y_train)

    @staticmethod
    def _logistic_regression_cv(X_train, y_train):
        return linear_model.LogisticRegressionCV(
            cv=model_selection.StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            )
        ).fit(X_train, y_train)

    @staticmethod
    def _random_forest_cv(X_train, y_train, scoring, m_name, name):
        def status_print(optim_result):
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print('Model #{}\nBest {}: {}\nBest params: {}\n'.format(
                len(all_models),
                m_name,
                np.round(bayes_cv_tuner.best_score_, 4),
                bayes_cv_tuner.best_params_
            ))

            # Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            all_models.to_csv('results\\' + clf_name + f"({name})_cv_results.csv")

        return BayesSearchCV(
            estimator=ensemble.RandomForestClassifier(),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'max_depth': (1, 50),
                'n_estimators': (50, 100),
            },
            scoring=scoring,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            ),
            n_jobs=3,
            n_iter=10,
            verbose=3,
            refit=True,
            random_state=42
        ).fit(X_train, y_train, callback=status_print)

    @staticmethod
    def _xgboost_cv(X_train, y_train, objective, scoring, eval_metric, m_name, name):
        def status_print(optim_result):
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print('Model #{}\nBest {}: {}\nBest params: {}\n'.format(
                len(all_models),
                m_name,
                np.round(bayes_cv_tuner.best_score_, 4),
                bayes_cv_tuner.best_params_
            ))

            # Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            all_models.to_csv('results\\' + clf_name + f"({name})_cv_results.csv")

        return BayesSearchCV(
            estimator=xgb.XGBClassifier(
                n_jobs=1,
                objective=objective,
                eval_metric=eval_metric,
                silent=1,
                tree_method='approx',
                n_estimators=200
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'max_depth': (1, 50),
                'n_estimators': (50, 100),
            },
            scoring=scoring,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            ),
            n_jobs=3,
            n_iter=10,
            verbose=3,
            refit=True,
            random_state=42
        ).fit(X_train, y_train, callback=status_print)

    @staticmethod
    def _svc(X_train, y_train):
        return svm.SVC().fit(X_train, y_train)
