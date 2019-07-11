import os

import optuna

from .model import LGBMRegressor


class ParameterTuning:
    def __init__(self, seed):
        self.seed = seed

    def run(self, X, y, n_trials=1):
        def objective(trial):
            params = {
                "boosting_type": "goss",
                "num_leaves": trial.suggest_int("num_leaves", 2e0, 1e1),
                "n_estimators": int(1e6),
                "subsample_for_bin": trial.suggest_int("subsample_for_bin", 2e0, 1e2),
                "min_child_samples": trial.suggest_int("min_child_samples", 3e0, 1e2),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 1e-1),
                "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 1e2),
            }
            model = LGBMRegressor(**params)

            return 1.0 - auc

        ## X_train, y_train, X_valid, y_validに分ける

        study_name = "lgb_study"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///../database/lgb.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        return study.best_params

    def _evaluate(self, model, X_train, y_train, X_valid, y_valid):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=3,
            verbose=False,
        )
        y_pred = model.predict(X_valid)
        return model.calculate_score(y_valid, y_pred)

    def get_best_params(self):
        storage = "../database/lgb.db"
        if os.path.isfile(storage):
            study_name = "lgb_study"
            study = optuna.create_study(
                study_name=study_name,
                storage="sqlite:///../database/lgb.db",
                load_if_exists=True,
            )
            print("Loading the best parameters")
            return study.best_params
        else:
            print("There are no best parameters")
            return None
