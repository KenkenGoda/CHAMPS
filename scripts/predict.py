import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .model import LGBMRegressor
from .tune import ParameterTuning


class Prediction:
    def __init__(self, config):
        self.pickle_dir = config.pickle_dir
        self.target_name = config.target_name
        self.seed = config.seed

    def run(
        self, X_train, y_train, X_test, tuning=False, n_trials=1, n_splits=5, save=False
    ):
        pt = ParameterTuning(self.seed)
        if tuning:
            params = pt.run(X_train, y_train, n_trials=n_trials)
        else:
            params = pt.get_best_params()
            if params is None:
                params = {}

        self.model = LGBMRegressor(**params)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        score = []
        y_pred = []
        for train_idx, valid_idx in kf.split(X_train):
            X_train_ = X_train.iloc[train_idx]
            y_train_ = y_train.iloc[train_idx][self.target_name]
            X_valid_ = X_train.iloc[valid_idx]
            y_valid_ = y_train.iloc[valid_idx]
            self.model.fit(X_train_, y_train_)
            y_pred_ = self.model.predict(X_valid_)
            score.append(self.model.calculate_score(y_valid_, y_pred_))
            y_pred.append(self.model.predict(X_test))
        print(f"Score: {np.mean(score)}")
        y_pred = np.mean(y_pred, axis=0)

        if save:
            self._save(y_pred, X_test.index)

        return y_pred

    def _save(self, y_pred, index):
        os.makedirs(self.pickle_dir, exist_ok=True)
        prediction = pd.DataFrame(y_pred, index=index, columns=self.target_name)
        prediction.to_pickle(
            os.path.join(self.pickle_dir, f"{self.target_name}_test.pkl")
        )
        print(f"save {self.target_name}_test to pickle")
