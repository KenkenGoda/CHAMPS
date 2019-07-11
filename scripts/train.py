import pandas as pd

from .model import LGBMRegressor
from .tune import ParameterTuning


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def run(self, X, y, tuning=False, n_trials=1):
        pt = ParameterTuning()
        if tuning:
            params = pt.run(X, y, n_trials=n_trials)
        else:
            params = pt.get_best_params()
            if params is None:
                params = {}

        self.model = LGBMRegressor(**params)

        self.model.fit(X, y)
        self.model.predict()
