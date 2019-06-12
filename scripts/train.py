import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
from lightgbm import LGBMRegressor
from functools import partial
import optuna


class ModelTrainer:

    def __init__(self, config, **params):
        self.config = config
        self.params = params

    def run(self, X, y):
        