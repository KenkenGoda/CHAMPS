from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
import lightgbm
from functools import partial
import optuna


class LGBMRegressor(lightgbm.LGBMRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

