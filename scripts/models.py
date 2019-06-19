from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate
import lightgbm
from functools import partial
import optuna


class LGBMRegressor(lightgbm.LGBMRegressor):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            parameters to sklearn.linear_model.LogisticRegression

        """
        super().__init__(**kwargs)

