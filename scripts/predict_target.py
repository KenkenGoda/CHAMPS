import os

import numpy as numpy
import pandas as pd
from sklearn.model_selection import KFold

from scripts.model import LogisticRegression


class TargetPrediction:
    def __init__(self):
        pass

    def run(self, X_train, y_train, X_test):
        self.model = LogisticRegression()
