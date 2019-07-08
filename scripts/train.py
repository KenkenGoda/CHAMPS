import pandas as pd

from .dataproc import DataProcessor


class ModelTrainer:
    def __init__(self, config, **params):
        self.config = config
        self.params = params
        self.dataproc = DataProcessor()

    def run(self, X, y):
        pass
