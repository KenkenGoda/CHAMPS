import os

import pandas as pd
from tqdm import tqdm

from .utility import reduce_mem_usage
from .feature import FeatureFactory


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.pickled_feature_dir = config.pickled_feature_dir
        self.feature_names = config.feature_names
        self.target_name = config.target_name
        self.save = config.save_X

    def run(self, dataset):
        # X
        X_train = self._make_X(dataset.train, dataset, "train")
        X_test = self._make_X(dataset.test, dataset, "test")

        # y
        y_train = dataset.train[["type", self.target_name]]

        return X_train, y_train, X_test

    def _make_X(self, df, dataset, kind):
        path = os.path.join(self.pickled_feature_dir, f"{kind}", "X.pkl")
        if os.path.isfile(path):
            return self.load_feature(path)

        ff = FeatureFactory()
        features = [ff(name) for name in self.feature_names]

        X = pd.DataFrame(index=df.index)
        index_names = X.index.names
        for feature in tqdm(features):
            _name = feature.__class__.__name__
            _path = os.path.join(self.pickled_feature_dir, f"{kind}", f"{_name}.pkl")
            if os.path.isfile(_path):
                values = self.load_feature(_path)
            else:
                values = pd.DataFrame(feature.run(df, dataset))
                values = reduce_mem_usage(values, _name)
                self._save_feature(values, _path)
                print(f"save {_name} feature for {kind} to pickle")
            X = X.join(values)
            if X.index.names != index_names:
                X = X.reset_index().set_index(index_names)
            del values

        if self.save_X is True:
            self._save_feature(X, path)

        return X

    @staticmethod
    def _save_feature(values, path):
        values.to_pickle(path)

    @classmethod
    def load_feature(cls, path):
        return pd.read_pickle(path)
