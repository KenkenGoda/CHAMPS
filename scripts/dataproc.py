import os

import pandas as pd

from .feature import FeatureFactory


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.pickle_dir = config.pickle_dir
        self.feature_names = config.feature_names
        self.target_name = config.target_name

    def run(self, dataset):
        # X
        X_train = self._make_X(dataset.train, dataset)
        X_test = self._make_X(dataset.test, dataset)

        # y
        if self.target_name == "scalar_coupling_constant":
            y_train = dataset.train[["scalar_coupling_constant"]]
        else:
            y_train = dataset.scalar_coupling_contributions[[self.target_name]]
        y_train["type"] = dataset.train["type"]

        return X_train, y_train, X_test

    def _make_X(self, df, dataset):
        ff = FeatureFactory()
        features = [ff(name) for name in self.feature_names]

        X = pd.DataFrame(index=df.index)
        for feature in features:
            name = feature.__class__.__name__
            if os.path.isfile(os.path.join(self.pickle_dir, f"{name}.pkl")):
                values = pd.read_pickle(os.path.join(self.pickle_dir, f"{name}.pkl"))
            else:
                values = pd.DataFrame(feature.run(dataset, df))
                self._save_feature(values, name)
            X = X.join(values)
        return X

    def _save_feature(self, values, name):
        os.makedirs(self.pickle_dir, exist_ok=True)
        values.to_pickle(os.path.join(self.pickle_dir, f"{name}.pkl"))
        print(f"save {name} class to pickle")

    def load_feature(self, name):
        return pd.read_pickle(os.path.join(self.pickle_dir, f"{name}.pkl"))
