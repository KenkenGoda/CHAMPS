import os
from tqdm import tqdm

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
        X_train = self._make_X(dataset.train, dataset, "train")
        X_test = self._make_X(dataset.test, dataset, "test")

        # y
        if self.target_name == "scalar_coupling_constant":
            y_train = dataset.train["scalar_coupling_constant"]
        else:
            y_train = dataset.scalar_coupling_contributions[self.target_name]
        y_train["type"] = dataset.train["type"]

        return X_train, y_train, X_test

    def _make_X(self, df, dataset, kind):
        ff = FeatureFactory()
        features = [ff(name) for name in self.feature_names]

        X = pd.DataFrame(index=df.index)
        for feature in tqdm(features):
            name = feature.__class__.__name__
            if os.path.isfile(os.path.join(self.pickle_dir, f"{name}_{kind}.pkl")):
                values = self.load_feature(self.pickle_dir, name, kind)
            else:
                values = pd.DataFrame(feature.run(df, dataset))
                self._save_feature(values, name, kind)
            X = X.join(values)
        return X

    def _save_feature(self, values, name, kind):
        os.makedirs(self.pickle_dir, exist_ok=True)
        values.to_pickle(os.path.join(self.pickle_dir, f"{name}_{kind}.pkl"))
        print(f"save {name}_{kind} class to pickle")

    @classmethod
    def load_feature(cls, save_dir, name, kind):
        return pd.read_pickle(os.path.join(save_dir, f"{name}_{kind}.pkl"))
