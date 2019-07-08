import os

import pandas as pd

from .feature import FeatureFactory


class DataProcessor:
    def __init__(self, config, fc=None):
        ff = FeatureFactory()
        if config.feature_names == []:
            self.feature = None
        else:
            self.features = [ff(name) for name in config.feature_names]
        self.target_name = config.target_name
        self.pickle_dir = config.pickle_dir
        self.fc = fc

    def run(self, dataset, feature_exists=False):
        # load X dataframe
        if feature_exists:
            X_train = pd.read_pickle(os.path.join(self.pickle_dir, "X_train.pkl"))
            X_test = pd.read_pickle(os.path.join(self.pickle_dir, "X_test.pkl"))
        else:
            X_train = None
            X_test = None

        X_train = self._make_X(dataset.train, dataset, X_train)
        X_test = self._make_X(dataset.test, dataset, X_test)

        if self.fc is not None:
            X_train = X_train.merge(
                dataset.scalar_coupling_constant[
                    ["molecule_name", "atom_index_0", "atom_index_1", "fc"]
                ],
                on=["molecule_name", "atom_index_0", "atom_index_1"],
            )

        X_train = dataset.train.drop(
            columns=[
                "id",
                "molecule_name",
                "scalar_coupling_constant",
                "atom_index_0",
                "atom_index_1",
            ]
        )
        X_test = dataset.test.drop(
            columns=["id", "molecule_name", "atom_index_0", "atom_index_1"]
        )
        if self.target_feature:
            y_train = dataset.train["scalar_coupling_constant"]
            return X_train, y_train, X_test
        else:
            return X_train, X_test

    def _make_X(self, df, dataset, X=None):
        if X is None:
            X = pd.DataFrame(index=df.index)

        if self.features is not None:
            for feature in self.features:
                values = pd.DataFrame(feature.apply(dataset, df))
                X_cols = values.columns.tolist()
                X = X.join(values)
                if feature.default is not None:
                    X[X_cols] = X[X_cols].fillna(feature.default)
        return X
