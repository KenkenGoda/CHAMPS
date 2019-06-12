import pandas as pd


class DataProcessor:
    def __init__(self, features, target_feature=None):
        self.features = features
        self.target_feature = target_feature

    def __call__(self, dataset):
        self._process(dataset)

    def _process(self, dataset):
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
