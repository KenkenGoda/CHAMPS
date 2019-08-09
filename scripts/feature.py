import inspect
import os

import pandas as pd

from .config import Config


class FeatureFactory:
    def __call__(self, feature_names, **kwargs):
        if feature_names in globals():
            return globals()[feature_names](**kwargs)
        else:
            raise ValueError("No feature defined named with {}".format(feature_names))

    def feature_list(self):
        lst = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                Config,
                FeatureFactory,
                Feature,
                BasicFeature,
                AtomCountFeature,
                PredictedFeature,
            ]:
                lst.append(obj.__name__)
        return lst


class Feature:
    categories = None
    dummy = True
    default = 0
    dummy_na = True

    def __init___(self, **kwargs):
        self.name = str(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def run(self, df, dataset):
        values = self.extract(df, dataset)

        if self.categories:
            if self.dummy:
                values = self.convert_into_categories(values)
            else:
                self.default = None

        if self.default is not None:
            values = values.fillna(self.default)

        return values

    def convert_into_categories(self, X):
        index = X.index
        X = pd.Categorical(X, categories=self.categories)
        if self.dummy:
            X = pd.get_dummies(X, dummy_na=self.dummy_na)
        X.columns = self.get_columns()
        X.index = index
        return X

    def get_columns(self):
        if isinstance(self.categories, list) and self.dummy:
            columns = [f"{self}_{cat}" for cat in self.categories]
            if self.dummy_na:
                columns += [f"{self}_none"]
            return columns
        else:
            return [str(self)]

    def extract(self, df, dataset):
        raise NotImplementedError

    @staticmethod
    def _get_converted_multi_columns(df, head_name=None):
        if head_name:
            return [
                head_name + "_" + col[0] + "_" + col[1] for col in df.columns.values
            ]
        else:
            return [col[0] + "_" + col[1] for col in df.columns.values]


class BasicFeature(Feature):
    def __init__(self):
        pass


class MoleculeStatisticsFeature(Feature):

    column = None
    representative_value = None
    head_name = None

    def extract(self, df, dataset):
        agg = {self.column: self.representative_value}
        values = df.groupby("molecule_name").agg(agg)
        values.columns = self._get_converted_multi_columns(
            values, head_name=self.head_name
        )
        return values


class AtomStatisticsFeature(Feature):

    column = None
    representative_value = None
    atom_idx = None
    head_name = None

    def extract(self, df, dataset):
        agg = {self.column: self.representative_value}
        values = df.groupby(["molecule_name", f"atom_index_{self.atom_idx}"]).agg(agg)
        values.columns = self._get_converted_multi_columns(
            values, head_name=self.head_name
        )
        return values


class MoleculeCount(Feature):
    def extract(self, df, dataset):
        values = df.groupby("molecule_name").agg({"id": "count"})
        values.columns = ["molecule_couples"]
        return values


class MoleculeDistanceStatistics(Feature):
    def extract(self, df, dataset):
        values = df.groupby("molecule_name").agg({"dist": ["mean", "min", "max"]})
        values.columns = [
            "molecule_dist_mean",
            "molecule_dist_min",
            "molecule_dist_max",
        ]
        return values


class AtomCountFeature(Feature):

    atom_idx = None

    def extract(self, df, dataset):
        values = df.groupby(["molecule_name", f"atom_index_{self.atom_idx}"]).agg(
            {"id": "count"}
        )
        values.columns = [f"atom_{self.atom_idx}_couples_count"]
        return values


class Atom0Count(AtomCountFeature):

    atom_idx = 0


class Atom1Count(AtomCountFeature):

    atom_idx = 1


class Atom(Feature):
    pass


"""
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']
    df = reduce_mem_usage(df)
"""


class PredictedFeature(Feature):
    """ Predicted features that construct scalar coupling constant """

    column = None

    def extract(self, df, dataset):
        if self.column in df.columns:
            values = df[self.column]
        else:
            try:
                values = pd.read_pickle(
                    os.path.join(Config().pickle_dir, f"{self.column}_test.pkl")
                )
            except ValueError:
                print(f"Not found pickled {self.column}.")
        return values


class FermiContact(PredictedFeature):

    column = "fc"


class SpinDipolar(PredictedFeature):

    column = "sd"


class ParaMagneticSpinOrbit(PredictedFeature):

    column = "pso"


class DiamagneticSpinOrbit(PredictedFeature):

    column = "dso"
