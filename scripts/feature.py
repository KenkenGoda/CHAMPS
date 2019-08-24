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
                MoleculeStatisticsFeature,
                AtomStatisticsFeature,
                TypeStatisticsFeature,
                PredictedFeature,
            ]:
                lst.append(obj.__name__)
        return lst


class Feature:
    def __init___(self, **kwargs):
        self.name = str(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def run(self, df, dataset):
        values = self.extract(df, dataset)
        values = values.fillna(0)
        return values

    def extract(self, df, dataset):
        raise NotImplementedError

    @staticmethod
    def _get_converted_multi_columns(values, head_name=None):
        col_names = [col[0] + "_" + col[1] for col in values.columns.values]
        if head_name:
            col_names = [head_name + "_" + col for col in col_names]
        return col_names


class BasicFeature(Feature):

    column = None
    prefix = None

    def extract(self, df, dataset):
        values = df[self.column]
        if values.dtype == "O":
            values = pd.get_dummies(values, prefix=self.prefix)
        return values


class MoleculeStatisticsFeature(Feature):

    column = None
    representative_values = None
    col_names = None
    head_name = None

    def extract(self, df, dataset):
        agg = {self.column: self.representative_values}
        values = df.groupby("molecule_name").agg(agg)
        if self.col_names:
            values.name = self.col_names
        else:
            values.columns = self._get_converted_multi_columns(
                values, head_name=self.head_name
            )
        return values


class AtomStatisticsFeature(Feature):

    atom_idx = None
    column = None
    representative_values = None
    col_names = None
    head_name = None

    def extract(self, df, dataset):
        agg = {self.column: self.representative_values}
        values = df.groupby(["molecule_name", f"atom_index_{self.atom_idx}"]).agg(agg)
        if self.col_names:
            values.columns = [self.col_names]
        else:
            values.columns = self._get_converted_multi_columns(
                values, head_name=self.head_name
            )
        return values


class TypeStatisticsFeature(Feature):

    column = None
    representative_values = None
    col_names = None
    head_name = None

    def extract(self, df, dataset):
        agg = {self.column: self.representative_values}
        values = df.groupby(["molecule_name", "type"]).agg(agg)
        if self.col_names:
            values.columns = [self.col_names]
        else:
            values.columns = self._get_converted_multi_columns(
                values, head_name=self.head_name
            )
        df_ = df["type"].reset_index()
        values = values.reset_index()
        values = df_.merge(values, on=["molecule_name", "type"])
        values = values.set_index(["molecule_name", "atom_index_0", "atom_index_1"])
        values = values.drop(columns="type")
        return values


class MoleculeType(BasicFeature):

    column = "type"


class MoleculeType0(BasicFeature):

    column = "type_0"
    prefix = "type"


class Atom0(BasicFeature):

    column = "atom_0"
    prefix = "atom_0"


class Atom1(BasicFeature):

    column = "atom_1"
    prefix = "atom_1"


class AtomX0(BasicFeature):

    column = "x_0"


class AtomX1(BasicFeature):

    column = "x_1"


class AtomY0(BasicFeature):

    column = "y_0"


class AtomY1(BasicFeature):

    column = "y_1"


class AtomZ0(BasicFeature):

    column = "z_0"


class AtomZ1(BasicFeature):

    column = "z_1"


class MoleculeDistance(BasicFeature):

    column = "dist"


class MoleculeDistanceX(BasicFeature):

    column = "dist_x"


class MoleculeDistanceY(BasicFeature):

    column = "dist_y"


class MoleculeDistanceZ(BasicFeature):

    column = "dist_z"


class MoleculeCount(MoleculeStatisticsFeature):

    column = "id"
    representative_values = "count"
    col_names = "molecule_couples_count"


class MoleculeX0Statistics(MoleculeStatisticsFeature):

    column = "x_0"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeX1Statistics(MoleculeStatisticsFeature):

    column = "x_1"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeY0Statistics(MoleculeStatisticsFeature):

    column = "y_0"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeY1Statistics(MoleculeStatisticsFeature):

    column = "y_1"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeZ0Statistics(MoleculeStatisticsFeature):

    column = "z_0"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeZ1Statistics(MoleculeStatisticsFeature):

    column = "z_1"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeDistanceStatistics(MoleculeStatisticsFeature):

    column = "dist"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeDistanceXStatistics(MoleculeStatisticsFeature):

    column = "dist_x"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeDistanceYStatistics(MoleculeStatisticsFeature):

    column = "dist_y"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class MoleculeDistanceZStatistics(MoleculeStatisticsFeature):

    column = "dist_z"
    representative_values = ["min", "max", "mean"]
    head_name = "molecule"


class Atom0Count(AtomStatisticsFeature):

    atom_idx = 0
    column = "id"
    representative_values = "count"
    col_names = "atom_0_couples_count"


class Atom1Count(AtomStatisticsFeature):

    atom_idx = 1
    column = "id"
    representative_values = "count"
    col_names = "atom_1_couples_count"


class Atom0X1Statistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "x_1"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1X0Statistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "x_0"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom0Y1Statistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "y_1"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1Y0Statistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "y_0"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom0Z1Statistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "z_1"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1Z0Statistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "z_0"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom0DistanceStatistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "dist"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1DistanceStatistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "dist"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_1"


class Atom0DistanceXStatistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "dist_x"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1DistanceXStatistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "dist_x"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_1"


class Atom0DistanceYStatistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "dist_y"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1DistanceYStatistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "dist_y"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_1"


class Atom0DistanceZStatistics(AtomStatisticsFeature):

    atom_idx = 0
    column = "dist_z"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_0"


class Atom1DistanceZStatistics(AtomStatisticsFeature):

    atom_idx = 1
    column = "dist_z"
    representative_values = ["min", "max", "mean"]
    head_name = "atom_1"


class TypeX0Statistics(TypeStatisticsFeature):

    column = "x_0"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeX1Statistics(TypeStatisticsFeature):

    column = "x_1"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeY0Statistics(TypeStatisticsFeature):

    column = "y_0"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeY1Statistics(TypeStatisticsFeature):

    column = "y_1"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeZ0Statistics(TypeStatisticsFeature):

    column = "z_0"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeZ1Statistics(TypeStatisticsFeature):

    column = "z_1"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeDistanceStatistics(TypeStatisticsFeature):

    column = "dist"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeDistanceXStatistics(TypeStatisticsFeature):

    column = "dist_x"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeDistanceYStatistics(TypeStatisticsFeature):

    column = "dist_y"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


class TypeDistanceZStatistics(TypeStatisticsFeature):

    column = "dist_z"
    representative_values = ["min", "max", "mean"]
    head_name = "type"


"""
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
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
                path = os.path.join(
                    Config().pickled_feature_dir, "test", f"{self.column}.pkl"
                )
                values = pd.read_pickle(path)
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
