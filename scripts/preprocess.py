import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __call__(self, raw):
        return self.run(raw)

    def run(self, raw):
        train = raw.train
        test = raw.test

        # map information of atoms
        train = self.map_atom_info(train, raw.structures, 0)
        train = self.map_atom_info(train, raw.structures, 1)
        test = self.map_atom_info(test, raw.structures, 0)
        test = self.map_atom_info(test, raw.structures, 1)

        # add distance between two atoms
        train = self.add_distance(train)
        test = self.add_distance(test)

        # devide type into number and charactor
        train = self.devide_type(train)
        test = self.devide_type(test)

        # normalize distance by type
        train = self.distance_normalization(train)
        test = self.distance_normalization(test)

        # label encode
        for col in ["atom_0", "atom_1", "type_0", "type_1", "type"]:
            lbe = LabelEncoder()
            lbe.fit(list(train[col].values) + list(test[col].values))
            train[col] = lbe.transform(list(train[col].values))
            test[col] = lbe.transform(list(test[col].values))

        return train, test

    @staticmethod
    def map_atom_info(df, structures, index):
        df = pd.merge(
            df,
            structures,
            how="left",
            left_on=["molecule_name", f"atom_index_{index}"],
            right_on=["molecule_name", "atom_index"],
        )
        df = df.drop(columns="atom_index")
        df = df.rename(
            columns={
                "atom": f"atom_{index}",
                "x": f"x_{index}",
                "y": f"y_{index}",
                "z": f"z_{index}",
            }
        )
        return df

    @staticmethod
    def add_distance(df):
        p_0 = df[["x_0", "y_0", "z_0"]].values
        p_1 = df[["x_1", "y_1", "z_1"]].values
        df["dist"] = np.linalg.norm(p_0 - p_1, axis=1)
        df["dist_x"] = (df["x_0"] - df["x_1"]) ** 2
        df["dist_y"] = (df["y_0"] - df["y_1"]) ** 2
        df["dist_z"] = (df["z_0"] - df["z_1"]) ** 2
        return df

    @staticmethod
    def devide_type(df):
        df["type_0"] = df["type"].apply(lambda x: x[0])
        df["type_1"] = df["type"].apply(lambda x: x[1:])
        return df

    @staticmethod
    def distance_normalization(df):
        df["dist_to_type_mean"] = df["dist"] / df.groupby("type")["dist"].transform(
            "mean"
        )
        df["dist_to_type_0_mean"] = df["dist"] / df.groupby("type_0")["dist"].transform(
            "mean"
        )
        df["dist_to_type_1_mean"] = df["dist"] / df.groupby("type_1")["dist"].transform(
            "mean"
        )
        return df
