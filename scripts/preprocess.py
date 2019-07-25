import numpy as np
import pandas as pd


class Preprocessor:
    def run(self, train, test, structures, scalar_couling_contributions):
        structures = self.preprocess_structures(structures)
        train = self.preprocess_train(train, structures, scalar_couling_contributions)
        test = self.preprocess_test(test, structures)
        return train, test, structures

    def preprocess_structures(self, structures):
        structures = structures.copy()

        ATOMIC_RADIUS = {"H": 0.38, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}
        FUDGE_FACTOR = 0.05
        ATOMIC_RADIUS = {k: v + FUDGE_FACTOR for k, v in ATOMIC_RADIUS.items()}

        ELECTRONEGATIVITY = {"H": 2.2, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}

        atoms = structures["atom"].values
        atoms_en = [ELECTRONEGATIVITY[x] for x in atoms]
        atoms_rad = [ATOMIC_RADIUS[x] for x in atoms]

        structures["electronegativity"] = atoms_en
        structures["radius"] = atoms_rad

        i_atom = structures["atom_index"].values
        p = structures[["x", "y", "z"]].values
        p_compare = p
        m = structures["molecule_name"].values
        m_compare = m
        r = structures["radius"].values
        r_compare = r

        source_row = np.arange(structures.shape[0])
        MAX_ATOMS = 28

        bonds = np.zeros((structures.shape[0] + 1, MAX_ATOMS + 1), dtype=np.int8)
        bond_dists = np.zeros(
            (structures.shape[0] + 1, MAX_ATOMS + 1), dtype=np.float32
        )

        for i in range(MAX_ATOMS - 1):
            p_compare = np.roll(p_compare, -1, axis=0)
            m_compare = np.roll(m_compare, -1, axis=0)
            r_compare = np.roll(r_compare, -1, axis=0)

            mask = np.where(m == m_compare, 1, 0)
            dists = np.linalg.norm(p - p_compare, axis=1) * mask
            r_bond = r + r_compare

            bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

            source_row = source_row
            target_row = source_row + i + 1
            target_row = np.where(
                np.logical_or(target_row > structures.shape[0], mask == 0),
                structures.shape[0],
                target_row,
            )

            source_atom = i_atom
            target_atom = i_atom + i + 1
            target_atom = np.where(
                np.logical_or(target_atom > MAX_ATOMS, mask == 0),
                MAX_ATOMS,
                target_atom,
            )

            bonds[(source_row, target_atom)] = bond
            bonds[(target_row, source_atom)] = bond
            bond_dists[(source_row, target_atom)] = dists
            bond_dists[(target_row, source_atom)] = dists

        bonds = np.delete(bonds, axis=0, obj=-1)
        bonds = np.delete(bonds, axis=1, obj=-1)
        bond_dists = np.delete(bond_dists, axis=0, obj=-1)
        bond_dists = np.delete(bond_dists, axis=1, obj=-1)

        bonds_numeric = [[i for i, x in enumerate(row) if x] for row in bonds]
        bond_lengths = [
            [dist for i, dist in enumerate(row) if i in bonds_numeric[j]]
            for j, row in enumerate(bond_dists)
        ]
        bond_lengths_mean = [np.mean(x) for x in bond_lengths]
        bond_lengths_std = [np.std(x) for x in bond_lengths]
        n_bonds = [len(x) for x in bonds_numeric]

        bond_data = {
            "n_bonds": n_bonds,
            "bond_lengths_mean": bond_lengths_mean,
            "bond_lengths_std": bond_lengths_std,
        }
        bond_df = pd.DataFrame(bond_data)
        structures = structures.join(bond_df)
        return structures

    def preprocess_train(self, train, structures, scalar_couling_contributions):
        train = train.copy()
        train = self._map_atom_info(train, structures, 0)
        train = self._map_atom_info(train, structures, 1)
        train = self._get_distance_between_atoms(train)
        train["type_0"] = train["type"].apply(lambda x: x[0])
        train = train.merge(
            scalar_couling_contributions,
            on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
        )
        return train.set_index(["molecule_name", "atom_index_0", "atom_index_1"])

    def preprocess_test(self, test, structures):
        test = test.copy()
        test = self._map_atom_info(test, structures, 0)
        test = self._map_atom_info(test, structures, 1)
        test = self._get_distance_between_atoms(test)
        test["type_0"] = test["type"].apply(lambda x: x[0])
        return test.set_index(["molecule_name", "atom_index_0", "atom_index_1"])

    def _map_atom_info(self, df, structures, atom_idx):
        df = pd.merge(
            df,
            structures,
            how="left",
            left_on=["molecule_name", f"atom_index_{atom_idx}"],
            right_on=["molecule_name", "atom_index"],
        )
        df = df.rename(
            columns={
                "atom": f"atom_{atom_idx}",
                "x": f"x_{atom_idx}",
                "y": f"y_{atom_idx}",
                "z": f"z_{atom_idx}",
            }
        )
        return df

    def _get_distance_between_atoms(self, df):
        df_p_0 = df[["x_0", "y_0", "z_0"]].values
        df_p_1 = df[["x_1", "y_1", "z_1"]].values

        df["dist"] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
        df["dist_x"] = (df["x_0"] - df["x_1"]) ** 2
        df["dist_y"] = (df["y_0"] - df["y_1"]) ** 2
        df["dist_z"] = (df["z_0"] - df["z_1"]) ** 2
        return df
