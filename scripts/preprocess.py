import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __call__(self, train, test, structures):
        return self.run(train, test, structures)

    def run(self, train, test, structures):
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
        structures = structures.copy().join(bond_df)
