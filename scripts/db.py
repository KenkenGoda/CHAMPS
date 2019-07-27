import os

import pandas as pd


class LocalFile:
    def __init__(self, config):
        self.config = config

    def get_train(self):
        if os.path.isfile(self.config.pickled_train_path):
            return pd.read_pickle(self.config.pickled_train_path)
        else:
            return pd.read_csv(self.config.train_path, nrows=self.config.nrows)

    def get_test(self):
        if os.path.isfile(self.config.pickled_test_path):
            return pd.read_pickle(self.config.pickled_test_path)
        else:
            return pd.read_csv(self.config.test_path, nrows=self.config.nrows)

    def get_structures(self):
        if os.path.isfile(self.config.pickled_structures_path):
            return pd.read_pickle(self.config.pickled_structures_path)
        else:
            return pd.read_csv(self.config.structures_path)

    def get_submission(self):
        return pd.read_csv(self.config.sample_submission_path)

    def get_dipole_moments(self):
        return pd.read_csv(self.config.dipole_moments_path)

    def get_magnetic_shielding_tensors(self):
        return pd.read_csv(self.config.magnetic_shielding_tensors_path)

    def get_mulliken_charges(self):
        return pd.read_csv(self.config.mulliken_charges_path)

    def get_potential_energy(self):
        return pd.read_csv(self.config.potential_energy_path)

    def get_scalar_coupling_contributions(self):
        return pd.read_csv(self.config.scalar_coupling_contributions_path)

