import pandas as pd


class LocalFile:
    def __init__(self, config):
        self.config = config

    def get_train(self):
        return pd.read_csv(self.config.train_path)

    def get_test(self):
        return pd.read_csv(self.config.test_path)

    def get_submission(self):
        return pd.read_csv(self.config.submission_path)

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

    def get_structures(self):
        return pd.read_csv(self.config.structures_path)

