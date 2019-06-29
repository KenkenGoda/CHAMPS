import os


class Config:
    def __init__(self, config_file=None):
        # raw data path
        self.data_dir = "../data"
        self.train_path = os.path.join(self.data_dir, "train.csv")
        self.test_path = os.path.join(self.data_dir, "test.csv")
        self.submission_path = os.path.join(self.data_dir, "sample_submission.csv")
        self.dipole_moments_path = os.path.join(self.data_dir, "dipole_moments.csv")
        self.magnetic_shielding_tensors_path = os.path.join(
            self.data_dir, "magnetic_shielding_tensors.csv"
        )
        self.mulliken_charges_path = os.path.join(self.data_dir, "mulliken_charges.csv")
        self.potential_energy_path = os.path.join(self.data_dir, "potential_energy.csv")
        self.scalar_coupling_contributions_path = os.path.join(
            self.data_dir, "scalar_coupling_contributions.csv"
        )
        self.structures_path = os.path.join(self.data_dir, "structures.csv")

        # pickle file path
        self.pickle_dir = os.path.join(self.data_dir, "pickle")

        self.feature_names = []
