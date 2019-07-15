import os


class Config:
    def __init__(self, config_file=None):
        # raw data path
        self.data_dir = "../data"
        self.train_path = os.path.join(self.data_dir, "train.csv")
        self.test_path = os.path.join(self.data_dir, "test.csv")
        self.sample_submission_path = os.path.join(
            self.data_dir, "sample_submission.csv"
        )
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

        # pickle directory path
        self.pickle_dir = os.path.join(self.data_dir, "pickle")

        # submission file path
        self.submission_path = "../results/submission.csv"

        self.feature_names = [
            "MoleculeCount",
            "MoleculeDistanceStatistics",
            "Atom0Count",
            "Atom1Count",
        ]

        self.target_name = ["fc"]

        # random seed
        self.seed = 42
