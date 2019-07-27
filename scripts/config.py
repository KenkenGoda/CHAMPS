import os

from .param_space import IntParamSpace, UniformParamSpace, LogUniformParamSpace


class Config:
    def __init__(self, nrows=None):
        # set the number of rows for reading files
        self.nrows = nrows

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

        # pickled files path
        self.pickle_dir = "../pickle"
        self.pickled_data_dir = os.path.join(self.pickle_dir, "data")
        self.pickled_feature_dir = os.path.join(self.pickle_dir, "feature")
        self.pickled_train_path = os.path.join(self.pickled_data_dir, "train.pkl")
        self.pickled_test_path = os.path.join(self.pickled_data_dir, "test.pkl")
        self.pickled_structures_path = os.path.join(
            self.pickled_data_dir, "structures.pkl"
        )

        # submission file path
        self.submission_path = "../results/submission.csv"

        # used feature names
        self.feature_names = [
            "MoleculeCount",
            "MoleculeDistanceStatistics",
            "Atom0Count",
            "Atom1Count",
        ]

        # target name
        self.target_name = "fc"

        # study name and storage path of parameters for the best model
        self.study_name = f"lgb_{self.target_name}"
        self.storage_path = f"../database/{self.study_name}.db"

        # static parameters for model
        self.fixed_params = {
            "boosting_type": "gbdt",
            "max_depth": 20,
            "learning_rate": 1e-2,
            "n_estimators": 100000,
            "reg_alpha": 0.0,
            # "metric": "binary",
        }

        # parameter space for searching with optuna
        self.param_space = {
            "num_leaves": IntParamSpace("num_leaves", 2, 100),
            "subsample": UniformParamSpace("subsample", 0.5, 1.0),
            "subsample_freq": IntParamSpace("subsample_freq", 1, 20),
            # "colsample_bytree": LogUniformParamSpace("colsample_bytree", 1e-2, 1e-1),
            # "min_child_weight": LogUniformParamSpace("min_child_weight", 1e-3, 1e1),
            # "min_child_samples": IntParamSpace("min_child_samples", 1, 50),
            # "reg_lambda": LogUniformParamSpace("reg_lambda", 1e-1, 1e4),
        }

        # random seed
        self.seed = 42
