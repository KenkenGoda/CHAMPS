from collections import namedtuple

from .config import Config
from .db import LocalFile
from .preprocess import Preprocessor


class DatasetCreator:
    def __init__(self):
        self.config = Config()

    def run(self):
        # load files
        db = LocalFile(self.config)
        train = db.get_train()
        test = db.get_test()
        submission = db.get_submission()
        dipole_moments = db.get_dipole_moments()
        magnetic_shielding_tensors = db.get_magnetic_shielding_tensors()
        mulliken_charges = db.get_mulliken_charges()
        potential_energy = db.get_potential_energy()
        scalar_coupling_contributions = db.get_scalar_coupling_contributions()
        structures = db.get_structures()

        # preprocess data
        preprocessor = Preprocessor()
        train, test, structures = preprocessor.run(train, test, structures)

        # create dataset
        Dataset = namedtuple(
            "Dataset",
            [
                "train",
                "test",
                "submission",
                "dipole_moments",
                "magnetic_shielding_tensors",
                "mulliken_charges",
                "potential_energy",
                "scalar_coupling_contributions",
                "structures",
            ],
        )
        dataset = Dataset(
            train,
            test,
            submission,
            dipole_moments.set_index("molecule_name"),
            magnetic_shielding_tensors.set_index("molecule_name"),
            mulliken_charges.set_index("molecule_name"),
            potential_energy.set_index("molecule_name"),
            scalar_coupling_contributions.set_index(
                ["molecule_name", "atom_index_0", "atom_index_1"]
            ),
            structures.set_index("molecule_name"),
        )
        return dataset
