import os

import pickle

from .config import Config
from .db import LocalFile
from .preprocess import Preprocessor


class DatasetCreator:
    def __init__(self):
        self.config = Config()
        self.preprocessor = Preprocessor()

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
        train, test, structures = self.preprocessor(train, test, structures)

        # create dataset
        dataset = Dataset(
            train,
            test,
            submission,
            dipole_moments,
            magnetic_shielding_tensors,
            mulliken_charges,
            potential_energy,
            scalar_coupling_contributions,
            structures,
        )

        # save dataset object to pickle
        dataset.save(self.config.pickle_dir)
        return dataset


class Dataset:
    def __init__(
        self,
        train,
        test,
        submission,
        dipole_moments,
        magnetic_shielding_tensors,
        mulliken_charges,
        potential_energy,
        scalar_coupling_contributions,
        structures,
    ):
        self.train = train
        self.test = test
        self.submission = submission
        self.dipole_moments = dipole_moments
        self.magnetic_shielding_tensors = magnetic_shielding_tensors
        self.mulliken_charges = mulliken_charges
        self.potential_energy = potential_energy
        self.scalar_coupling_contributions = scalar_coupling_contributions
        self.structures = structures

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        pickle.dump(self, open(os.path.join(dir, "dataset.pkl"), "wb"))
        print("save the dataset pickle")

    @classmethod
    def load(cls, dir):
        print("load the dataset pickle")
        return pickle.load(open(os.path.join(dir, "dataset.pkl"), "rb"))
