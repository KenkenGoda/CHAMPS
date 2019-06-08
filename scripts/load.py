import pandas as pd

class RawData:
    def __init__(self):
        self.train = pd.read_csv("../data/train.csv")
        self.test = pd.read_csv("../data/test.csv")
        self.submission = pd.read_csv("../data/sample_submission.csv")
        self.dipole = pd.read_csv('../data/dipole_moments.csv')
        self.magnetic = pd.read_csv('../data/magnetic_shielding_tensors.csv')
        self.mulliken = pd.read_csv('../data/mulliken_charges.csv')
        self.potential = pd.read_csv('../data/potential_energy.csv')
        self.structures = pd.read_csv('../data/structures.csv')