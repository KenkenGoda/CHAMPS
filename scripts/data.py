from config import Config
from load import RawData
from preprocess import Preprocessor


class DatasetCreator:
    def __init__(self):
        self.config = Config()
        self.preprocessor = Preprocessor()

    def run(self):
        raw = RawData()
        train, test = self.preprocessor(raw)
        dataset = Dataset(train, test)
        return dataset

class Dataset:
    def __init__(self, train, test):
        self.train = train
        self.test = test
