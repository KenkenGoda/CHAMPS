from .config import Config
from .data import Dataset


def run():
    config = Config()
    dataset = Dataset.load(config.pickle_dir)
    return dataset
