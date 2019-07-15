from .config import Config
from .db import LocalFile
from .data import DatasetCreator, Dataset
from .dataproc import DataProcessor
from .predict import Prediction


class Experiment:
    def run(self):
        config = Config()
        creator = DatasetCreator()
        dataset = creator.run()
        # dataset = Dataset.load(config.pickle_dir)
        return dataset

        processor = DataProcessor(config)
        X_train, y_train, X_test = processor.run(dataset)
        return X_train, y_train, X_test

        predict = Prediction(config)
        y_pred = predict.run(X_train, y_train, X_test, n_splits=3)
        return y_pred
