from .config import Config
from .db import LocalFile


class Submission:
    def __init__(self):
        self.config = Config()
        self.save_path = self.config.submission_path

    def save(self, prediction):
        db = LocalFile(self.config)
        self.submission = db.get_submission()
        self.submission["scalar_coupling_constant"] = prediction
        self.submission.to_csv(self.save_path, index=False)
