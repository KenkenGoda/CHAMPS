from .db import LocalFile


class Submission:
    def __init__(self, config):
        self.config = config
        self.save_path = config.submission_path

    def save(self, prediction):
        db = LocalFile(self.config)
        submission = db.get_submission()
        submission["scalar_coupling_constant"] = prediction
        submission.to_csv(self.save_path, index=False)
