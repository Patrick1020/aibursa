# app/calib.py
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbCalibrator:
    def __init__(self):
        self.model = None

    def fit(self, probs_0_100, outcomes_01):
        x = np.asarray(probs_0_100, dtype=float) / 100.0
        y = np.asarray(outcomes_01, dtype=int)
        self.model = IsotonicRegression(out_of_bounds="clip").fit(x, y)

    def transform(self, p):
        if self.model is None or p is None:
            return p
        return float(self.model.predict([p / 100.0])[0] * 100.0)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
