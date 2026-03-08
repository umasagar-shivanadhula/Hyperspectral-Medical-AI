
from pathlib import Path
import datetime
import joblib

class ModelRegistry:

    def __init__(self):
        self.model_dir = Path("outputs/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, name):

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        path = self.model_dir / f"{name}_{ts}.pkl"

        joblib.dump(model, path)
        return path
