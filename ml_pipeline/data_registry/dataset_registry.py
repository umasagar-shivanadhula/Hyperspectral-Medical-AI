
from pathlib import Path

class DatasetRegistry:

    def __init__(self, root="datasets"):
        self.root = Path(root)

    def list_datasets(self):
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def get_dataset(self, name):
        path = self.root / name
        if not path.exists():
            raise ValueError("Dataset not found")
        return path
