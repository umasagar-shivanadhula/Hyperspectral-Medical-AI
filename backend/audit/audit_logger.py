
import json
from datetime import datetime
from pathlib import Path

class AuditLogger:

    def __init__(self):
        self.log_dir = Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, task, prediction, confidence, metadata=None):

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "task": task,
            "prediction": prediction,
            "confidence": confidence,
            "metadata": metadata or {}
        }

        filename = self.log_dir / f"audit_{datetime.utcnow().date()}.json"

        if filename.exists():
            data = json.loads(filename.read_text())
        else:
            data = []

        data.append(record)

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
