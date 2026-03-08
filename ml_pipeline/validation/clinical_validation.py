
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClinicalValidator:

    def __init__(self):
        self.records = []

    def add_case(self, prediction, ground_truth):
        self.records.append((prediction, ground_truth))

    def compute_metrics(self):

        preds = [p for p, g in self.records]
        gt = [g for p, g in self.records]

        return {
            "accuracy": accuracy_score(gt, preds),
            "precision": precision_score(gt, preds, average="weighted"),
            "recall": recall_score(gt, preds, average="weighted"),
            "f1": f1_score(gt, preds, average="weighted")
        }
