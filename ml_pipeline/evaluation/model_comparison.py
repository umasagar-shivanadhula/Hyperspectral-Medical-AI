
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        })

    return pd.DataFrame(results)


def plot_model_comparison(df, output_path):

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.set_index("model").plot(kind="bar", figsize=(8,5))

    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
