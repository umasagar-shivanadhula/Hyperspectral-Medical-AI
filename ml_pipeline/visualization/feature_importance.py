
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_feature_importance(model, feature_names, output_path):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    top_k = min(20, len(sorted_importances))

    sorted_importances = sorted_importances[:top_k]
    sorted_names = sorted_names[:top_k]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,6))

    plt.barh(sorted_names[::-1], sorted_importances[::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances (Random Forest)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
