
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_tumor_heatmap(cube_shape, patches, probabilities, patch_size, stride):

    H, W, _ = cube_shape

    heatmap = np.zeros((H, W))
    count_map = np.zeros((H, W))

    idx = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):

            prob = probabilities[idx]

            heatmap[y:y+patch_size, x:x+patch_size] += prob
            count_map[y:y+patch_size, x:x+patch_size] += 1

            idx += 1

    heatmap = heatmap / (count_map + 1e-8)
    return heatmap


def save_heatmap(heatmap, output_path):

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar(label="Tumor Probability")
    plt.title("Tumor Probability Heatmap")
    plt.axis("off")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
