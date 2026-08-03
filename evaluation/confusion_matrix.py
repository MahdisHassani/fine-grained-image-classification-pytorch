import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    np.save(
        save_path.replace(".png", ".npy"),
        cm
    )

    plt.figure(figsize=(18,18))
    plt.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return cm