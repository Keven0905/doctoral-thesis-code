from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from .common import write_csv, write_json


def classification_metrics(labels, probabilities, classes):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(classes)
    if probabilities.shape != (len(labels), count) or len(labels) == 0:
        raise ValueError("Unexpected probability dimensions or empty labels.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one.")
    predictions = probabilities.argmax(axis=1)
    indices = np.arange(count)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0)
    aucs = []
    per_class = []
    for index, name in enumerate(classes):
        binary = (labels == index).astype(int)
        auc = float(roc_auc_score(binary, probabilities[:, index])) if len(np.unique(binary)) == 2 else None
        aucs.append(auc)
        per_class.append({"class": name, "index": index, "precision": float(precision[index]),
                          "recall": float(recall[index]), "f1": float(f1[index]),
                          "auc_ovr": auc, "support": int(support[index])})
    auc = float(np.mean(aucs)) if all(a is not None for a in aucs) else None
    result = {"accuracy": float(accuracy_score(labels, predictions)),
              "loss": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
              "precision": float(precision.mean()), "recall": float(recall.mean()),
              "f1": float(f1.mean()), "auc": auc,
              "auc_note": None if auc is not None else "Undefined: at least one class lacks positive or negative samples.",
              "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
              "per_class": per_class}
    return result


def export_evaluation(directory, split, dataset, probabilities, classes):
    directory = Path(directory)
    labels = [r["label"] for r in dataset.records]
    result = classification_metrics(labels, probabilities, classes)
    write_json(directory / f"{split}_metrics.json", result)
    write_csv(directory / f"{split}_per_class.csv", result["per_class"])
    rows = []
    for record, probability in zip(dataset.records, probabilities):
        row = {"path": record["path"], "true_index": record["label"],
               "predicted_index": int(probability.argmax())}
        row.update({f"p_{i}": float(p) for i, p in enumerate(probability)})
        rows.append(row)
    write_csv(directory / f"{split}_predictions.csv", rows)
    matrix = np.asarray(result["confusion_matrix"])
    write_csv(directory / f"{split}_confusion_matrix.csv",
              [{"true_index": i, "true_label": name,
                **{f"pred_{j}": int(matrix[i, j]) for j in range(len(classes))}}
               for i, name in enumerate(classes)])
    size = min(28, max(6, len(classes) * 0.25))
    figure, axis = plt.subplots(figsize=(size, size))
    display = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(display, ax=axis)
    axis.set(xlabel="Predicted class index", ylabel="True class index", title=f"{split} confusion matrix")
    if len(classes) <= 25:
        axis.set_xticks(range(len(classes)), labels=range(len(classes)))
        axis.set_yticks(range(len(classes)), labels=range(len(classes)))
        for i in range(len(classes)):
            for j in range(len(classes)):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{split}_confusion_matrix.png", dpi=160)
    plt.close(figure)
    return result


def plot_history(history, path, boundaries=()):
    if not history:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key in zip(axes, ("accuracy", "loss")):
        for split in ("train", "val"):
            axis.plot([r["step"] for r in history], [r[f"{split}_{key}"] for r in history], label=split)
        for boundary in boundaries:
            axis.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        axis.set(xlabel="Training epoch (cumulative across stages)", ylabel=key)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
