import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(
    y_true,
    y_pred
):

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

    }

    return metrics


def topk_accuracy(
    outputs,
    labels,
    k=5
):

    _, pred = outputs.topk(
        k,
        dim=1
    )

    correct = pred.eq(
        labels.view(-1,1)
    )

    return correct.any(dim=1).float().mean().item()