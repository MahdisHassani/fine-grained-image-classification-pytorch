import torch
from evaluation.metrics import calculate_metrics, topk_accuracy
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.classification_report import save_classification_report

def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
    model_name
):

    model.eval()

    total_loss = 0.0

    y_true = []

    y_pred = []

    top5_scores = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(preds.cpu().numpy())

            top5_scores.append(

                topk_accuracy(
                    outputs,
                    labels,
                    k=5
                )
            )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=f"outputs/{model_name}_confusion_matrix.png"
    )
    
    save_classification_report(
    y_true,
    y_pred,
    save_path=f"outputs/{model_name}_classification_report.csv"
    )

    metrics["loss"] = total_loss / len(dataloader)

    metrics["top5"] = sum(top5_scores) / len(top5_scores)

    return metrics