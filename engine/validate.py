import torch

def validate(
        model,
        dataloader,
        criterion,
        device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

    loss = running_loss / len(dataloader)
    acc = correct / total

    return loss, acc