import torch

def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc