import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from datasets.dataloader import create_dataloaders
from models.model_factory import create_model
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 196
EPOCHS = 15
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

train_loader, val_loader = create_dataloaders(
    train_csv="outputs/train_split.csv",
    val_csv="outputs/val_split.csv",
    train_dir="cars_train",
    val_dir="cars_train",
    batch_size=BATCH_SIZE
)

MODEL_NAME = "convnext"      # or efficientnet
model = create_model(MODEL_NAME, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
best_acc = 0

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Train Loss: {train_loss:.4f}"
        f" | Train Acc: {train_acc:.4f}"
        f" | Val Loss: {val_loss:.4f}"
        f" | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc

        torch.save(
            model.state_dict(),
            f"outputs/{MODEL_NAME}_best.pth")

print("Training Finished")


history_df = pd.DataFrame(history)

history_df.to_csv(
    f"outputs/history_{MODEL_NAME}/training_history.csv",
    index=False)

print("Training history saved.")