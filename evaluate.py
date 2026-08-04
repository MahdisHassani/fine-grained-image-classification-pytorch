import torch
import torch.nn as nn

from datasets.dataloader import create_dataloaders
from models.model_factory import create_model
from evaluation.evaluate import evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 196
MODEL_NAME = "convnext"      # or efficientnet

_, val_loader = create_dataloaders(
    train_csv="outputs/train_split.csv",
    val_csv="outputs/val_split.csv",
    train_dir="cars_train",
    val_dir="cars_train",
    batch_size=32
)

model = create_model(MODEL_NAME, NUM_CLASSES)

model.load_state_dict(
    torch.load(
        f"outputs/{MODEL_NAME}_best.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

metrics = evaluate_model(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=DEVICE,
    model_name=MODEL_NAME
)

print("\nEvaluation Results")
print("-"*30)

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")