import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "convnext"      # or efficientnet

history = pd.read_csv(f"outputs/history_{MODEL_NAME}/training_history.csv")

epochs = range(1, len(history) + 1)

# Loss
plt.figure(figsize=(8,5))
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"outputs/history_{MODEL_NAME}/loss_curve.png")
plt.show()

# Accuracy
plt.figure(figsize=(8,5))
plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"outputs/history_{MODEL_NAME}/accuracy_curve.png")
plt.show()