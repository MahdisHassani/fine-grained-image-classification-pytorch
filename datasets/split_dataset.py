import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

OUTPUT_DIR = Path("outputs")
df = pd.read_csv(OUTPUT_DIR / "train.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class_id"]
)

train_df.to_csv(OUTPUT_DIR/"train_split.csv", index=False)
val_df.to_csv(OUTPUT_DIR/"val_split.csv", index=False)

print(f"Train samples:{len(train_df)}")
print(f"Validation samples:{len(val_df)}")