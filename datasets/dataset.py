from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class StanfordCarsDataset(Dataset):
    
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.image_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["class_id"]) - 1

        if self.transform:
            image  = self.transform(image)

        return image, label