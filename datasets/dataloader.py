from torch.utils.data import DataLoader
from datasets.dataset import StanfordCarsDataset
from datasets.transforms import get_train_transforms, get_val_transforms

def create_dataloaders(
        train_csv,
        val_csv,
        train_dir,
        val_dir,
        batch_size=32
):
    train_dataset = StanfordCarsDataset(
        csv_file=train_csv,
        image_dir=train_dir,
        transform=get_train_transforms()
    )

    val_dataset = StanfordCarsDataset(
        csv_file=val_csv,
        image_dir=val_dir,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader