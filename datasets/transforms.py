from torchvision import transforms

def get_train_transforms():
     
     return transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.ColorJitter(
               brightness=0.2,
               contrast=0.2,
               saturation=0.2,
               hue=0.05
          ),
          transforms.ToTensor(),
          transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
          )
     ])

def get_val_transforms():

     return transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
          )
     ])