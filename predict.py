import json
import torch
from PIL import Image
from torchvision import transforms
from models.model_factory import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 196


transform = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


MODEL_NAME = "convnext"      # or efficientnet
model = create_model(MODEL_NAME, NUM_CLASSES)

model.load_state_dict(
    torch.load(
        f"outputs/{MODEL_NAME}_best.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


with open("outputs/class_names.json","r") as f:

    class_names = json.load(f)


image = Image.open("sample.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(DEVICE)


with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs,dim=1)

    values, indices = torch.topk(
        probabilities,
        k=5
    )

print("Top 5 Predictions")
print("-"*50)

for prob, idx in zip(values[0],indices[0]):

    print(
        f"{class_names[idx]}"
        f" : {prob.item()*100:.2f}%"
    )