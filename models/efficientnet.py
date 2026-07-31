import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_model(num_classes):

    model = efficientnet_b0(
        weights = EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model