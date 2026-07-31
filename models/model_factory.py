from models.efficientnet import build_model as efficientnet_b0
from models.convnext import build_model as convnext_tiny


def create_model(model_name, num_classes):

    if model_name == "efficientnet":

        return efficientnet_b0(num_classes)

    elif model_name == "convnext":

        return convnext_tiny(num_classes)

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )