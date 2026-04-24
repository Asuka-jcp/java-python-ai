import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        model = resnet34(weights=weights)
    elif model_name == "resnet50":
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"不支持的模型: {name}. 可选: resnet18/resnet34/resnet50")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
