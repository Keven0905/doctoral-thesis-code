from torch import nn
from torchvision import models


MODEL_NAMES = {"M1": "alexnet", "M2": "resnet50", "M3": "googlenet",
               "M4": "densenet121", "M5": "convnext_tiny", "M6": "efficientnet_b0"}
CHAIN = ["M2", "M4", "M1", "M3", "M5", "M6"]
WEIGHTS = {"M1": "AlexNet_Weights.IMAGENET1K_V1",
           "M2": "ResNet50_Weights.IMAGENET1K_V2",
           "M3": "GoogLeNet_Weights.IMAGENET1K_V1",
           "M4": "DenseNet121_Weights.IMAGENET1K_V1",
           "M5": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
           "M6": "EfficientNet_B0_Weights.IMAGENET1K_V1"}


def build_model(model_id, num_classes, pretrained=True):
    weights = models.get_weight(WEIGHTS[model_id]) if pretrained else None
    options = {"weights": weights}
    if model_id == "M3":
        options.update(aux_logits=pretrained, init_weights=False, transform_input=True)
    model = models.get_model(MODEL_NAMES[model_id], **options)
    if model_id in ("M2", "M3"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_id == "M4":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        layer = model.classifier[-1]
        model.classifier[-1] = nn.Linear(layer.in_features, num_classes)
    if model_id == "M3":
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
    return model
