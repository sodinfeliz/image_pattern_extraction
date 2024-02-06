import torch
from torchvision.models import (resnet101, ResNet101_Weights, 
                                efficientnet_b0, EfficientNet_B0_Weights)


class Model():
    def __init__(self) -> None:
        self.backbone = ''
        self.model = None

    def set_model(self, backbone: str):
        assert isinstance(backbone, str), f"Type mismatched: Expected str but {type(backbone)}"
        assert backbone.lower() in ["resnet", "efficientnet"], f"Invalid Backbone {backbone}."
        
        self.backbone = backbone
        if self.backbone == "resnet":
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif self.backbone == "efficientnet":
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()

        return self
    
    def start_eval(self, device='cuda'):
        self.model.to(device)
        self.model.eval()

    def predict(self, X):
        return self.model.forward(X).detach().cpu().numpy()


