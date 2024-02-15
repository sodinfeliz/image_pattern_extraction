import torch
from torchvision.models import (resnet101, ResNet101_Weights, 
                                efficientnet_b0, EfficientNet_B0_Weights)


class Model():
    def __init__(self, backbone: str='resnet') -> None:
        self._backbone = ''
        self._model = None
        self.set_backbone(backbone)

    def __repr__(self):
        return str(self._model)

    @property
    def backbone(self):
        return self._backbone
    
    @backbone.setter
    def backbone(self, value):
        raise AttributeError(f"Directly modification of backbone disabled, using 'set_backbone' instead.")


    def set_backbone(self, backbone: str):
        assert isinstance(backbone, str), f"Type mismatched: Expected str but {type(backbone)}"
        assert backbone.lower() in ["resnet", "efficientnet"], f"Invalid Backbone {backbone}."
        
        self._backbone = backbone
        if self._backbone == "resnet":
            self._model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif self._backbone == "efficientnet":
            self._model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self._model.fc = torch.nn.Identity()

        return self
    
    def start_eval(self, device='cuda'):
        self._model.to(device)
        self._model.eval()

    def predict(self, X):
        return self._model.forward(X).detach().cpu().numpy()


