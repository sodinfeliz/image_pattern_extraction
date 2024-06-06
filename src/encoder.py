import torch
from torchvision.models import (
    resnet101, 
    ResNet101_Weights, 
    efficientnet_b0, 
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class Encoder():

    _AVAILABLE_BACKBONES = [
        "ResNet",
        "EfficientNet",
        "MobileNetV3"
    ]

    def __init__(self, backbone: str='ResNet') -> None:
        self._backbone = ''
        self._model = None
        self._device = None
        self.set_backbone(backbone)

    def __repr__(self):
        return str(self._model)

    @property
    def backbone(self):
        return self._backbone
    
    @backbone.setter
    def backbone(self, _):
        raise AttributeError(f"Directly modification of backbone disabled, using 'set_backbone' instead.")

    def get_device(self):
        return self._device

    def set_backbone(self, backbone: str):
        if not isinstance(backbone, str):
            raise TypeError(f"Type mismatched: Expected str but got {type(backbone).__name__}")
        if backbone not in self._AVAILABLE_BACKBONES:
            raise ValueError(f"Invalid Backbone '{backbone}'")
        
        self._backbone = backbone
        try:
            if self._backbone == "ResNet":
                self._model = resnet101(weights=ResNet101_Weights.DEFAULT)
            elif self._backbone == "EfficientNet":
                self._model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            elif self._backbone == "MobileNetV3":
                self._model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self._model.fc = torch.nn.Identity()
        except Exception as e:
            raise e

        return self
    
    def start_eval(self):
        if torch.cuda.is_available():
            self._device = 'cuda'
        elif torch.backends.mps.is_available():
            self._device = 'mps'
        else:
            self._device = 'cpu'
        self._model.to(self._device)
        self._model.eval()

    def predict(self, X):
        return self._model(X).detach().cpu().numpy()
