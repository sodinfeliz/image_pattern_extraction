import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskID

from .model import Model
from .dataset import CustomImageDataset


class FeatureExtractor:

    AVAILABLE_MODELS = [
        "ResNet",
        "EfficientNet"
    ]

    def __init__(self, configs: dict, backbone: str='') -> None:
        self.configs = configs
        self.model = None
        self.set_model(backbone)

    def set_model(self, backbone: str):
        """ 
        Set the backbone for the model,
        only "EfficientNet", and "ResNet" available.

        Args:
            backbone (str): backbone of model
        """
        if backbone in self.AVAILABLE_MODELS:
            self.model = Model(backbone) if backbone else Model
            self.model.start_eval()
        else:
            raise ValueError(f"Invalid Backbone {backbone}.")

    def _set_dataset(self, path) -> CustomImageDataset:
        dataset = CustomImageDataset(
            main_dir=path,
            input_sz=self.configs['backbone'][self.model.backbone]['input'],
            blur_kernel=self.configs['dataset']['blur_kernel'],
            blur_sigma=self.configs['dataset']['blur_sigma']
        )
        return dataset

    def extract(self, path: str):
        """
        Extract the image features.

        Args:
            path (list): directory path

        Returns:
            NDArray: (n_samples, n_features)
            list: list of images name
        """
        dataset = self._set_dataset(path)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.configs['dataset']['batch_size'], 
            shuffle=False
        )
        features = np.empty((0, self.configs['backbone'][self.model.backbone]['output']))

        with torch.no_grad():
            with Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                BarColumn(bar_width=None, style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                expand=True
            ) as progress:
                task_id: TaskID = progress.add_task("[cyan]Extracting features: ", total=len(dataloader))

                for batch in dataloader:
                    images = batch.to(self.model.get_device())
                    out = self.model.predict(images)
                    features = np.concatenate((features, out))
                    progress.update(task_id, advance=1)

        return features, dataset.get_all_imgs()
