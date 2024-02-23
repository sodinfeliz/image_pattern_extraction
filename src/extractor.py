import sys
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import Progress, TaskID

from .model import Model
from .dataset import CustomImageDataset
from .prompt import select_prompt

logger = logging.getLogger(__name__)


class FeatureExtractor:

    _AVAILABLE_MODELS = [
        "ResNet",
        "EfficientNet"
    ]

    def __init__(self, configs: dict, backbone: str='') -> None:
        self.configs: dict = configs
        self.model: Model = None
        self.set_model(backbone)

    def set_model(self, backbone: str) -> None:
        """ 
        Set the backbone for the model,
        only "EfficientNet", and "ResNet" available.

        Args:
            backbone (str): backbone of model
        """
        if backbone in self._AVAILABLE_MODELS:
            self.model = Model(backbone) if backbone else Model()
            self.model.start_eval()
        else:
            logger.exception(f"Invalid Backbone '{backbone}'.")
            sys.exit(1)

    def _set_dataset(self, path: Path) -> CustomImageDataset:
        return CustomImageDataset(
            main_dir=path,
            input_sz=self.configs['backbone'][self.model.backbone]['input'],
            blur_kernel=self.configs['dataset']['blur_kernel'],
            blur_sigma=self.configs['dataset']['blur_sigma']
        )

    def extract(self, path: Path):
        """
        Extract the image features.

        Args:
            path (list): directory path

        Returns:
            NDArray: (n_samples, n_features)
            list: list of images name
        """
        dataset: CustomImageDataset = self._set_dataset(path)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.configs['dataset']['batch_size'], 
            shuffle=False
        )
        features = np.empty((0, self.configs['backbone'][self.model.backbone]['output']))

        with (
            torch.no_grad(), 
            Progress() as progress
        ):
            task_id: TaskID = progress.add_task("[cyan]Extracting features: ", total=len(dataloader))

            for batch in dataloader:
                images = batch.to(self.model.get_device())
                out = self.model.predict(images)
                features = np.concatenate((features, out))
                progress.update(task_id, advance=1)

        return features, dataset.get_all_imgs()
    
    @classmethod
    def prompt(cls):
        return select_prompt(
            "Select the backbone of the extraction model:",
            choices=cls._AVAILABLE_MODELS
        )


