import sys
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.progress import Progress, TaskID

from .encoder import Encoder
from .dataset import CustomImageDataset
from .prompt import select_prompt

logger = logging.getLogger(__name__)


class FeatureExtractor:

    def __init__(self, configs: dict, backbone: str='') -> None:
        self.configs: dict = configs
        self.model: Encoder = None
        self.set_encoder(backbone)

    def set_encoder(self, backbone: str) -> None:
        """ 
        Set the backbone for the model,
        only "EfficientNet", and "ResNet" available.

        Args:
            backbone (str): backbone of model
        """
        try:
            self.model = Encoder(backbone) if backbone else Encoder()
            self.model.start_eval()
        except Exception as e:
            logger.exception(f"Error occurred while setting the model: {backbone=}, {e}")
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

        return features, dataset.get_img_paths()
    
    @classmethod
    def prompt(cls):
        """Prompt for selecting the backbone of the encoder."""
        return select_prompt(
            "Select the backbone of the encoder:",
            choices=Encoder._AVAILABLE_BACKBONES
        )


