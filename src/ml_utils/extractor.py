import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rich.progress import Progress, TaskID
from torch.utils.data import DataLoader

from ..dataset import CustomImageDataset
from ..prompt import select_prompt
from .encoder import Encoder

logger = logging.getLogger(__name__)


class FeatureExtractor:

    def __init__(self, configs: dict, backbone: str = "") -> None:
        self.configs: dict = configs
        self.model: Optional[Encoder] = None
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
            logger.exception(
                f"Error occurred while setting the model: {backbone=}, {e}"
            )
            sys.exit(1)

    def _set_dataset(self, path: Path) -> CustomImageDataset:
        if not self.model:
            logger.exception("Model not set. Call 'set_encoder' first.")
            sys.exit(1)

        return CustomImageDataset(
            main_dir=path,
            input_sz=self.configs["backbone"][self.model.backbone]["input"],
            blur_kernel=self.configs["dataset"]["blur_kernel"],
            blur_sigma=self.configs["dataset"]["blur_sigma"],
        )

    def extract(self, path: Path) -> tuple[np.ndarray, List[Path]]:
        """
        Extract the image features.

        Args:
            path (list): directory path

        Returns:
            NDArray: (n_samples, n_features)
            list: list of images name
        """
        if not self.model:
            logger.exception("Model not set. Call 'set_encoder' first.")
            sys.exit(1)

        dataset: CustomImageDataset = self._set_dataset(path)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.configs["dataset"]["batch_size"],
            shuffle=False,
        )
        features = np.empty(
            (0, self.configs["backbone"][self.model.backbone]["output"])
        )

        with Progress() as progress:
            task_id: TaskID = progress.add_task(
                "[cyan]Extracting features: ", total=len(dataloader)
            )

            with torch.no_grad():
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
            "Select the backbone of the encoder:", choices=Encoder._AVAILABLE_BACKBONES
        )
