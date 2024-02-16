import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Model
from .dataset import CustomImageDataset


class FeatureExtrator():
    def __init__(self, configs: dict, backbone: str='') -> None:
        self.configs = configs
        self.model = None

        self.set_model(backbone)

    def set_model(self, backbone: str):
        """ 
        Set the backbone for the model,
        only "efficientnet", and "resnet" available.

        Args:
            backbone (str): backbone of model
        """
        if not backbone:
            self.model = Model()
        else:
            self.model = Model(backbone)
        self.model.start_eval(device=self.configs['device'])

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
            for batch in tqdm(dataloader, desc='Extracting features: '):
                images = batch.to(self.configs['device'])
                out = self.model.predict(images)
                features = np.concatenate((features, out))

        if self.configs['device'] == 'cuda':
            torch.cuda.empty_cache()

        return features, dataset.get_all_imgs()
