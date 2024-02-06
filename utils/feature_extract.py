import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import Model
from .dataset import CustomImageDataset


def feature_extract(path: str, backbone: str, configs: dict) -> (np.ndarray, list[str]):
    extractor = Model().set_model(backbone)
    extractor.start_eval(device=configs['device'])

    dataset = CustomImageDataset(
        main_dir=path,
        input_sz=configs['backbone'][backbone]['input'],
        blur_kernel=configs['dataset']['blur_kernel'],
        blur_sigma=configs['dataset']['blur_sigma']
    )
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=configs['dataset']['batch_size'], 
        shuffle=False)

    features = np.empty((0, configs['backbone'][backbone]['output']))
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running the model inference'):
            images = batch.to(configs['device'])
            out = extractor.predict(images)
            features = np.concatenate((features, out))

    if configs['device'] == 'cuda':
        torch.cuda.empty_cache()

    return features, dataset.get_all_imgs()


class FeatureExtrator():
    def __init__(self, path: str, backbone: str, configs: dict) -> None:
        self.path = path
        self.configs = configs
        self.backbone = backbone
        self.model = None
        self.dataset = None
        self.set_backbone(backbone)

    def set_backbone(self, backbone: str):
        self.backbone = backbone
        self.model = Model().set_model(backbone)

    def set_dataset(self):
        dataset = CustomImageDataset(
            main_dir=self.path,
            input_sz=self.configs['backbone'][self.backbone]['input'],
            blur_kernel=self.configs['dataset']['blur_kernel'],
            blur_sigma=self.configs['dataset']['blur_sigma']
        )

    def extract(self,):
        ...






