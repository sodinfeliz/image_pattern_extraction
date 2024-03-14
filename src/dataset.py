import os
from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        *,
        main_dir: Path,
        input_sz: int,
        blur_kernel: int,
        blur_sigma: int,
    ):
        self.main_dir = main_dir
        self.transform = transforms.Compose([
            transforms.GaussianBlur(blur_kernel, sigma=blur_sigma),
            transforms.Resize((input_sz, input_sz)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.img_paths = list(main_dir.glob("**/*.jpg"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            tensor_image = self.transform(image)
        return tensor_image
    
    def get_img_paths(self) -> list[str]:
        return self.img_paths
