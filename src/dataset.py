import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(
        self,
        *,
        main_dir: str,
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
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            tensor_image = self.transform(image)
        return tensor_image
    
    def get_all_imgs(self) -> list[str]:
        return self.all_imgs
