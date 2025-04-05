import os
import torch
import numpy as np
import torchvision.transforms as T
import settings

from PIL import Image
from torch.utils.data import Dataset


class Loader(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = T.Resize((128, 128))

        self.image_filenames = sorted([os.path.join(root, name)
                                       for root, dirs, files in os.walk(image_dir)
                                       for name in files
                                       if (name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                                       name.startswith(".")])
        self.mask_filenames = sorted([os.path.join(root, name)
                                       for root, dirs, files in os.walk(mask_dir)
                                       for name in files
                                       if (name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                                       name.startswith(".")])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask = self.mask_transform(mask)
        mask = np.array(mask) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        return image, mask
