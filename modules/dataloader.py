import os
import torch
import numpy as np
import settings

from PIL import Image
from torch.utils.data import Dataset


class Loader(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.mask_transform = T.Resize((128, 128))

        self.image_filenames = sorted([os.path.join(root, name)
                                       for root, dirs, files in os.walk(image_dir)
                                       for name in files
                                       if (name.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION) or
                                           name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                                       name.startswith(".")])

        self.mask_filenames = None
        if mask_dir:
            self.mask_filenames = sorted([os.path.join(root, name)
                                           for root, dirs, files in os.walk(mask_dir)
                                           for name in files
                                          if (name.endswith(settings.GEOGRAPHIC_ACCEPT_EXTENSION) or
                                              name.endswith(settings.NON_GEOGRAPHIC_ACCEPT_EXTENSION)) and not
                                           name.startswith(".")])
            if len(self.image_filenames) != len(self.mask_filenames):
                raise ValueError("Number of images and masks do not match!")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mask_filenames:
            mask_path = self.mask_filenames[idx]
            mask = Image.open(mask_path).convert("L")
            # mask = self.mask_transform(mask)
            mask = np.array(mask) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask
        return image, img_path