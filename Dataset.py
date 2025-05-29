import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class horse_zebra_dataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)

        self.length_dataset = max(
            len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = Image.open(zebra_path).convert("RGB")
        horse_img = Image.open(horse_path).convert("RGB")

        if self.transform:
            zebra_img = self.transform(zebra_img)
            horse_img = self.transform(horse_img)

        return zebra_img, horse_img
