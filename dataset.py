import os
import torch
import random
import numpy as np
from PIL import Image
from image import load_data
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class ListDataset(Dataset):
    """
    custom dataset class for loading images
    """
    def __init__(self, root, shape=None, shuffle=True, transform=None,
                 train=False, seen=0, batch_size=1, num_workers=4):
        """
        root: list of images
        """
        if train:
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        """
        # of samples.
        """
        return self.nSamples

    def __getitem__(self, index):
        """
        return tuple of image and target
        """
        if index >= len(self):
            raise IndexError("Index out of range")

        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)

        # img = 255.0 * F.to_tensor(img)
        # img[0, :, :] = img[0, :, :] - 92.8207477031
        # img[1, :, :] = img[1, :, :] - 95.2757037428
        # img[2, :, :] = img[2, :, :] - 104.877445883

        if self.transform:
            img = self.transform(img)
        return img, target
