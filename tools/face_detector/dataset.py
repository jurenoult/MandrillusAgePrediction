import os
import glob
from PIL import Image
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import pytorch_lightning as pl
import numpy as np


class DAVISDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        img_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "Annotations")

        self.samples: List[Tuple[str, str]] = []
        for vid in sorted(os.listdir(img_dir)):
            frames = sorted(glob.glob(os.path.join(img_dir, vid, "*.jpg")))
            for f in frames:
                fname = os.path.basename(f).replace(".jpg", ".png")
                mask_path = os.path.join(mask_dir, vid, fname)
                if os.path.exists(mask_path):
                    self.samples.append((f, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask


class SAM2LikeTransform:
    def __init__(self, is_train=True):
        if is_train:
            self.img_transform = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.3, 0.3, 0.3, 0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.img_transform = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.mask_transform = T.Compose(
            [
                T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
            ]
        )

    def __call__(self, img, mask):
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = np.array(mask) / 255.0
        mask = torch.as_tensor(mask, dtype=torch.int64)
        return img, mask


class DAVISDataModule(pl.LightningDataModule):
    def __init__(self, root: str, batch_size=2, val_split=0.2, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = DAVISDataset(self.root, transform=SAM2LikeTransform(is_train=True))
        val_len = int(len(dataset) * self.val_split)
        train_len = len(dataset) - val_len

        self.train_set, self.val_set = random_split(dataset, [train_len, val_len])
        self.val_set.dataset.transform = SAM2LikeTransform(is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
