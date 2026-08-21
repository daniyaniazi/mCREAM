import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from typing import Any
from PIL import Image
import pandas as pd
import torch


class AWA2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        seed: int = 44,
        transform: Any = None,
        return_concepts: bool = True,
        return_images: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_concepts = return_concepts
        self.return_images = return_images

        split_path = os.path.join(root_dir, f"{split}_seed_{seed}.pkl")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_path} not found!")

        self.data = pd.read_pickle(split_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        item = self.data[idx]

        label = item["class_label"]
        concepts = item["attribute_label"]

        if not self.return_images:
            return torch.tensor(concepts, dtype=torch.float), label

        img = Image.open(item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if not self.return_concepts:
            return img, label
        else:
            return img, torch.tensor(concepts, dtype=torch.float), label


class AWA2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/AwA2",
        batch_size: int = 64,
        workers: int = 2,
        return_concepts: bool = True,
        return_images: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.split_seed = 44  # splits were generated with seed 44
        self.return_concepts = return_concepts
        self.return_images = return_images

        self.train_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                # transforms.RandomResizedCrop(299),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                # transforms.CenterCrop(299),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = AWA2Dataset(
                self.data_dir,
                split="train",
                seed=self.split_seed,
                transform=self.train_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )
            self.val_dataset = AWA2Dataset(
                self.data_dir,
                split="val",
                seed=self.split_seed,
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )

        if stage == "test":
            self.test_data = AWA2Dataset(
                self.data_dir,
                split="test",
                seed=self.split_seed,
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )

        if stage == "predict":
            self.predict_data = AWA2Dataset(
                self.data_dir,
                split="test",
                seed=self.split_seed,
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=(self.workers > 0),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=(self.workers > 0),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=(self.workers > 0),
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=(self.workers > 0),
            pin_memory=True,
        )
