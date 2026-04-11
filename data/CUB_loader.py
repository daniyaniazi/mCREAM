import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from typing import Any
from PIL import Image
import pandas as pd
import torch


class CUBDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Any = None,
        return_concepts: bool = True,
        return_images: bool = True,
    ):
        """
        Custom Dataset for the CUB Dataset.
        Args:
            root_dir (str): Root directory containing the dataset.
            split (str): One of "train", "val", or "test".
            transform (callable, optional): Transform to be applied to the images.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_concepts = return_concepts
        self.return_images = return_images

        # Load the corresponding split data
        split_path = os.path.join(
            root_dir, "CUB_processed", "class_attr_data_10", f"{split}.pkl"
        )
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_path} not found!")

        self.data = pd.read_pickle(split_path)

        for item in self.data:
            # Replace the hard-coded path prefix with the correct one
            item["img_path"] = item["img_path"].replace(
                "/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011",
                ".",
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.root_dir, "CUB_200_2011", item["img_path"])

        img = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Retrieve labels and concepts
        # print(item)
        label = item["class_label"]
        concepts = item["attribute_label"]

        if not self.return_images:
            return torch.tensor(concepts, dtype=torch.float), label
        elif not self.return_concepts:
            return img, label
        else:
            return img, torch.tensor(concepts, dtype=torch.float), label


class CUBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/CUB",
        batch_size: int = 32,
        workers: int = 1,
        return_concepts: bool = True,
        return_images: bool = True,
        **kwargs: Any,
    ):
        """
        PyTorch Lightning DataModule for the CUB dataset.
        Args:
            data_dir (str): Root directory for the CUB dataset.
            batch_size (int): Batch size for DataLoader.
            input_seed (int): Random seed for reproducibility.
            workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.return_concepts = return_concepts
        self.return_images = return_images

        # Following the transformations from CBM paper
        resol = 299
        # Define image transformations
        self.train_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
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
                transforms.CenterCrop(resol),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = CUBDataset(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )
            self.val_dataset = CUBDataset(
                self.data_dir,
                split="val",
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )

        if stage == "test":
            self.test_data = CUBDataset(
                self.data_dir,
                split="test",
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )
        if stage == "predict":
            self.predict_data = CUBDataset(
                self.data_dir,
                split="test",
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
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )
