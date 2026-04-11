import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
import pandas as pd
import os
from PIL import Image
from typing import Any


class FashionMNISTDataModule(pl.LightningDataModule):
    """Modified from MNIST data module, https://lightning.ai/docs/pytorch/stable/data/datamodule.html"""

    def __init__(
        self,
        data_dir: str = "./data/FashionMNIST",
        batch_size: int = 32,
        input_seed: int = 42,
        train_size: int = 55000,
        workers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir

        # transformation from: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=nn%20crossentropyloss
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = input_seed
        self.workers = workers
        self.total_train_instances = 60000

    def prepare_data(self) -> None:
        # download data
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            Fashionmnist_full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.Fashionmnist_train, self.Fashionmnist_val = random_split(
                Fashionmnist_full,
                [self.train_size, self.total_train_instances - self.train_size],
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Fashionmnist_test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.Fashionmnist_predict = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.Fashionmnist_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.Fashionmnist_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.Fashionmnist_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.Fashionmnist_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
            pin_memory=True,
        )


# Custom Dataset for loading images and concept vectors
class ConceptFashionMNISTDataset(FashionMNIST):
    @property
    def raw_folder(self) -> str:
        return os.path.join(
            self.root, "FashionMNIST", "raw"
        )  # changed from MNIST implementation to fool it

    @property
    def processed_folder(self) -> str:
        return os.path.join(
            self.root, "FashionMNIST", "processed"
        )  # changed from MNIST implementation to fool it

    def __init__(
        self,
        root: str,
        concept_dir: str,
        train: bool,
        transform: Any = None,
        return_labels: bool = True,
        return_images: bool = True,
        full_concepts: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            return_labels (bool): If we want labels. Used for X->C model. Defaults to True.
            return_images (bool): If we want images. Used for C->Y model. Defaults to True.
            full_concepts (bool): If we want to use the complete concept set. Defaults to False.
        """
        super().__init__(root=root, train=train, transform=transform, **kwargs)
        self.concept_dir = concept_dir
        self.return_labels = return_labels
        self.return_images = return_images
        self.full_concepts = full_concepts

        if os.path.exists(concept_dir):
            self.concepts = self._load_concepts_from_tensor(self.concept_dir)
        else:
            # Load from CSV if tensor file is not found
            self.concepts = self._load_concepts_from_csv(self.concept_dir)

    def _load_concepts_from_tensor(self, concept_dir: str) -> torch.Tensor:
        if self.full_concepts:
            print("\n\nLoading the full concept vectors")
            concept_file = (
                f"{'train' if self.train else 'test'}_COMPLETE_concept_tensor.pt"
            )
        else:
            print("\n\nLoading the original concept vectors")
            concept_file = f"{'train' if self.train else 'test'}_concept_tensor.pt"
        concepts_tensor = torch.load(
            os.path.join(concept_dir, concept_file), weights_only=True
        )

        return concepts_tensor

    def _load_concepts_from_csv(self, concept_dir: str) -> list[torch.Tensor]:
        """Based on _load_data found in pytorch MNIST dataset implementation. Way slower."""
        concept_file = (
            f"{'train' if self.train else 'test'}_concept_vectors_with_index.csv"
        )
        concepts_df = pd.read_csv(os.path.join(concept_dir, concept_file))
        concepts = concepts_df.drop(columns=["Index"])

        concepts_tensors = [
            torch.tensor(row.values, dtype=torch.int)
            for index, row in concepts.iterrows()
        ]

        return concepts_tensors

    def __getitem__(
        self, index: int
    ) -> (
        tuple[Any, torch.Tensor, int]
        | tuple[Any, torch.Tensor]
        | tuple[torch.Tensor, Any]
    ):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")
        concept_vector = self.concepts[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_images:
            return concept_vector.float(), target
        if not self.return_labels:
            return img, concept_vector
        else:
            return img, concept_vector, target


class ConceptFashionMNISTDataModule(FashionMNISTDataModule):
    def __init__(
        self,
        concept_dir: str = "./data/FashionMNIST/concept_vectors",
        data_dir: str = "./data/FashionMNIST",
        return_labels: bool = True,
        return_images: bool = True,
        full_concepts: bool = False,
        **kwargs: Any,
    ):
        # Load the CSV file
        super().__init__(data_dir, **kwargs)
        self.concept_dir = concept_dir
        self.return_labels = return_labels
        self.return_images = return_images
        self.full_concepts = full_concepts
        # Load the FashionMNIST dataset to access the images

    def prepare_data(self) -> None:
        # download data
        ConceptFashionMNISTDataset(
            self.data_dir,
            self.concept_dir,
            train=True,
            download=True,
            return_labels=self.return_labels,
            return_images=self.return_images,
            full_concepts=self.full_concepts,
        )
        ConceptFashionMNISTDataset(
            self.data_dir,
            self.concept_dir,
            train=False,
            download=True,
            return_labels=self.return_labels,
            return_images=self.return_images,
            full_concepts=self.full_concepts,
        )

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            Fashionmnist_full = ConceptFashionMNISTDataset(
                self.data_dir,
                self.concept_dir,
                train=True,
                transform=self.transform,
                return_labels=self.return_labels,
                return_images=self.return_images,
                full_concepts=self.full_concepts,
            )
            self.Fashionmnist_train, self.Fashionmnist_val = random_split(
                Fashionmnist_full,
                [self.train_size, self.total_train_instances - self.train_size],
                generator=torch.Generator().manual_seed(42),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Fashionmnist_test = ConceptFashionMNISTDataset(
                self.data_dir,
                self.concept_dir,
                train=False,
                transform=self.transform,
                return_labels=self.return_labels,
                return_images=self.return_images,
                full_concepts=self.full_concepts,
            )

        if stage == "predict":
            self.Fashionmnist_predict = ConceptFashionMNISTDataset(
                self.data_dir,
                self.concept_dir,
                train=False,
                transform=self.transform,
                return_labels=self.return_labels,
                return_images=self.return_images,
                full_concepts=self.full_concepts,
            )


def convert_csv_to_tensor_file(
    concept_dir: str = "./data/FashionMNIST/concept_vectors",
) -> None:
    """Also found in the notebook to generate the concept vectors."""
    # Load and preprocess the train concepts
    train_concepts_df = pd.read_csv(
        os.path.join(concept_dir, "train_concept_vectors_with_index.csv")
    )
    train_concepts = torch.stack(
        [
            torch.tensor(row[1:], dtype=torch.int)
            for _, row in train_concepts_df.iterrows()
        ]
    )  # Skip the index column

    torch.save(train_concepts, os.path.join(concept_dir, "train_concept_tensor.pt"))

    # Load and preprocess the test concepts
    test_concepts_df = pd.read_csv(
        os.path.join(concept_dir, "test_concept_vectors_with_index.csv")
    )
    test_concepts = torch.stack(
        [
            torch.tensor(row[1:], dtype=torch.int)
            for _, row in test_concepts_df.iterrows()
        ]
    )

    # Save tensors to a file
    torch.save(test_concepts, os.path.join(concept_dir, "test_concept_tensor.pt"))
