from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Any, Optional

# Note - you must have torchvision installed for this example
from torchvision.datasets import CelebA
import torch

# transformations masking
from PIL import Image
import PIL
import os
import numpy as np


class CustomCelebA(CelebA):
    @property
    def base_folder(self) -> str:
        return "CelebA"

    def __init__(
        self,
        root: str,
        attributes: list[str],
        class_attribute: str,
        split: str = "train",
        target_type: str = "attr",
        transform: Any = None,
        target_transform: Any = None,
        download: bool = False,
        return_concepts: bool = True,
        return_images: bool = True,
    ):
        """
        Custom CelebA wrapper to return (image, list of attributes, class).

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split to use ('train', 'valid', 'test').
            target_type (str): The target type, default is 'attr'.
            transform (Any): Transformations for the image.
            target_transform (Any): Transformations for the target.
            download (bool): Whether to download the dataset.
            attributes (List[str]): List of attribute names to extract (e.g., ['Smiling', 'Wearing_Hat']).
            class_attribute (str): The attribute name to use as the class (e.g., 'Male').
        """
        super().__init__(
            root=root,
            split=split,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.attributes = attributes
        self.class_attribute = class_attribute
        self.return_concepts = return_concepts
        self.return_images = return_images

        # Create a mapping of attribute names to their indices
        self._attr_idx_map = {name: idx for idx, name in enumerate(self.attr_names)}

    def __getitem__(
        self, index: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image, list of attributes, class), where:
                - image: The transformed image.
                - list of attributes: A list of binary values for the specified attributes.
                - class: The binary value for the class attribute.
        """
        # Get the image and attributes from the parent class
        # IMAGE IS ALREADY TRANSFORMED HERE!
        image, attributes = super().__getitem__(index)

        # Extract the desired attributes
        attribute_values = [
            int(attributes[self._attr_idx_map[name]] > 0) for name in self.attributes
        ]

        # Extract the class value
        class_value = int(attributes[self._attr_idx_map[self.class_attribute]] > 0)

        if not self.return_images:
            return torch.tensor(attribute_values, dtype=torch.float), torch.tensor(
                [class_value], dtype=torch.float
            )
        elif not self.return_concepts:
            return (
                image,
                torch.tensor([class_value], dtype=torch.float),
            )
        else:
            return (
                image,
                torch.tensor(attribute_values, dtype=torch.float),
                torch.tensor([class_value], dtype=torch.float),
            )


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        class_name: Optional[str] = None,
        concept_names: Optional[list[str]] = None,
        data_dir: str = "data",
        batch_size: int = 32,
        workers: int = 1,
        return_concepts: bool = True,
        return_images: bool = True,
        masking: bool = False,
        **kwargs: Any,
    ):
        """
        PyTorch Lightning DataModule for the CelebA dataset.
        Args:
            data_dir (str): Root directory for the CelebA dataset.
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
        self.masking = masking

        # hardcoded code for the experiments
        if (class_name in ["unfair", "fairness", "c2bm"]) and (concept_names is None):
            if class_name == "unfair":
                concept_names = [
                    "Arched_Eyebrows",
                    "Bags_Under_Eyes",
                    "Double_Chin",
                    "High_Cheekbones",
                    "Mouth_Slightly_Open",
                    "Narrow_Eyes",
                    "Rosy_Cheeks",
                ]
                class_name = "Smiling"
            elif class_name == "c2bm":
                concept_names = [
                    "Attractive",
                    "Big_Lips",
                    "Heavy_Makeup",
                    "High_Cheekbones",
                    "Male",
                    "Oval_Face",
                    "Smiling",
                    "Wavy_Hair",
                    "Wearing_Lipstick",
                ]
                class_name = "Mouth_Slightly_Open"
            elif class_name == "fairness":
                concept_names = [
                    "Arched_Eyebrows",
                    "Bags_Under_Eyes",
                    "Double_Chin",
                    "High_Cheekbones",
                    "Mouth_Slightly_Open",
                    "Narrow_Eyes",
                    "Rosy_Cheeks",
                    "Male",
                    "Blond_Hair",
                ]
                class_name = "Smiling"

            else:
                concept_names = [
                    "Black_Hair",
                    "Blond_Hair",
                    "Brown_Hair",
                    "Gray_Hair",
                ]
                class_name = "Male"
        else:
            raise NotImplementedError

        assert concept_names is not None
        assert class_name is not None
        self.concept_names = concept_names
        self.class_name = class_name

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
            self.train_dataset = CustomCelebA(
                attributes=self.concept_names,
                class_attribute=self.class_name,
                root=self.data_dir,
                split="train",
                transform=self.train_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )
            self.val_dataset = CustomCelebA(
                attributes=self.concept_names,
                class_attribute=self.class_name,
                root=self.data_dir,
                split="valid",
                transform=self.test_transform,
                return_concepts=self.return_concepts,
                return_images=self.return_images,
            )

        if stage == "test":
            if self.masking is True:
                self.test_data = MaskedCeleba(
                    attributes=self.concept_names,
                    class_attribute=self.class_name,
                    root=self.data_dir,
                    split="test",
                    transform=self.test_transform,
                    return_concepts=self.return_concepts,
                    return_images=self.return_images,
                )

            else:
                self.test_data = CustomCelebA(
                    attributes=self.concept_names,
                    class_attribute=self.class_name,
                    root=self.data_dir,
                    split="test",
                    transform=self.test_transform,
                    return_concepts=self.return_concepts,
                    return_images=self.return_images,
                )

        if stage == "predict":
            self.predict_data = CustomCelebA(
                attributes=self.concept_names,
                class_attribute=self.class_name,
                root=self.data_dir,
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


class MaskedCeleba(CustomCelebA):
    def __getitem__(
        self, index: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor] | tuple[Any, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image, list of attributes, class), where:
                - image: The transformed image.
                - list of attributes: A list of binary values for the specified attributes.
                - class: The binary value for the class attribute.
        """
        # Get the image and attributes from the parent class

        ##### CelebA code #####
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        attributes: Any = []
        attributes.append(self.attr[index, :])
        nose_y = self.landmarks_align[index, 5]  # nose_y coordinate

        ##### Erase Pixels Above nose_y BEFORE Transformation #####
        # Convert the image to a numpy array
        X_np = np.array(X)

        # Mask the pixels above the nose_y (set them to black)
        X_np[nose_y:, :, :] = 0  # Set all pixels above the nose_y to black (0,0,0)

        # Convert back to a PIL Image
        X = Image.fromarray(X_np)
        ###########################################################

        if self.transform is not None:
            X = self.transform(X)

        if attributes:
            attributes = tuple(attributes) if len(attributes) > 1 else attributes[0]

            if self.target_transform is not None:
                attributes = self.target_transform(attributes)
        else:
            attributes = None

        ##### concepts code #####

        # IMAGE IS ALREADY TRANSFORMED HERE!
        image, attributes = X, attributes

        # Extract the desired attributes
        attribute_values = [
            int(attributes[self._attr_idx_map[name]] > 0) for name in self.attributes
        ]

        # Extract the class value
        class_value = int(attributes[self._attr_idx_map[self.class_attribute]] > 0)

        if not self.return_images:
            return torch.tensor(attribute_values, dtype=torch.float), torch.tensor(
                [class_value], dtype=torch.float
            )
        elif not self.return_concepts:
            return (
                image,
                torch.tensor([class_value], dtype=torch.float),
            )
        else:
            return (
                image,
                torch.tensor(attribute_values, dtype=torch.float),
                torch.tensor([class_value], dtype=torch.float),
            )
