from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset


class CachedFeatureBackbone(nn.Module):
    """Pass precomputed frozen-backbone features to the trainable CREAM head."""

    def __init__(self) -> None:
        super().__init__()
        self.concept_extractor = nn.Identity()


class CachedFeatureDataset(Dataset):
    def __init__(self, source: Dataset, cache_path: str | Path) -> None:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.features: Tensor = cache["features"]
        self.concepts: Tensor = cache["concepts"]
        self.labels: Tensor = cache["labels"]

        if isinstance(source, Subset):
            self.indices = [int(index) for index in source.indices]
        else:
            self.indices = list(range(len(source)))
        if self.indices and max(self.indices) >= len(self.features):
            raise ValueError(
                f"Feature cache {cache_path} has {len(self.features)} rows but "
                f"dataset index {max(self.indices)} was requested."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        cached_index = self.indices[index]
        return (
            self.features[cached_index],
            self.concepts[cached_index],
            self.labels[cached_index],
        )


class CachedFeatureDataModule(pl.LightningDataModule):
    """Wrap an existing CREAM data module and replace images with cached features."""

    def __init__(self, base: pl.LightningDataModule, cache_dir: str | Path) -> None:
        super().__init__()
        self.base = base
        self.cache_dir = Path(cache_dir)

    def prepare_data(self) -> None:
        self.base.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        stages = ("fit", "test") if stage is None else (stage,)
        for current_stage in stages:
            self.base.setup(current_stage)
            if current_stage == "fit":
                self._wrap_fit_datasets()
            if current_stage == "test":
                self._wrap_test_datasets()
            if current_stage == "predict":
                self._wrap_predict_dataset()

    def _wrap_fit_datasets(self) -> None:
        if hasattr(self.base, "Fashionmnist_train"):
            self._wrap("Fashionmnist_train", "train")
            self._wrap("Fashionmnist_val", "train")
        else:
            self._wrap("train_dataset", "train")
            self._wrap("val_dataset", "val")

    def _wrap_test_datasets(self) -> None:
        if hasattr(self.base, "Fashionmnist_test"):
            self._wrap("Fashionmnist_test", "test")
        else:
            self._wrap("test_data", "test")

    def _wrap_predict_dataset(self) -> None:
        if hasattr(self.base, "Fashionmnist_predict"):
            self._wrap("Fashionmnist_predict", "test")
        elif hasattr(self.base, "predict_data"):
            self._wrap("predict_data", "test")

    def _wrap(self, attribute: str, split: str) -> None:
        if not hasattr(self.base, attribute):
            return
        source = getattr(self.base, attribute)
        if source is not None and not isinstance(source, CachedFeatureDataset):
            setattr(
                self.base,
                attribute,
                CachedFeatureDataset(source, self.cache_dir / f"{split}.pt"),
            )

    def train_dataloader(self) -> DataLoader:
        return self.base.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.base.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.base.test_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.base.predict_dataloader()

    def __getattr__(self, name: str) -> Any:
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)
