import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import get_component_with_dicts, load_config


def make_data_module(config: dict):
    dataset_class = get_component_with_dicts("dataset", config["dataset_name"])
    params = dict(config["dataset_params"])
    if "FMNIST" in config["dataset_name"]:
        params["seed"] = config["seed"]
        if config["dataset_name"] == "Complete_Concept_FMNIST":
            params["full_concepts"] = True
    return dataset_class(**params)


def cache_split(
    dataset: Dataset,
    feature_extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    output_path: Path,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    concepts = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            images, batch_concepts, batch_labels = batch
            features.append(feature_extractor(images.to(device, non_blocking=True)).cpu())
            concepts.append(torch.as_tensor(batch_concepts).cpu().float())
            labels.append(torch.as_tensor(batch_labels).cpu())

    payload = {
        "features": torch.cat(features).contiguous(),
        "concepts": torch.cat(concepts).contiguous(),
        "labels": torch.cat(labels).contiguous(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved {len(payload['features'])} rows to {output_path} "
        f"(feature shape={tuple(payload['features'].shape)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen-backbone features and targets for a dataset."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = make_data_module(config)
    dataset.prepare_data()
    if hasattr(dataset, "train_transform") and hasattr(dataset, "test_transform"):
        # Cache one deterministic view per image so all graph variants see identical inputs.
        dataset.train_transform = dataset.test_transform
    dataset.setup("fit")
    dataset.setup("test")

    model_class = get_component_with_dicts("model", config["model_name"])
    backbone = model_class.load_from_checkpoint(
        checkpoint_path=config["paths"]["input_model_path"],
        dataset=config["dataset_name"],
        frozen=True,
    )
    feature_extractor = backbone.concept_extractor.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    feature_extractor.eval()
    device = next(feature_extractor.parameters()).device

    batch_size = int(config["dataset_params"]["batch_size"])
    workers = int(config["dataset_params"].get("workers", 0))

    if hasattr(dataset, "Fashionmnist_train"):
        # Train/validation are fixed subsets of this one training dataset.
        train_dataset = dataset.Fashionmnist_train.dataset
        test_dataset = dataset.Fashionmnist_test
        cache_split(
            train_dataset,
            feature_extractor,
            device,
            batch_size,
            workers,
            args.output_dir / "train.pt",
        )
        cache_split(
            test_dataset,
            feature_extractor,
            device,
            batch_size,
            workers,
            args.output_dir / "test.pt",
        )
    else:
        for split, attribute in (
            ("train", "train_dataset"),
            ("val", "val_dataset"),
            ("test", "test_data"),
        ):
            cache_split(
                getattr(dataset, attribute),
                feature_extractor,
                device,
                batch_size,
                workers,
                args.output_dir / f"{split}.pt",
            )


if __name__ == "__main__":
    main()
