import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_component_with_dicts, load_config


def get_experiment_dir(config_path: str, config: dict) -> Path:
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    mode = config["mode"]
    config_folder = Path(config_path).parent.name
    exp_name = config.get(
        "experiment_name",
        Path(config_path).stem.split("_")[0] + "_" + Path(config_path).stem.split("_")[1],
    )
    return (
        Path(config["paths"]["default_root_dir"])
        / dataset_name
        / mode
        / model_name
        / config_folder
        / exp_name
    )


def image_id_from_path(img_path: str) -> str:
    return Path(str(img_path)).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config["seed"])

    dataset_class = get_component_with_dicts("dataset", config["dataset_name"])
    datamodule = dataset_class(**config["dataset_params"])
    datamodule.setup(stage="fit" if args.split in ("train", "val") else "test")

    dataset_attr = {
        "train": "train_dataset",
        "val": "val_dataset",
        "test": "test_data",
    }[args.split]
    dataset = getattr(datamodule, dataset_attr)

    image_ids = [image_id_from_path(item["img_path"]) for item in dataset.data]

    if args.output:
        output_path = Path(args.output)
    else:
        exp_dir = get_experiment_dir(args.config, config)
        output_path = exp_dir / "last_metrics" / f"{args.split}_image_ids.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(image_ids) + "\n")

    print(f"Saved {len(image_ids)} {args.split} image IDs to {output_path}")


if __name__ == "__main__":
    main()
