import argparse
import pytorch_lightning as pl
import torch
from src.models import (
    FashionMNIST_for_CBM,
    UtoY_model,
)
from src.utils import get_component_with_dicts, load_config, dict_to_csv
from pathlib import Path
import time
from pandas import DataFrame, read_csv, Series

import sage
import pandas as pd
import torch.nn as nn


def prepare_test_data(
    csv_file: str, subsample_fraction: float = 1.0
) -> tuple[DataFrame, Series, dict, list]:
    """
    Prepare test data from a CSV file for predictions.

    Parameters:
        csv_file (str): Path to the CSV file.
        subsample_fraction (float): Fraction of the dataset to keep (1.0 means no subsampling).

    Returns:
        test_x (pd.DataFrame): Feature data for testing.
        test_y (pd.Series): Labels for testing.
        column_groups (dict): Dictionary with column group names and column lists.
    """
    import numpy as np

    # Step 1: Load the dataset
    df = pd.read_csv(csv_file)

    # Step 2: Identify column groups
    all_columns = list(df.columns)
    label_idx = all_columns.index("labels")
    s_dim_start_idx = all_columns.index("s_dim_0")

    # Group 1: Columns between "label" and "s_dim_0"
    group1_columns = all_columns[label_idx + 1 : s_dim_start_idx]
    # Group 2: Columns from "s_dim_0" to the end
    group2_columns = all_columns[s_dim_start_idx:]

    column_groups_names = {"concepts": group1_columns, "side_channel": group2_columns}

    # Step 3: Subsample the dataset if required
    if subsample_fraction < 1.0:
        df = df.sample(frac=subsample_fraction, random_state=42).reset_index(drop=True)

    # Step 4: Extract test features and labels
    test_x = df[group1_columns + group2_columns]
    test_y = df["labels"]

    ### From sage airbnb.ipynb example:
    feature_names = group1_columns + group2_columns
    group_names = [group for group in column_groups_names]
    cols = test_x.columns.tolist()
    for col in feature_names:
        if np.all([col not in group[1] for group in column_groups_names.items()]):
            group_names.append(col)

    # Group indices
    groups = []
    for _, group in column_groups_names.items():
        ind_list = []
        for feature in group:
            ind_list.append(cols.index(feature))
        groups.append(ind_list)

    return test_x, test_y, column_groups_names, groups


def group_importance_metric(explanation_values: dict[str, float]) -> float:
    if len(explanation_values) != 2:
        raise ValueError(
            "More than just concepts and side channel. Maybe there is a mistake somewhere?"
        )
    total_sum = sum(explanation_values.values())
    return explanation_values["concepts"] / total_sum


def my_main(config_path: Path) -> None:
    config = load_config(config_path)
    config_path = Path(config_path)
    config_simple_name_version = (
        config_path.stem.split("_")[0] + "_" + config_path.stem.split("_")[1]
    )
    config_folder = config_path.parent.name

    # Load dataset
    test_x, test_y, column_groups_names, groups = prepare_test_data(
        config["dataset_path"], 0.1
    )

    # Variables from config
    seed = config["seed"]

    # Dataset variables
    dataset_name = config["dataset_name"]

    # model variables
    model_name = config["model_name"]
    hyperparams = config["hyperparameters"]

    # trainer variables
    mode = config["mode"]

    default_root_dir = (
        Path(config["paths"]["default_root_dir"])
        / dataset_name
        / mode
        / model_name
        / config_folder
    )
    metrics_dir = default_root_dir / Path(config["paths"]["metric_dir"])

    # Seed the randomness
    pl.seed_everything(seed, workers=True)

    model_class = get_component_with_dicts("model", model_name)

    checkpoint_path = Path(config["paths"]["input_model_path"])

    hyperparams_model2 = config["hyperparameters_model2"]
    num_hyperparameters = {
        key: hyperparams_model2[key]
        for key in [
            "num_classes",
            "num_exogenous",
            "num_side_channel",
            "num_concepts",
        ]
    }
    if "DAG_file" in config["paths"]:
        df_dag = read_csv(config["paths"]["DAG_file"], index_col=0)
        # Convert the DataFrame to a NumPy array, then to a boolean (torch.bool) tensor
        whole_causal_graph = torch.tensor(df_dag.values, dtype=torch.bool)
        # model2 = UtoY_model(**hyperparams_model2, causal_graph=whole_causal_graph)

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model1=FashionMNIST_for_CBM().concept_extractor,
        model2=UtoY_model(**hyperparams_model2, causal_graph=whole_causal_graph),
        **num_hyperparameters,
        **hyperparams,
    )

    starting_time = time.time()  # change this with pytorch's time measurement?
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    # IMPORTANT: add softmax at the end so the model gives non-encoded output
    explained_model_with_activation = nn.Sequential(
        model.u_to_CY.last_layer, nn.Softmax(dim=1)
    )

    imputer = sage.GroupedMarginalImputer(
        explained_model_with_activation, test_x[:512], groups
    )
    estimator = sage.PermutationEstimator(imputer, "cross entropy", random_state=seed)
    sage_values = estimator(
        test_x, test_y, batch_size=config["batch_size"], thresh=0.05
    )
    test_time = time.time() - starting_time

    if True:
        sage_values.plot(list(column_groups_names.keys()))
        import matplotlib.pyplot as plt

        plt.show()

    explanation_values = dict(zip(column_groups_names, sage_values.values))
    my_metric = group_importance_metric(explanation_values)

    results = {
        "seed": config["seed"],
        "experiment_name": config_simple_name_version,
        "test_time": test_time,
        "sage_metric": my_metric,
        **model.hparams,
    }
    print("\n\n metric:", my_metric)
    dict_to_csv(results, metrics_dir, config_path, parents=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")

    my_main(args.config)
