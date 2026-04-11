import pytorch_lightning as pl
from pandas import read_csv, DataFrame
from pytorch_lightning.callbacks import Callback
import os


class LogIntermediateLayerCallback_Inputs(Callback):
    def __init__(
        self,
        dataset_name: str,
        DAG_path: str,
        seed: int,
        training_set: bool,
        save_directory: str,
    ):
        super().__init__()
        self.hooks = []  # type: ignore
        self.data = {  # type: ignore
            "datapoint_idx": [],
            "labels": [],
        }  # Initialize dictionary for storing outputs
        self.exogenous_data = {  # type: ignore
            "datapoint_idx": [],
            "labels": [],
        }  # Initialize dictionary for storing outputs
        self.dataset_name = dataset_name
        self.percentile_df = DataFrame()
        self.num_concepts = 0
        self.DAG_path = DAG_path
        self.seed = seed
        self.training = training_set
        self.save_directory = (
            save_directory  # THE CHECKPOINT DIR FROM THE TRAINER USED IN TRAINING
        )

        # config_simple_name_version: str <-- USED TO SAVE AS FILE
        self.output_folder = "internal_representations"

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Register hooks to capture the outputs of layers
        def hook_fn(module, input, output):  # type: ignore
            num_concepts = getattr(pl_module, "num_concepts", 0)
            self.num_concepts = num_concepts
            assert num_concepts != 0
            num_side_channel = getattr(pl_module, "num_side_channel", 0)
            try:
                side_dropout = getattr(pl_module.u_to_CY, "side_dropout", 0)
                masking_alg = getattr(pl_module.u_to_CY, "masking_algorithm", 0)
            except:
                pass

            # Save input of last_layer
            self.c = input[0][:, :num_concepts]
            if num_side_channel > 0:
                self.s = input[0][:, num_concepts:]  # Get the rest for 'side_channel'
            # CBM + side channel
            elif side_dropout is True and masking_alg == "none":
                self.s = input[0][:, num_concepts:]  # Get the rest for 'side_channel'
            else:
                self.s = None  # If no side channel, set to None

        def exogenous_hook_fn(module, input, output):  # type: ignore
            self.u = output

        self.hooks.append(
            pl_module.u_to_CY.u2u_model.register_forward_hook(exogenous_hook_fn)
        )
        self.hooks.append(pl_module.u_to_CY.last_layer.register_forward_hook(hook_fn))

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore
        # Log c and s (if available)

        batch_size = batch[0].size(0)  # Assume batch[0] is the input tensor
        true_concepts = batch[1]
        labels = batch[2]
        for i in range(batch_size):
            # Save datapoint index
            datapoint_idx = batch_idx * batch_size + i
            self.data["datapoint_idx"].append(datapoint_idx)
            self.exogenous_data["datapoint_idx"].append(datapoint_idx)

            # Save the label for the current datapoint
            self.data["labels"].append(
                labels[i].item()
            )  # Convert the label to a scalar value
            self.exogenous_data["labels"].append(labels[i].item())

            # Save true concepts for the current datapoint
            concept_values = (
                true_concepts[i].detach().cpu().numpy()
            )  # Convert to NumPy array
            for dim_idx, value in enumerate(concept_values):
                column_name = f"true_concept_dim_{dim_idx}"
                if column_name not in self.data:
                    self.data[column_name] = []
                self.data[column_name].append(value)

            if hasattr(self, "u"):
                # Process and save c for the current datapoint
                u_values = self.u[i].detach().cpu().numpy()  # Convert to NumPy array
                for dim_idx, value in enumerate(u_values):
                    column_name = f"u_dim_{dim_idx}"
                    if column_name not in self.exogenous_data:
                        self.exogenous_data[column_name] = []
                    self.exogenous_data[column_name].append(value)

            if hasattr(self, "c"):
                # Process and save c for the current datapoint
                c_values = self.c[i].detach().cpu().numpy()  # Convert to NumPy array
                for dim_idx, value in enumerate(c_values):
                    column_name = f"c_dim_{dim_idx}"
                    if column_name not in self.data:
                        self.data[column_name] = []
                    self.data[column_name].append(value)

            if hasattr(self, "s"):
                if self.s is not None:
                    s_values = (
                        self.s[i].detach().cpu().numpy()
                    )  # Convert to NumPy array
                    for dim_idx, value in enumerate(s_values):
                        column_name = f"s_dim_{dim_idx}"
                        if column_name not in self.data:
                            self.data[column_name] = []
                        self.data[column_name].append(value)

    def on_test_end(self, trainer, pl_module):  # type: ignore
        df = DataFrame(self.data)
        self.exogenous_df = DataFrame(self.exogenous_data)

        # Split DataFrame into two:
        # Step 1. DataFrame with predicted c and s and intervention percentiles
        c_columns = [col for col in df.columns if col.startswith("c_dim_")]
        s_columns = [col for col in df.columns if col.startswith("s_dim_")]
        self.latent_df_c_s = df[["datapoint_idx", "labels"] + c_columns + s_columns]

        df_percentiles_interventions = compute_percentiles_for_c_dimensions(
            self.latent_df_c_s
        )
        df_percentiles_interventions = rename_columns(
            df_percentiles_interventions,
            self.dataset_name,
            num_concepts=self.num_concepts,
            concept_path=self.DAG_path,
        )
        self.latent_df_c_s = rename_columns(
            df=self.latent_df_c_s,
            dataset_name=self.dataset_name,
            num_concepts=self.num_concepts,
            concept_path=self.DAG_path,
        )

        self.df_all_activations = rename_all_columns(
            df=df,
            dataset_name=self.dataset_name,
            num_concepts=self.num_concepts,
            concept_path=self.DAG_path,
        )

        self.percentile_df = df_percentiles_interventions

        # Step 2. DataFrame with true concepts and s
        true_concept_columns = [
            col for col in df.columns if col.startswith("true_concept_dim_")
        ]
        self.latent_df_true_c_s = df[
            ["datapoint_idx", "labels"] + true_concept_columns + s_columns
        ]

        # Rename true_concept_dim_* to c_dim_* in the second DataFrame
        column_rename_map = {
            col: col.replace("true_concept_dim_", "c_dim_")
            for col in true_concept_columns
        }
        self.latent_df_true_c_s = self.latent_df_true_c_s.rename(
            columns=column_rename_map
        )

        self.latent_df_true_c_s = rename_columns(
            df=self.latent_df_true_c_s,
            dataset_name=self.dataset_name,
            num_concepts=self.num_concepts,
            concept_path=self.DAG_path,
        )

        # # Save DataFrames to CSV
        if self.seed is not None:
            if self.training:
                base_name = f"seed{self.seed}_train_set"
            else:
                base_name = f"seed{self.seed}_test_set"

            output_path = os.path.join(
                self.save_directory,
                f"perc_{base_name}_C_and_S.csv",
            )
            df_percentiles_interventions.to_csv(output_path, index=False)

            output_path = os.path.join(self.save_directory, f"{base_name}_C_and_S.csv")
            self.latent_df_true_c_s.to_csv(output_path, index=False)

            output_path = os.path.join(
                self.save_directory, f"{base_name}_concept_activations_C_and_S.csv"
            )
            self.df_all_activations.to_csv(output_path, index=False)

            output_path = os.path.join(
                self.save_directory, f"{base_name}_exogenous.csv"
            )
            self.exogenous_df.to_csv(output_path, index=False)

        # Remove the hooks to avoid memory leak
        for hook in self.hooks:
            hook.remove()


def rename_columns(
    df: DataFrame, dataset_name: str, num_concepts: int, concept_path: str
) -> DataFrame:
    # Predefined rename dictionaries

    rename_dicts = {}

    adj_matrix = read_csv(concept_path, index_col=0)
    column_names = adj_matrix.columns[:num_concepts]
    rename_dicts[dataset_name] = {
        f"c_dim_{i}": col_name for i, col_name in enumerate(column_names)
    }

    # Validate the name
    if dataset_name not in rename_dicts:
        raise ValueError(
            f"Name '{dataset_name}' not found in predefined rename dictionaries."
        )

    # Make sure you are using correct values
    assert num_concepts == len(rename_dicts[dataset_name])

    # Get the corresponding rename dictionary
    rename_dict = rename_dicts[dataset_name]

    # Rename columns using the dictionary
    df_new = df.rename(columns=rename_dict)

    return df_new


def rename_all_columns(
    df: DataFrame, dataset_name: str, num_concepts: int, concept_path: str
) -> DataFrame:
    # Predefined rename dictionaries

    rename_dicts = {}

    adj_matrix = read_csv(concept_path, index_col=0)
    column_names = adj_matrix.columns[:num_concepts]
    rename_dicts[dataset_name] = {}

    for i, col_name in enumerate(column_names):
        rename_dicts[dataset_name][f"c_dim_{i}"] = col_name
        rename_dicts[dataset_name][f"true_concept_dim_{i}"] = f"true_{col_name}"

    # Validate the name
    if dataset_name not in rename_dicts:
        raise ValueError(
            f"Name '{dataset_name}' not found in predefined rename dictionaries."
        )

    # Make sure you are using correct values
    assert 2 * num_concepts == len(rename_dicts[dataset_name])

    # Get the corresponding rename dictionary
    rename_dict = rename_dicts[dataset_name]

    # Rename columns using the dictionary
    df_new = df.rename(columns=rename_dict)

    return df_new


def compute_percentiles_for_c_dimensions(df: DataFrame) -> DataFrame:
    """
    Computes the 5th and 95th percentiles for each 'c' dimension in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'c' dimensions (e.g., 'c_dim_0', 'c_dim_1').

    Returns:
        pd.DataFrame: DataFrame with percentile statistics for each 'c' dimension.
    """
    # Filter for columns related to 'c'
    c_columns = [col for col in df.columns if col.startswith("c_dim_")]

    # Compute 5th and 95th percentiles for each 'c' column
    percentiles = {"dimension": [], "5th_percentile": [], "95th_percentile": []}  # type: ignore

    for col in c_columns:
        percentiles["dimension"].append(col)
        percentiles["5th_percentile"].append(df[col].quantile(0.05))
        percentiles["95th_percentile"].append(df[col].quantile(0.95))

    # Create a DataFrame with the percentile results
    percentiles_df = DataFrame(percentiles)

    return percentiles_df


def save_intermediate_values(
    dataset: pl.LightningDataModule,
    dataset_name: str,
    model: pl.LightningModule,
    training_set: bool,
    DAG_path: str,
    seed: int,
    save_directory: str,
) -> tuple[DataFrame, DataFrame]:
    """Used for the calculation of sage values. Returns 2 dataframes: (c_pred, side channel) and (c_true, side channel)

    Args:
        training_set (bool): Whose activation values am I saving? The train set's or test's?

    """
    if training_set is True:
        train_dataset = dataset.train_dataloader()
    else:
        train_dataset = dataset.test_dataloader()
    train_callback = LogIntermediateLayerCallback_Inputs(
        dataset_name, DAG_path, seed, training_set, save_directory
    )
    trainer = pl.Trainer(
        callbacks=[train_callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.test(model, dataloaders=train_dataset, verbose=False)
    train_latent = (
        train_callback.latent_df_c_s
    )  # predicted (on train/test set) concept values
    train_true = (
        train_callback.latent_df_true_c_s
    )  # true (on train/test set) concept values

    return train_latent, train_true


def save_activation_percentiles(
    dataset: pl.LightningDataModule,
    dataset_name: str,
    model: pl.LightningModule,
    DAG_path: str,
) -> DataFrame:
    """Used for the 95th and 5th percentiles of the concept activations, in the non-hard CBM cases."""
    dataset.setup(stage="fit")
    train_dataset = dataset.train_dataloader()
    train_callback = LogIntermediateLayerCallback_Inputs(
        dataset_name, DAG_path, None, None, None
    )
    trainer = pl.Trainer(
        callbacks=[train_callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.test(model, dataloaders=train_dataset, verbose=False)
    intervention_values = train_callback.percentile_df

    return intervention_values


# def my_main(config_path: Path) -> None:
#     config = load_config(config_path)
#     config_path = Path(config_path)
#     # config_simple_name_version = (
#     #     config_path.stem.split("_")[0] + "_" + config_path.stem.split("_")[1]
#     # )

#     # Load configuration

#     # Variables from config
#     seed = config["seed"]

#     # Dataset variables
#     dataset_name = config["dataset_name"]
#     # batch_size = config["dataset_params"]["batch_size"]
#     # workers = config["dataset_params"]["workers"]

#     # model variables
#     model_name = config["model_name"]
#     hyperparams = config["hyperparameters"]

#     # trainer variables
#     mode = config["mode"]

#     # Seed the randomness
#     pl.seed_everything(seed, workers=True)

#     dataset_class = get_component_with_dicts("dataset", dataset_name)

#     if "MNIST" in dataset_name:  # FMNIST has seed in its constructor
#         dataset = dataset_class(**config["dataset_params"], seed=seed)
#     else:
#         dataset = dataset_class(**config["dataset_params"])

#     model_class = get_component_with_dicts("model", model_name)

#     if mode == "predict_cbm":
#         checkpoint_path = Path(config["paths"]["input_model_path"])

#         hyperparams_model2 = config["hyperparameters_model2"]
#         num_hyperparameters = {
#             key: hyperparams_model2[key]
#             for key in [
#                 "num_classes",
#                 "num_exogenous",
#                 "num_side_channel",
#                 "num_concepts",
#             ]
#         }
#         if "DAG_file" in config["paths"]:
#             df_dag = read_csv(config["paths"]["DAG_file"], index_col=0)
#             whole_causal_graph = torch.tensor(df_dag.values, dtype=torch.bool)

#         model = model_class.load_from_checkpoint(
#             checkpoint_path=checkpoint_path,
#             model1=FashionMNIST_for_CBM().concept_extractor,
#             model2=UtoY_model(**hyperparams_model2, causal_graph=whole_causal_graph),
#             **num_hyperparameters,
#             **hyperparams,
#         )

#         # Define which layers you want to extract outputs from
#         # layer_names = {
#         #     "u_to_CY.u2c_model": "output_1",
#         #     "u_to_CY.side_channel": "output_2",
#         # }

#         dataset.prepare_data()
#         dataset.setup(stage="fit")
#         train_dataset = dataset.train_dataloader()

#         mycallback = LogIntermediateLayerCallback_Inputs(
#             dataset_name, DAG_path=config["paths"]["DAG_file"]
#         )
#         trainer = pl.Trainer(callbacks=[mycallback])
#         trainer.test(model, dataloaders=train_dataset)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Training Script")
#     parser.add_argument(
#         "--config", type=str, required=True, help="Path to the YAML configuration file"
#     )
#     args = parser.parse_args()
#     torch.set_float32_matmul_precision("high")

#     my_main(args.config)
# ## RUN: python save_latent_dataset.py --config yaml_configs/config_20_just_predict.yaml
