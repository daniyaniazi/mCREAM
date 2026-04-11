import argparse
import pytorch_lightning as pl
import torch
from src.models import (
    Template_CBM_MultiClass,
)
from src.cream_with_propagating_interventions import UtoY_model_propagating_interventions
from src.utils import get_component_with_dicts, load_config, dict_to_csv
from pathlib import Path
from pandas import read_csv
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def interventions(model, dataset, config, pl_results_dir):
    from src.saving_intermediate_utils import save_activation_percentiles
    from json import load

    if config["hyperparameters_model2"]["concept_representation"] in (
        "group_soft",
        "soft",
        "logits",
    ):
        # get the dataframe with the 95th and 5th percentiles of the concepts
        intervention_percentiles_df = save_activation_percentiles(
            dataset=dataset,
            dataset_name=config["dataset_name"],
            model=model,
            DAG_path=config["paths"]["DAG_file"],
        )
        model.intervention_percentile_df = intervention_percentiles_df

    model.interventions = True
    max_epochs = config["trainer_param"]["max_epochs"]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        default_root_dir=pl_results_dir,
        enable_progress_bar=False,
        deterministic=True,
        logger=False,
    )
    all_results = []  # collect everything here for one run

    for group_interventions in (False,):
        if group_interventions is True and "softmax_mask" in config["paths"]:
            if "softmax_mask" in config["paths"]:
                with open(config["paths"]["softmax_mask"], "r") as file:
                    softmax_mask = load(file)
                total_interventions = len(softmax_mask)
        elif group_interventions is True and "softmax_mask" not in config["paths"]:
            continue
        else:
            total_interventions = config["hyperparameters_model2"]["num_concepts"]
        model.u_to_CY.group_interventions = group_interventions

        for num_interventions in range(
            total_interventions + 1
        ):  # added +1 because of range
            
            model.num_interventions = num_interventions

            trainer.test(
                model, dataloaders=dataset, verbose=False
            )  # test the model immediately
            test_metrics = {
                key: value.item() for key, value in trainer.callback_metrics.items()
            }

            # store one "block" of results
            all_results.append(
                {
                    "group_interventions": group_interventions,
                    "num_interventions": num_interventions,
                    "metrics": test_metrics,
                }
            )

    return all_results


def my_main(config_path: Path) -> None:
    gradient_clip_val = None  # used for gradient clipping
    gradient_clip_algorithm = None  # used for gradient clipping
    config = load_config(config_path)
    config_path = Path(config_path)
    config_simple_name_version = (
        config_path.stem.split("_")[0] + "_" + config_path.stem.split("_")[1]
    )
    config_folder = config_path.parent.name

    # Load configuration

    # Variables from config
    seed = config["seed"]

    # Dataset variables
    dataset_name = config["dataset_name"]
    assert dataset_name in ("Concept_FMNIST","Complete_Concept_FMNIST")

    # model variables
    model_name = config["model_name"]
    hyperparams = config["hyperparameters"]

    # trainer variables
    mode = config["mode"]
    max_epochs = config["trainer_param"]["max_epochs"]

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

    dataset_class = get_component_with_dicts("dataset", dataset_name)

    if "FMNIST" in dataset_name:  # FMNIST has seed in its constructor
        if dataset_name == "Complete_Concept_FMNIST":
            dataset = dataset_class(
                **config["dataset_params"], seed=seed, full_concepts=True
            )
        else:
            dataset = dataset_class(**config["dataset_params"], seed=seed)
    else:
        dataset = dataset_class(**config["dataset_params"])

    model_class = get_component_with_dicts("model", model_name)

    if "train" in mode:  ## Training a "Standard" model
        # first model
        if "input_model_path" not in config["paths"]:
            # training from scratch
            hyperparams_model1 = {
                "num_classes": config["hyperparameters_model2"]["num_classes"],
                "learning_rate": config["hyperparameters"]["learning_rate"],
            }
            model1 = model_class(**hyperparams_model1)
        else:  # using pretrained model (Resnet)
            checkpoint_path = Path(config["paths"]["input_model_path"])
            pretrained_model = model_class.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset=dataset_name,
                frozen=config["hyperparameters"]["frozen_model1"],
            )
            model1 = pretrained_model

        # second model
        hyperparams_model2 = config["hyperparameters_model2"]
        if "DAG_file" in config["paths"]:
            df_dag = read_csv(config["paths"]["DAG_file"], index_col=0)
            # Convert the DataFrame to a NumPy array, then to a boolean (torch.bool) tensor
            whole_causal_graph = torch.tensor(df_dag.values, dtype=torch.bool)

            if "softmax_mask" in config["paths"]:
                from json import load

                with open(config["paths"]["softmax_mask"], "r") as file:
                    softmax_mask = load(file)
                model2 = UtoY_model_propagating_interventions(
                    **hyperparams_model2,
                    causal_graph=whole_causal_graph,
                    mutually_exclusive_concepts=softmax_mask,
                )
            else:
                model2 = UtoY_model_propagating_interventions(
                    **hyperparams_model2, causal_graph=whole_causal_graph
                )

        else:
            # raise NotImplementedError
            model2 = UtoY_model_propagating_interventions(**hyperparams_model2)

        num_hyperparameters = {
            key: hyperparams_model2[key]
            for key in [
                "num_classes",
                "num_exogenous",
                "num_side_channel",
                "num_concepts",
                "concept_representation",
            ]
        }

        if config["hyperparameters_model2"]["concept_representation"] in (
            "hard",
            "group_hard",
        ):
            gradient_clip_val = 100
            gradient_clip_algorithm = "norm"

        model = Template_CBM_MultiClass(
            model1.concept_extractor,
            model2,
            **num_hyperparameters,
            **hyperparams,
        )

        # training time
        from pytorch_lightning.callbacks import LearningRateMonitor

        callback_list = []
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callback_list.append(lr_monitor)


        trainer = pl.Trainer(
            max_epochs=max_epochs,
            default_root_dir=default_root_dir / config_simple_name_version,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            callbacks=callback_list,
            enable_progress_bar=False,
            # limit_train_batches=2,
            # limit_val_batches=2,
            # limit_test_batches=2,
        )
        pl_checkpoint_path = trainer.logger.log_dir

        # model training/validation
        trainer.fit(model, dataset)
        val_train_metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items()
        }  # returns whatever i log, not their tensors

        # model testing
        trainer.test(model, dataloaders=dataset)  # test the model immediately
        test_metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items()
        }


        results = {
            "max_epochs": max_epochs,
            "seed": config["seed"],
            "experiment_name": config_simple_name_version,
            **val_train_metrics,  # unpacks the validation/training metrics dictionary
            **test_metrics,  # unpacks the test metrics dictionary
            **model.hparams,
        }


        ### interventions
        if config["mode"] != "train_x2y":  # not blackbox

            intervention_results = interventions(
                model,
                dataset,
                config,
                default_root_dir / config_simple_name_version,
            )
            results.update({"propagated_intervention_results": intervention_results})


        dict_to_csv(results, metrics_dir, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    print("\nUsing config:", str(args.config))
    my_main(args.config)
