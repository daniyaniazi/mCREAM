import argparse
import pytorch_lightning as pl
import torch
from src.cream_removing_concepts import (
    UtoY_model_removed_concepts,
    Template_CBM_MultiClass_removed_concepts,
)
from src.utils import get_component_with_dicts, load_config, dict_to_csv, run_benchmark
from pathlib import Path
import time
from pandas import read_csv
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from collections import defaultdict


def group_rows_by_prefix(df):
    groups = defaultdict(list)

    for pos, idx in enumerate(df.index):
        if "::" in idx:
            prefix = idx.split("::", 1)[0]
            groups[prefix].append(pos)

    # keep only groups with at least 2 rows
    return [positions for positions in groups.values() if len(positions) >= 2]


def first_index_of_each_group(df):
    # helper function
    groups = defaultdict(list)

    for pos, idx in enumerate(df.index):
        if "::" in idx:
            prefix = idx.split("::", 1)[0]
            groups[prefix].append(pos)

    # take the first position of each group
    return [positions[0] for positions in groups.values()]


def cumulative_sublists(indices):
    # helper function
    result = []
    temp = []
    for idx in indices:
        temp.append(idx)
        result.append(temp.copy())  # copy to avoid mutation
    return result


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

    for group_interventions in (True, False):
        if group_interventions is True and "softmax_mask" in config["paths"]:
            if config["dataset_name"] == "CelebA":
                continue  # CELEBA DOES NOT HAVE GROUP INTERVENTIONS

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
    assert dataset_name == "CUB"
    batch_size = config["dataset_params"]["batch_size"]
    workers = config["dataset_params"]["workers"]

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
        if mode == "train_cbm":
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
                # We know the DAG file is [num_concepts + num_classes x num_concepts + num_classes] in this order
                df_dag = read_csv(config["paths"]["DAG_file"], index_col=0)
                df_dag_concepts = df_dag.iloc[
                    : hyperparams_model2["num_concepts"],
                    : hyperparams_model2["num_concepts"],
                ]
                starting_index_of_each_concept_group_including_binaries = (
                    first_index_of_each_group(df_dag_concepts)
                )
                df_dag_classes = df_dag.iloc[
                    hyperparams_model2["num_concepts"] :,
                    hyperparams_model2["num_concepts"] :,
                ]
                next_index = starting_index_of_each_concept_group_including_binaries[
                    len(hyperparams_model2["concepts_to_remove"])
                ]
                df_dag = df_dag.iloc[next_index:, next_index:]
                # Convert the DataFrame to a NumPy array, then to a boolean (torch.bool) tensor
                whole_causal_graph = torch.tensor(df_dag.values, dtype=torch.bool)

                # from json import load

                # with open(config["paths"]["softmax_mask"], "r") as file:
                #     softmax_mask = load(file)
                softmax_mask = group_rows_by_prefix(df_dag)
                num_removed_concepts = -1 * (
                    df_dag.shape[0]
                    - df_dag_classes.shape[0]
                    - hyperparams_model2["num_concepts"]
                )

                concept_exogenous_multiplicity = (
                    hyperparams_model2["num_exogenous"]
                    - hyperparams_model2["num_side_channel"]
                ) // hyperparams_model2["num_concepts"]
                # print("multiplicity before class", concept_exogenous_multiplicity)
                hyperparams_model2["num_concepts"] = (
                    whole_causal_graph.shape[0] - df_dag_classes.shape[0]
                )
                hyperparams_model2["num_exogenous"] = (
                    concept_exogenous_multiplicity * hyperparams_model2["num_concepts"]
                    + hyperparams_model2["num_side_channel"]
                )

                model2 = UtoY_model_removed_concepts(
                    **hyperparams_model2,
                    causal_graph=whole_causal_graph,
                    mutually_exclusive_concepts=softmax_mask,
                )
                model2.num_removed_concepts = num_removed_concepts

            else:
                model2 = UtoY_model_removed_concepts(**hyperparams_model2)

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

            model = Template_CBM_MultiClass_removed_concepts(
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
        # if dataset_name == "CelebA":
        #     if mode == "train_x2y":
        #         monitor = "val_loss"
        #     else:
        #         monitor = "val_total_loss"
        #     early_stopping = EarlyStopping(
        #         monitor=monitor, mode="min", patience=15, min_delta=0.0
        #     )
        #     callback_list.append(early_stopping)

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
        torch.cuda.synchronize()
        starting_time = time.perf_counter()
        trainer.fit(model, dataset)
        torch.cuda.synchronize()
        process_end_time = time.perf_counter()

        training_validation_time = process_end_time - starting_time

        val_train_metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items()
        }  # returns whatever i log, not their tensors

        # model testing
        torch.cuda.synchronize()
        starting_time = time.perf_counter()
        trainer.test(model, dataloaders=dataset)  # test the model immediately
        torch.cuda.synchronize()
        process_end_time = time.perf_counter()
        test_time = process_end_time - starting_time
        test_metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items()
        }
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        from src.utils import count_maskedmlp_params

        disconnected_weights = count_maskedmlp_params(model)
        num_params -= disconnected_weights

        results = {
            "max_epochs": max_epochs,
            "seed": config["seed"],
            "experiment_name": config_simple_name_version,
            "train_val_time": training_validation_time / 60.0,  # minutes
            "test_time": test_time / 60.0,  # minutes
            "num_trainable_parameters": num_params,
            **val_train_metrics,  # unpacks the validation/training metrics dictionary
            **test_metrics,  # unpacks the test metrics dictionary
            **model.hparams,
        }

        ### efficiency metrics
        if config["mode"] not in ("train_x2y", "train_c2y", "train_backbone"):
            benchmark_results = run_benchmark(model, dataset.test_dataloader())

            results.update(benchmark_results)

            from src.saving_intermediate_utils import (
                save_intermediate_values,
            )

            train_latent = save_intermediate_values(
                dataset=dataset,
                dataset_name=dataset_name,
                model=model,
                training_set=True,
                DAG_path=config["paths"]["DAG_file"],
                seed=config["seed"],
                save_directory=pl_checkpoint_path,
            )

            ## Pass the test set through the model
            test_latent = save_intermediate_values(
                dataset=dataset,
                dataset_name=dataset_name,
                model=model,
                training_set=False,
                DAG_path=config["paths"]["DAG_file"],
                seed=config["seed"],
                save_directory=pl_checkpoint_path,
            )

        ### Side channel metrics
        if "hyperparameters_model2" in config and (
            config["hyperparameters_model2"]["num_side_channel"] > 0
            or config["hyperparameters_model2"].get("side_dropout", False)
        ):
            import sage
            from torch import nn

            from src.sage_importance_functions import (
                prepare_shap_data,
                group_importance_metric,
            )

            from src.diff_permutation_estimator import (
                PermutationEstimator as my_PermutationEstimator,
            )
            from src.PFI_accuracy import PFI_accuracies

            print("now doing testing with removed side channel:")
            trainer.predict(model, dataloaders=dataset)  # test the model immediately
            results["test_dropout_task_accuracy"] = model.dropout_test_acc.item()

            num_concepts = config["hyperparameters_model2"]["num_concepts"]

            print("PFI importances")
            concept_dropped_score, side_dropped_score = PFI_accuracies(
                model.u_to_CY.last_layer, test_latent[0], num_concepts, repeat=100
            )
            PFI_concept_importance = (
                results["test_task_accuracy"] - concept_dropped_score
            )
            PFI_side_importance = results["test_task_accuracy"] - side_dropped_score
            print("PFI concept importance:", PFI_concept_importance)
            results["PFI_concept_importance"] = PFI_concept_importance
            print("PFI side importance:", PFI_side_importance)
            results["PFI_side_importance"] = PFI_side_importance

            # if dataset_name == "CelebA":
            #     male_auc = celeba_predict_male(
            #         test_latent[1], num_concepts, dataset, n_jobs=workers
            #     )
            #     print("Side Channel to Male AUC:", male_auc)
            #     results["Male_AUC"] = male_auc

            #     male_mi = celeba_male_mi(
            #         test_latent[1],
            #         num_concepts=num_concepts,
            #         celeba_dataset=dataset,
            #         n_jobs=workers,
            #     )
            #     results["Male_MI"] = male_mi
            #     print("Side Channel to Male MI:", male_auc)

            print("\nStarting SAGE calculation\n")
            from src.utils import timeout

            @timeout(3600)
            def run_sage(concept_set, train_latent, test_latent, config, results):
                if concept_set == "true_concepts":
                    sage_df_train = train_latent[1]
                    sage_df_test = test_latent[1]

                train_x, _, train_group_names, train_groups = prepare_shap_data(
                    sage_df_train
                )
                train_x = train_x.to_numpy()

                test_x, test_y, test_group_names, test_groups = prepare_shap_data(
                    sage_df_test
                )
                test_x = test_x.to_numpy()
                test_y = test_y.to_numpy()

                assert (train_group_names == test_group_names) and (
                    train_groups == test_groups
                )

                ## sage calculation
                if config["hyperparameters_model2"]["num_classes"] == 1:
                    explained_model_with_activation = nn.Sequential(
                        model.u_to_CY.last_layer, nn.Sigmoid()
                    )
                else:
                    explained_model_with_activation = nn.Sequential(
                        model.u_to_CY.last_layer, nn.Softmax(dim=1)
                    )

                twenty_percent_index = int(len(sage_df_train) * 0.2)
                imputer = sage.GroupedMarginalImputer(
                    explained_model_with_activation,
                    train_x[:twenty_percent_index],
                    test_groups,
                )

                estimator = my_PermutationEstimator(
                    imputer, "cross entropy", random_state=seed, n_jobs=workers
                )

                sage_values = estimator(
                    test_x,
                    test_y,
                    batch_size=batch_size,
                    thresh=0.05,
                    bar=False,
                )

                explanation_values = dict(zip(test_group_names, sage_values.values))
                concept_grouped_importance = group_importance_metric(explanation_values)
                print(
                    "\n\n CCI:",
                    concept_grouped_importance,
                    "\n debugging_sage_metrics_concepts",
                    explanation_values["concepts"],
                    "\n debugging_sage_metrics_side_channel",
                    explanation_values["side_channel"],
                )

                # add them to the results dict
                results["CCI"] = concept_grouped_importance
                results["debugging_sage_metrics_concepts"] = explanation_values[
                    "concepts"
                ]
                results["debugging_sage_metrics_side_channel"] = explanation_values[
                    "side_channel"
                ]
                return results

            concept_set = "true_concepts"
            try:
                results = run_sage(
                    concept_set, train_latent, test_latent, config, results
                )
            except:
                print("Sage TimeoutError")
                results[concept_set + "_concept_grouped_importance"] = None
                results[concept_set + "_debugging_sage_metrics_concepts"] = None
                results[concept_set + "_debugging_sage_metrics_side_channel"] = None

        ### interventions
        if config["mode"] != "train_x2y":  # not blackbox
            try:
                intervention_results = interventions(
                    model,
                    dataset,
                    config,
                    default_root_dir / config_simple_name_version,
                )
                results.update({"intervention_results": intervention_results})
            except:
                pass

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
