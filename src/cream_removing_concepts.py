from src.models import UtoY_model,Template_CBM_MultiClass
from typing import Any, Optional, Union, Callable
import torch
from torch import Tensor,BoolTensor




class UtoY_model_removed_concepts(UtoY_model):
    def __init__(self, 
                 num_exogenous: int, 
                 num_concepts: int, 
                 num_side_channel: int, 
                 num_classes: int, 
                 learning_rate: float = 0, 
                 lambda_weight: Any = None, 
                 causal_graph: BoolTensor | None = None, 
                 masking_algorithm: str = "zuko", 
                 num_hidden_layers_in_maskedmlp: int = 0, 
                 previous_model_output_size: int | None = None, 
                 last_layer_mask: bool = False, 
                 concept_representation: str = "logit", 
                 side_dropout: bool = False, 
                 dropout_prob: float = 0, 
                 mutually_exclusive_concepts: list | None = None, 
                 concepts_to_remove: Optional[list] = None,
                 **kwargs: Any) -> None:
        
        super().__init__(num_exogenous, 
                         num_concepts, 
                         num_side_channel, 
                         num_classes, 
                         learning_rate, 
                         lambda_weight, 
                         causal_graph, 
                         masking_algorithm, 
                         num_hidden_layers_in_maskedmlp, 
                         previous_model_output_size, 
                         last_layer_mask, 
                         concept_representation, 
                         side_dropout, 
                         dropout_prob, 
                         mutually_exclusive_concepts, 
                         **kwargs)

        self.concepts_to_remove=concepts_to_remove


class Template_CBM_MultiClass_removed_concepts(Template_CBM_MultiClass):
    def training_step(self, batch: Tensor, batch_idx: int) -> Any:
        """Defines a single training step for the model."""
        batch[1]=batch[1][:,self.u_to_CY.num_removed_concepts:] # keep only last remaining concepts
        (
            _,
            _,
            concept_loss,
            task_loss,
            concept_acc,
            task_acc,
            total_loss,
            task_loss_percentile,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {
                "train_task_loss": task_loss,
                "train_task_accuracy": task_acc,
                "train_concept_loss": concept_loss,
                "train_concept_accuracy": concept_acc,
                "train_total_loss": total_loss,
                "train_task_loss_percent": task_loss_percentile,
            },
            prog_bar=True,
        )

        # if self.num_side_channel > 0:
        #     self.log(
        #         "train_avg_cross_correlation",
        #         self.u_to_CY.cross_concept_latent_avg_corr,
        #     )

        return total_loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> tuple[Tensor, Tensor]:
        """Defines a single validation step for the model."""
        batch[1]=batch[1][:,self.u_to_CY.num_removed_concepts:] # keep only last remaining concepts
        (
            task_preds,
            concept_preds,
            concept_loss,
            task_loss,
            concept_acc,
            task_acc,
            total_loss,
            task_loss_percentile,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {
                "val_task_loss": task_loss,
                "val_task_accuracy": task_acc,
                "val_concept_loss": concept_loss,
                "val_concept_accuracy": concept_acc,
                "val_total_loss": total_loss,
                "val_task_loss_percent": task_loss_percentile,
            },
            prog_bar=True,
        )

        # if self.num_side_channel > 0:
        #     self.log(
        #         "val_avg_cross_correlation",
        #         self.u_to_CY.cross_concept_latent_avg_corr,
        #     )

        return task_preds, concept_preds

    def test_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Defines a single testing step for the model."""
        batch[1]=batch[1][:,self.u_to_CY.num_removed_concepts:] # keep only last remaining concepts
        (
            task_preds,
            concept_preds,
            concept_loss,
            task_loss,
            concept_acc,
            task_acc,
            total_loss,
            task_loss_percentile,
        ) = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {
                "test_task_loss": task_loss,
                "test_task_accuracy": task_acc,
                "test_concept_loss": concept_loss,
                "test_concept_accuracy": concept_acc,
                "test_total_loss": total_loss,
                "test_task_loss_percent": task_loss_percentile,
            },
            prog_bar=True,
        )

        # if self.num_side_channel > 0:
        #     self.log(
        #         "test_avg_cross_correlation",
        #         self.u_to_CY.cross_concept_latent_avg_corr,
        #     )

        return task_preds, concept_preds

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Defines a single testing step for the model."""
        batch[1]=batch[1][:,self.u_to_CY.num_removed_concepts:] # keep only last remaining concepts

        # drop everything from side-channel
        self.u_to_CY.side_channel[-1].p = 1
        self.u_to_CY.side_channel[-1].train()
        # print(
        #     "Dropout is in training mode 2:", self.u_to_CY.side_channel[-1].training
        # )  # Should be False in eval mode

        (
            task_preds,
            concept_preds,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._get_preds_loss_accuracy(batch)

        try:
            self.all_preds.append(task_preds)  # Store batch predictions
            self.all_labels.append(batch[2])  # Store batch labels
        except:
            self.all_preds = []  # Store all predictions
            self.all_labels = []  # Store all ground truths

        # reset it to the normal probability
        self.u_to_CY.side_channel[-1].p = self.u_to_CY.dropout_prob
        # Log loss and metric
        return task_preds, concept_preds
