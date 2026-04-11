from pandas import DataFrame

import torch
from torch import BoolTensor, nn


import pytorch_lightning as pl
from torch import Tensor
from torch.optim.adam import Adam
from torchmetrics.functional import accuracy
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from typing import Any, Optional, Union, Callable

from abc import ABC, abstractmethod


from torch.optim.optimizer import Optimizer
import torchvision.models as models
from zuko.nn import MaskedMLP, MaskedLinear
from torchvision.ops import StochasticDepth
import random


def binarize_mutex(concept_pred_probs, concept_groups):
    # torch.set_printoptions(profile="full")  # Ensures full tensor printing
    # torch.set_printoptions(linewidth=1000)  # Increase line width to fit full rows

    # print(concept_pred_probs)
    mutex_groups = concept_groups[0]
    non_mutex_groups = concept_groups[1]

    binarized_output = torch.zeros_like(concept_pred_probs)

    for group in mutex_groups:
        # Extract probabilities for the group
        group_probs = concept_pred_probs[:, group]

        group_hard_preds = torch.zeros_like(group_probs).scatter_(
            1, torch.argmax(group_probs, dim=-1, keepdim=True), 1.0
        )
        binarized_output[:, group] = group_hard_preds

    remaining_indices = non_mutex_groups

    if remaining_indices:
        # print(
        #     f"Applying binary straight-through to the following indices: {remaining_indices}"
        # )
        remaining_probs = concept_pred_probs[:, remaining_indices]
        hard_preds = (remaining_probs > 0.5).float()
        binarized_output[:, remaining_indices] = hard_preds

    return binarized_output


def calculate_mixed_loss(
    task_logits: Tensor,
    concept_logits: Tensor,
    target_concepts: Tensor,
    y: Tensor,
    concept_loss_function: Callable,
    task_loss_function: Callable,
    num_concepts: int,
    num_classes: int,
    lambda_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Function to calculate both task loss and concept loss in joint training. Used by CBM and u2y model."""

    target_concepts = target_concepts.float()
    concept_loss = concept_loss_function(concept_logits, target_concepts)

    concept_acc = accuracy(
        concept_logits,
        target_concepts,
        task="multilabel",
        num_labels=num_concepts,  # AUTOMATICALLY CONVERTS TO PREDICTIONS:  https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#torchmetrics.classification.MultilabelAccuracy
    )

    concept_preds = (concept_logits > 0.5).int()

    task_loss = task_loss_function(task_logits, y)

    if num_classes == 1:
        task_probs = torch.sigmoid(task_logits)

        task_preds = (task_probs > 0.5).int()

        task_acc = accuracy(task_logits, y, task="binary")
    else:
        # task prediction
        task_preds = torch.argmax(task_logits, dim=1)
        task_acc = accuracy(task_preds, y, task="multiclass", num_classes=num_classes)

    # lambda = λ
    total_loss = task_loss + lambda_weight * concept_loss

    task_loss_percentile = task_loss / total_loss * 100

    return (
        task_preds,
        concept_preds,
        concept_loss,
        task_loss,
        concept_acc,
        task_acc,
        total_loss,
        task_loss_percentile,
    )


def calculate_concept_loss(
    concept_logits: Tensor, target_concepts: Tensor, loss: Callable
) -> Tensor:
    """NOT USED ANYMORE.Function shared by x2c, cbm classes"""

    target_concepts = target_concepts.float()  # convert the concepts into float tensor

    # BCEWithLogitsLoss(reduction="none")
    samples_concept_loss = loss(concept_logits, target_concepts)
    avg_per_concept_loss = samples_concept_loss.mean(dim=1)
    # you can calculate this immediately with BCE(reduction="mean")
    concept_loss = avg_per_concept_loss.mean()

    return concept_loss


def freeze_model(model: nn.Module | pl.LightningModule) -> None:
    """Custom function to freeze model."""
    # we can also do from pl.Lightingmodule: module.freeze() -> req_grad=false and eval mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


class Template_MultiClass(pl.LightningModule, ABC):
    """Implements a multi-class classifier. Needs a model architecture as input.
    Combination of: https://docs.wandb.ai/guides/integrations/lightning and https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/backbone_image_classifier.py
    also check out: https://wandb.me/lit-colab"""

    @abstractmethod
    def __init__(self, learning_rate: float, num_classes: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        if self.num_classes == 1:
            self.loss = BCEWithLogitsLoss()
        else:
            self.loss = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        # use forward for inference/predictions
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Any:
        """Defines a single training step for the model."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {"train_loss": loss, "train_accuracy": acc},
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Defines a single validation step for the model."""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {"val_loss": loss, "val_accuracy": acc},
            prog_bar=True,
        )
        return preds

    def test_step(self, batch: Tensor) -> Tensor:
        """Defines a single testing step for the model."""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log_dict(
            {"test_loss": loss, "test_accuracy": acc},
            prog_bar=True,
        )
        return preds

    def configure_optimizers(self) -> Optimizer:
        """Defines model optimizer. Check also: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#PyTorch-Lightning for a better example"""
        return Adam(self.parameters(), lr=self.learning_rate)

    def _get_preds_loss_accuracy(self, batch: Tensor) -> tuple[Tensor, Any, Any]:
        """convenience function since train/valid/test steps are similar.
        Source: https://docs.wandb.ai/guides/integrations/lightning#log-metrics"""
        x, target = batch

        logits = self(x)

        loss = self.loss(logits, target)

        if self.num_classes == 1:
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).int()

            acc = accuracy(logits, target, task="binary")
        else:
            preds = torch.argmax(logits, dim=1)
            # task prediction
            acc = accuracy(
                preds, target, task="multiclass", num_classes=self.num_classes
            )

        return preds, loss, acc


class Template_CBM_MultiClass(pl.LightningModule):
    def __init__(
        self,
        model1: pl.LightningModule,
        model2: pl.LightningModule,
        num_exogenous: int,
        num_classes: int,
        num_concepts: int,
        num_side_channel: int = 0,
        learning_rate: float = 1e-5,
        lambda_weight: float = 0.01,
        frozen_model1: bool = True,
        concept_representation: str = "logits",
        # lambda_decay: Optional[float] = None,
    ):
        super().__init__()

        # assert self._get_model_output_size(model1) >= num_exogenous
        assert num_exogenous >= num_concepts + num_side_channel

        self.x_to_u = model1
        self.frozen_model1 = frozen_model1

        if frozen_model1:
            freeze_model(model1)

        self.u_to_CY = model2

        if num_classes == 1:
            self.task_loss_function = BCEWithLogitsLoss()
        else:
            self.task_loss_function = CrossEntropyLoss()

        self.concept_representation = concept_representation

        if num_classes == 200 and num_concepts == 112:  # CUB
            concept_weights = torch.load("data/CUB/concept_weights_cub.pt")
        else:
            concept_weights = None

        if self.concept_representation in ("logits",):
            # self.concept_loss_function = BCEWithLogitsLoss(
            #     reduction="none"
            # )  # none, so we can calculate individual concept losses
            self.concept_loss_function = BCEWithLogitsLoss(pos_weight=concept_weights)
        elif self.concept_representation in ("hard", "soft"):
            self.concept_loss_function = BCELoss()
        elif self.concept_representation in (
            "group_hard",
            "group_soft",
        ):
            assert self.u_to_CY.mutually_exclusive_concepts is not None
            self.concept_loss_function = BCELoss(weight=concept_weights)
        else:
            raise NotImplementedError

        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.lambda_weight = lambda_weight  # lambda = λ
        self.num_side_channel = num_side_channel

        # USED FOR INTERVENTIONS
        self.interventions = False
        self.intervention_percentile_df = DataFrame()
        self.num_interventions = 1

        self.save_hyperparameters(ignore=["model1", "model2"])
        self.save_hyperparameters(
            {"x2u": type(model1).__name__, "u2c": type(model2).__name__}
        )

        all_model2_hparams = model2.hparams.keys()
        shared_hparams = set(self.hparams.keys()).intersection(all_model2_hparams)

        # Identify the hyperparameters that are unique to model2 (i.e., not shared)
        unique_model2_hparams = {
            key: value
            for key, value in model2.hparams.items()
            if key not in shared_hparams
        }
        # Save the hyperparameters that are not shared (using self.save_hyperparameters)
        self.save_hyperparameters(unique_model2_hparams)

    def _get_model_output_size(self, model: nn.Sequential) -> int:
        # resnet
        if any(isinstance(layer, nn.Sequential) for layer in model.children()):
            return 512
        elif any(isinstance(layer, nn.Conv2d) for layer in model):  # fmnist
            last_layer = list(model.children())[-2]
            return last_layer.out_features
        else:
            print(model)
            raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # calculate concepts
        exogenous_variables = self.x_to_u(x)  # also contains side-channel!

        # calculate task
        y, c = self.u_to_CY(exogenous_variables)

        return y, c

    def training_step(self, batch: Tensor, batch_idx: int) -> Any:
        """Defines a single training step for the model."""

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

        return total_loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> tuple[Tensor, Tensor]:
        """Defines a single validation step for the model."""
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

        return task_preds, concept_preds

    def test_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Defines a single testing step for the model."""
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

        return task_preds, concept_preds

    def predict_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Defines a single testing step for the model."""

        # drop everything from side-channel
        self.u_to_CY.side_channel[-1].p = 1
        self.u_to_CY.side_channel[-1].train()

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

    def on_predict_epoch_end(self):
        """Compute final accuracy after all batches"""
        all_preds = torch.cat(self.all_preds)
        all_labels = torch.cat(self.all_labels)

        if self.num_classes == 1:
            self.dropout_test_acc = accuracy(all_preds, all_labels, task="binary")
        else:
            self.dropout_test_acc = accuracy(
                all_preds, all_labels, task="multiclass", num_classes=self.num_classes
            )

        return self.dropout_test_acc

    def configure_optimizers(self) -> Optimizer:
        """Defines model optimizer. Check also: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#PyTorch-Lightning for a better example"""
        return Adam(self.parameters(), lr=self.learning_rate)

    def _convert_hard_interventions_to_soft(self, true_concepts: Tensor) -> Tensor:
        """Given a csv will convert the binary concepts to the 5th and 95th percentile values.

        Example CSV
        dimension,5th_percentile,95th_percentile
        c_dim_0,-33.069519996643066,18.02792682647705
        c_dim_1,-22.091160202026366,13.25743198394775
        c_dim_2,-39.89247531890869,13.843578529357908
        c_dim_3,-44.440517044067384,11.705989217758159
        c_dim_4,-45.43427505493164,8.42915105819701
        c_dim_5,-20.72880620956421,35.39493045806884
        c_dim_6,-23.334728240966797,19.883290672302206
        c_dim_7,-24.21763219833374,17.654902076721193
        """

        self.intervention_percentiles = self.intervention_percentile_df

        # Mapping the dimensions to the 5th and 95th percentiles
        percentiles_5th = torch.tensor(
            self.intervention_percentiles["5th_percentile"].values, device=self.device
        )
        percentiles_95th = torch.tensor(
            self.intervention_percentiles["95th_percentile"].values, device=self.device
        )

        # Convert TR to the desired values based on the percentiles
        real_valued_interventions = true_concepts * (percentiles_95th) + (
            1 - true_concepts
        ) * (percentiles_5th)

        return real_valued_interventions

    def forward_with_interventions_cbm(
        self, x: Tensor, true_concepts: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        # calculate concepts
        u = self.x_to_u(x)  # also contains side-channel!

        if self.concept_representation not in (
            "hard",
            "group_hard",
        ):
            # convert the true concept values from hard to soft
            true_concepts = self._convert_hard_interventions_to_soft(true_concepts)

        y, c = self.u_to_CY.forward_with_interventions(
            x=u,
            true_concepts=true_concepts,
            num_interventions=self.num_interventions,
        )
        self.log("num_interventions", self.num_interventions)
        return (y, c)

    def _get_preds_loss_accuracy(
        self, batch: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """convenience function since train/valid/test steps are similar.
        Source: https://docs.wandb.ai/guides/integrations/lightning#log-metrics"""
        x, target_concepts, y = batch

        if self.interventions is True:
            task_logits, concept_output = self.forward_with_interventions_cbm(
                x, target_concepts, y
            )

        else:
            task_logits, concept_output = self(x)
        return calculate_mixed_loss(
            task_logits=task_logits,
            concept_logits=concept_output,
            target_concepts=target_concepts,
            y=y,
            concept_loss_function=self.concept_loss_function,
            task_loss_function=self.task_loss_function,
            num_concepts=self.num_concepts,
            num_classes=self.num_classes,
            lambda_weight=self.lambda_weight,
        )

    def configure_gradient_clipping(  # type: ignore
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        """Manual configuration of lightning's gradient clipping for the second model.
        NOTE: tensorboard/wandb log BEFORE clipping https://github.com/Lightning-AI/pytorch-lightning/issues/12595"""

        if self.concept_representation in ("group_hard", "hard"):
            for name, param in self.u_to_CY.named_parameters():
                # print(gradient_clip_val)
                # Clip gradients of 2nd model only
                if gradient_clip_algorithm == "norm":
                    torch.nn.utils.clip_grad_norm_(
                        param, gradient_clip_val
                    )  # Clip by norm
                else:
                    torch.nn.utils.clip_grad_value_(param, gradient_clip_val)


class Standard_resnet18(Template_MultiClass):
    def __init__(
        self,
        num_classes: int = 200,
        learning_rate: float = 1e-3,
        resnet18_path: str = "./pretrained_models/resnet18.pth",
        frozen: bool = False,
        dataset: str = "CUB",
    ):
        super().__init__(learning_rate, num_classes)
        self.dataset = dataset
        # Load the desired ResNet model
        self.resnet = models.resnet18(pretrained=False)

        state_dict = torch.load(resnet18_path)

        if frozen is False:
            print("\n Loading imagenet resnet18 weights")
            self.resnet.load_state_dict(state_dict)

        # Remove the final fully connected layer

        self.concept_extractor = nn.Sequential(
            *list(self.resnet.children())[:-1], nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )

        self.frozen = frozen
        if frozen:
            freeze_model(self.concept_extractor)

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.concept_extractor(x)

        y = self.classifier(x)
        return y

    def configure_optimizers(self) -> Optimizer | tuple[list[Optimizer], list[Any]]:
        """Defines model optimizer. Check also: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#PyTorch-Lightning for a better example"""
        if self.frozen:
            return Adam(self.parameters(), lr=self.learning_rate)
        else:
            assert self.dataset == "CUB"
            print("LR scheduling for CUB")
            opt = Adam(self.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=1 / 2)
            return [opt], [scheduler]


class FashionMNIST_for_CBM(Template_MultiClass):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        frozen: bool = False,
    ) -> None:
        super().__init__(learning_rate, num_classes)

        self.concept_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),  # bottleneck layer
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        if frozen:
            freeze_model(self.concept_extractor)
            backbone_path = "pretrained_models/FMNIST/version_0/checkpoints/epoch=49-step=10750.ckpt"
            state_dict = torch.load(backbone_path)["state_dict"]
            new_state_dict = {
                k.replace("concept_extractor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("concept_extractor.")
            }

            self.concept_extractor.load_state_dict(new_state_dict)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Args:
            x : The input data.

        Returns:
            torch.Tensor: The output of the network.

        """
        x = self.concept_extractor(x)
        y = self.classifier(x)
        return y


class X2C_model(Template_MultiClass):
    def __init__(
        self,
        learning_rate: float,
        num_concepts: int,
        pretrained_model: Optional[Union[pl.LightningModule, nn.Module]] = None,
        classifier_head: Optional[Union[pl.LightningModule, nn.Module]] = None,
        pretrained_frozen: bool = True,
        concept_indexes: Optional[list] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            num_classes=num_concepts,
        )

        if pretrained_model is None:
            raise NotImplementedError
        else:
            concept_extractor = pretrained_model.concept_extractor
            if classifier_head is None:
                last_layer_in_concept_ex = list(concept_extractor.children())[-2]

                size_of_last_layer = last_layer_in_concept_ex.out_features

                classifier = nn.Linear(size_of_last_layer, num_concepts)
            else:
                classifier = nn.Sequential(classifier_head)
            self.model = nn.Sequential(concept_extractor, classifier)

        # freeze concept extractor
        if pretrained_frozen is True:
            freeze_model(self.model[0])

        self.loss = BCEWithLogitsLoss(reduction="none")

        self.concept_indexes = concept_indexes
        if concept_indexes is not None:
            assert len(concept_indexes) == num_concepts

        self.save_hyperparameters(ignore=["pretrained_model"])

    def _get_preds_loss_accuracy(self, batch: Tensor) -> tuple[Tensor, Any, Any]:
        """Changes to make concept floats."""
        x, target_concepts = batch
        if self.concept_indexes is not None:
            target_concepts = target_concepts[:, self.concept_indexes]

        concept_logits = self(x)

        concept_loss = calculate_concept_loss(
            concept_logits, target_concepts, self.loss
        )

        if self.num_classes == 1:
            concept_acc = accuracy(
                concept_logits,
                target_concepts,
                task="binary",
            )
        else:
            concept_acc = accuracy(
                concept_logits,
                target_concepts,
                task="multilabel",
                num_labels=self.num_classes,  # num_classes=num_concepts
            )
        concept_preds = (concept_logits > 0).int()

        return concept_preds, concept_loss, concept_acc


class UtoY_model(Template_MultiClass):
    def __init__(
        self,
        num_exogenous: int,
        num_concepts: int,
        num_side_channel: int,
        num_classes: int,
        learning_rate: float = 0.0,
        lambda_weight: Any = None,
        causal_graph: Optional[BoolTensor] = None,
        masking_algorithm: str = "zuko",
        num_hidden_layers_in_maskedmlp: int = 0,  # how many hidden layers do you want in the maskedmlp,
        previous_model_output_size: Optional[int] = None,
        last_layer_mask: bool = False,
        concept_representation: str = "logit",
        side_dropout: bool = False,
        dropout_prob: float = 0.0,
        mutually_exclusive_concepts: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(learning_rate=learning_rate, num_classes=num_classes)

        if num_side_channel > 0:
            assert (
                num_exogenous >= num_concepts + num_side_channel
            )  # at least 1 exogenous per concept/and y

            # assert num_side_channel % num_classes == 0
            if masking_algorithm == "zuko":
                assert (num_exogenous - num_side_channel) % num_concepts == 0

        # convert the causal graph into adjacency matrix when you have more exogenous than 1 per concept
        input_per_concept = (num_exogenous - num_side_channel) // num_concepts
        # exogenous_per_class = num_side_channel // num_classes
        self.previous_model_output_size = previous_model_output_size
        self.side_dropout = side_dropout
        self.dropout_prob = dropout_prob
        if side_dropout is True:
            assert dropout_prob != 0
        self.lambda_weight = lambda_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_concepts = num_concepts
        self.num_side_channel = num_side_channel
        self.num_exogenous = num_exogenous
        self.causal_graph = causal_graph
        self.masking_algorithm = masking_algorithm
        self.ctoy_depth = kwargs.get("ctoy_depth", None)
        self.concept_representation = concept_representation
        self.mutually_exclusive_concepts = mutually_exclusive_concepts
        # print("mutex concepts", self.mutually_exclusive_concepts)
        # print("len groups", len(self.mutually_exclusive_concepts))
        self.group_interventions = False

        self._check_mutually_exclusive()
        self.non_mutually_exclusive_concepts = self._find_non_mutually_exclusive()
        # print("len groups", len(self.non_mutually_exclusive_concepts))

        print(self._split_into_in_direct_concepts())

        self.init_concept_concept(
            masking_algorithm, num_hidden_layers_in_maskedmlp, input_per_concept
        )

        self.init_side_channel()

        self.init_representation_splitter()

        self.init_concept_task(last_layer_mask)

        self.save_hyperparameters()

    def init_concept_concept(
        self, masking_algorithm, num_hidden_layers_in_maskedmlp, input_per_concept
    ):
        if self.causal_graph is None or masking_algorithm == "none":
            print("No masking algorithm given! Just a simple MLP is used now.")
            self.u2c_model = nn.Linear(
                self.num_exogenous - self.num_side_channel, self.num_concepts
            )

        else:
            # check if adj matrix is square
            assert self.causal_graph.ndim == 2  # type: ignore

            self.u2c_graph = self.causal_graph[: -self.num_classes, : -self.num_classes]  # type: ignore
            self.c2y_graph = self.causal_graph[-self.num_classes :, :]  # type: ignore

            # handles exogenous per concepts

            multidim_concept_graph = self._replicate_columns(
                self.u2c_graph,
                input_per_concept,  ### Kronecker product
            )

            # handles exogenous_per_class
            # self.multidim_task_graph = self._replicate_columns(
            #     self.c2y_graph, exogenous_per_class, num_concepts
            # )
            if self.num_side_channel == 0:
                self.multidim_task_graph = self._replicate_columns(
                    self.c2y_graph, 0, self.num_concepts
                )
            else:
                self.multidim_task_graph = self._replicate_columns(
                    self.c2y_graph, 1, self.num_concepts
                )

            # assert self.multidim_task_graph.shape[1] == num_side_channel + num_concepts
            assert self.multidim_task_graph.shape[0] == self.num_classes
            assert (
                multidim_concept_graph.shape[0] <= multidim_concept_graph.shape[1]
            )  # check that input is bigger than concepts

            # check adjmatrix contains exactly as many outputs as concepts
            assert (
                multidim_concept_graph.shape[1]
                == self.num_exogenous - self.num_side_channel
            )

            if masking_algorithm == "zuko":
                # if num_hidden_layers_in_maskedmlp >= 0:
                # if depth==0 then this is maskedLinear, WITHOUT activation. Activation is handled in forward function!
                self.u2c_model = MaskedMLP(
                    multidim_concept_graph,
                    hidden_features=[
                        multidim_concept_graph.shape[0]
                        for _ in range(num_hidden_layers_in_maskedmlp)
                    ],
                )

            elif masking_algorithm == "leakage_experiment":
                # if depth==0 then this is maskedLinear, WITHOUT activation. Activation is handled in forward function!
                self.u2c_model = nn.Sequential(
                    nn.Linear(
                        self.num_exogenous - self.num_side_channel, self.num_concepts
                    ),
                )

            else:  # Other masking methods can be added here.
                raise NotImplementedError

    def init_concept_task(self, last_layer_mask):
        if last_layer_mask is True:
            if self.ctoy_depth is None:
                self.last_layer = MaskedLinear(adjacency=self.multidim_task_graph)
            else:  # CREAM without linear classifier
                self.last_layer = MaskedMLP(
                    adjacency=self.multidim_task_graph,
                    hidden_features=[
                        self.multidim_task_graph.shape[1]
                        for _ in range(self.ctoy_depth)
                    ],
                )

        else:
            # CBM
            if self.num_side_channel == 0 and self.side_dropout is False:
                self.last_layer = nn.Sequential(
                    nn.Linear(self.num_concepts, self.num_classes),
                )
            elif (
                self.num_side_channel == 0 and self.side_dropout is True
            ):  # CBM with side channel
                self.last_layer = nn.Sequential(
                    nn.Linear(self.num_concepts + self.num_classes, self.num_classes),
                )

    def init_representation_splitter(self) -> None:
        if self.previous_model_output_size is not None:
            self.u2u_model = nn.Sequential(
                nn.Linear(self.previous_model_output_size, self.num_exogenous),
                nn.ReLU(),
            )
        else:
            print("No linear layer previous")
            self.u2u_model = nn.Sequential(nn.Identity())

    def init_side_channel(self) -> None:
        # CBM with side channel
        if self.masking_algorithm == "none" and self.side_dropout:
            assert self.previous_model_output_size is None
            self.side_channel = nn.Sequential(
                nn.Identity(self.num_exogenous, self.num_exogenous),
                nn.Linear(self.num_exogenous, self.num_classes),
                nn.ReLU(),
            )  # side channel
        # CREAM with side channel
        else:
            self.side_channel = nn.Sequential(
                nn.Identity(self.num_side_channel, self.num_side_channel),
                nn.Linear(self.num_side_channel, self.num_classes),
                nn.ReLU(),
            )  # side channel

        if self.side_dropout is True:
            self.side_channel = nn.Sequential(
                *self.side_channel,
                StochasticDepth(p=self.dropout_prob, mode="batch"),
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        u = self.u2u_model(x)

        # [Uc;Uy]
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel :]
        concept_before_activation = self.u2c_model(Uc)

        c = self.concept_activation_function(concept_before_activation)

        if self.side_dropout is True and self.masking_algorithm == "none":
            s = self.side_channel(Uc)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)

        elif self.num_side_channel > 0:
            s = self.side_channel(Uy)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)

        else:
            y = self.last_layer(c)
        return y, c

    def concept_activation_function(self, c: Tensor) -> Tensor:
        """Changes the value of the concept (logit) to the respective representation."""

        # Handle the activations for each concept
        if self.concept_representation == "logits":
            pass
        elif self.concept_representation == "hard":
            c = torch.sigmoid(c)
            c = self._straight_through(c)
        elif (
            self.concept_representation == "group_hard"
            and self.mutually_exclusive_concepts is not None
        ):
            c = self._apply_group_softmax(c)
            c = self._apply_sigmoid_to_remaining(c)
            c = self._straight_through(c)
        elif self.concept_representation == "soft":  # soft values in [0,1]
            c = self._apply_sigmoid_to_remaining(c)
        elif (
            self.concept_representation == "group_soft"
            and self.mutually_exclusive_concepts is not None
        ):
            c = self._apply_group_softmax(c)
            # Make sure that the remaining have similar value range
            c = self._apply_sigmoid_to_remaining(c)
        else:
            raise NotImplementedError

        return c

    def forward_with_interventions(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Perform forward pass using interventions
        Args:
            x (Tensor): Output of model 1
            true_concepts (Tensor): The true values of the concepts
            intervention_mask (Tensor): A mask that selects which true values will be copied on the predicted concepts.
        """
        u = self.u2u_model(x)

        # [Uc;Uy]
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel :]

        c = self.u2c_model(Uc)
        c = self.concept_activation_function(c)
        c_predicted = c.clone()
        ### Perform intervention
        if intervention_mask is None:  # if its not given by another method
            if self.group_interventions:
                intervention_mask = self.generate_group_intervention_mask(
                    num_group_interventions=num_interventions,
                    batch_size=c.size(0),
                )
            else:
                intervention_mask = self.generate_intervention_mask(
                    num_interventions=num_interventions,
                    batch_size=c.size(0),
                )

        c_predicted[intervention_mask] = (true_concepts[intervention_mask]).type(
            c_predicted.dtype
        )

        c = c_predicted  # CHANGE C TO THE INTERVENED VALUES

        # CBM + side channel
        if self.side_dropout is True and self.masking_algorithm == "none":
            s = self.side_channel(Uc)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)
        # CREAM + side channel
        elif self.num_side_channel > 0:
            s = self.side_channel(Uy)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)

        else:
            y = self.last_layer(c)
        return y, c

    def _apply_group_softmax(self, c: Tensor) -> Tensor:
        """Applies softmax to mutually exclusive concepts."""
        import torch.nn.functional as F

        # Clone so to handle concepts not in part of any softmax groups
        temp_c = c.clone()
        for group in self.mutually_exclusive_concepts:  # type: ignore
            group_outputs = c[:, group]  # Select specific indices for the group

            softmaxed_group = F.softmax(
                group_outputs, dim=1
            )  # Apply softmax to the group
            temp_c[:, group] = softmaxed_group  # Select specific indices for the group

        return temp_c

    def _apply_sigmoid_to_remaining(self, c: Tensor) -> Tensor:
        """Applies sigmoid to indices not part of any softmax group or all logits if no groups are defined."""
        # If no groups are defined / doing just hard concepts
        if not self.mutually_exclusive_concepts and self.concept_representation not in (
            "group_hard",
            "group_soft",
        ):
            # print(
            #     "No mutually exclusive groups defined. Applying sigmoid to all indices."
            # )

            return torch.sigmoid(c)

        remaining_indices = self.non_mutually_exclusive_concepts
        if remaining_indices:  # Check if there are indices outside groups
            # print(f"Applying sigmoid to the following indices: {remaining_indices}")
            c[:, remaining_indices] = torch.sigmoid(c[:, remaining_indices])

        return c

    def c2y_forward(self, c: Tensor) -> Tensor:
        y = self.last_layer(c)
        return y

    def _get_preds_loss_accuracy(self, batch: Tensor) -> tuple[Tensor, ...]:
        """convenience function since train/valid/test steps are similar.
        Source: https://docs.wandb.ai/guides/integrations/lightning#log-metrics"""
        if not hasattr(self, "task_loss_function"):
            if self.num_classes == 1:
                self.task_loss_function = BCEWithLogitsLoss()
            else:
                self.task_loss_function = CrossEntropyLoss()

        c, y = batch
        task_logits = self.c2y_forward(c.float())  # used when training U/C->Y
        # task prediction
        task_loss = self.task_loss_function(task_logits, y)
        if self.num_classes == 1:
            task_probs = torch.sigmoid(task_logits)

            task_preds = (task_probs > 0.5).int()

            task_acc = accuracy(task_logits, y, task="binary")
        else:
            # task prediction
            task_preds = torch.argmax(task_logits, dim=1)
            task_acc = accuracy(
                task_preds, y, task="multiclass", num_classes=self.num_classes
            )

        return (
            task_preds,
            task_loss,
            task_acc,
        )

    def _replicate_columns(
        self,
        bool_tensor: torch.Tensor,
        num_replicates: int,
        starting_column: int = 0,
    ) -> torch.Tensor:
        """Given a DAG that represents a SCM, change the adjacency matrix by replicating columns
        starting from a given index, so that selected nodes take n-dimensional input instead of 1-dim.
        i.e. Kronecker product

        Args:
            bool_tensor (torch.Tensor): adjacency matrix.
            num_replicates (int): number of times to replicate each column.
            starting_column (int): the index from which to start replicating columns.
        """
        # Calculate the number of columns before and after `starting_column`
        initial_columns = bool_tensor.size(1) - starting_column
        replicated_columns = initial_columns * num_replicates + starting_column

        # Create the expanded tensor with space for original and replicated columns
        expanded_tensor = torch.empty(
            bool_tensor.size(0), replicated_columns, dtype=torch.bool
        )

        # Copy columns up to `starting_column` without replication
        expanded_tensor[:, :starting_column] = bool_tensor[:, :starting_column]

        # Replicate columns starting from `starting_column`
        for i in range(starting_column, bool_tensor.size(1)):
            start_idx = starting_column + (i - starting_column) * num_replicates
            end_idx = start_idx + num_replicates
            expanded_tensor[:, start_idx:end_idx] = (
                bool_tensor[:, i].unsqueeze(1).expand(-1, num_replicates)
            )

        return expanded_tensor

    def generate_intervention_mask(
        self, num_interventions: int, batch_size: int
    ) -> Tensor:
        """Generates random masks for interventions. Always samples first from the direct concepts.
        Will sample from indirect even num_intervention is high enough."""
        indirect_concept_indices, direct_concept_indices = (
            self._split_into_in_direct_concepts()
        )

        all_intervention_masks = []

        if num_interventions > self.num_concepts:
            raise ValueError("Num of interventions exceeds number of concepts")
        if num_interventions > len(direct_concept_indices):
            print("Also intervening on non-direct concepts!")

        for _ in range(batch_size):
            intervention_mask = torch.zeros(self.num_concepts, dtype=torch.bool)
            selected_indices = random.sample(
                direct_concept_indices,
                min(num_interventions, len(direct_concept_indices)),
            )

            intervention_mask[selected_indices] = 1

            if num_interventions > len(direct_concept_indices):
                selected_indices = random.sample(
                    indirect_concept_indices,
                    num_interventions - len(direct_concept_indices),
                )
                intervention_mask[selected_indices] = 1

            all_intervention_masks.append(intervention_mask)

        return torch.stack(all_intervention_masks)

    def generate_group_intervention_mask(
        self, num_group_interventions: int, batch_size: int
    ) -> Tensor:
        """
        Generates random masks for interventions at the group level, prioritizing direct groups/indices.
        If num_group_interventions exceeds the number of direct groups/indices, additional groups/indices are
        chosen from indirect groups/indices.

        Args:
            num_group_interventions (int): Number of groups/indices to intervene on.
            batch_size (int): Number of intervention masks to generate.

        Returns:
            Tensor: A tensor of shape (batch_size, num_concepts) where each row is an intervention mask.
        """
        # Step 1: Categorize groups and ungrouped indices

        indirect_concept_indices, all_concept_indices = (
            self._split_into_in_direct_concepts_GROUP_INTERVENTIONS()
        )

        direct_groups, indirect_groups, ungrouped_direct, ungrouped_indirect = (
            self.find_direct_and_indirect_groups(
                set(all_concept_indices), set(indirect_concept_indices)
            )
        )

        # Step 2: Flatten ungrouped indices into pseudo-groups
        ungrouped_direct = [[index] for index in ungrouped_direct]
        ungrouped_indirect = [[index] for index in ungrouped_indirect]

        # Step 3: Combine grouped and ungrouped data
        all_direct = direct_groups + ungrouped_direct
        all_indirect = indirect_groups + ungrouped_indirect

        if num_group_interventions > len(all_direct) + len(all_indirect):
            raise ValueError(
                f"Number of groups exceeds total available groups/indices: {len(all_direct) + len(all_indirect)}"
            )
        if num_group_interventions > len(all_direct):
            print("Also intervening on indirect groups/indices!")

        all_intervention_masks = []

        for _ in range(batch_size):
            # Initialize an intervention mask with all zeros
            intervention_mask = torch.zeros(self.num_concepts, dtype=torch.bool)

            selected_groups = []
            if num_group_interventions <= len(all_direct):
                # Only select from direct groups/indices
                selected_groups = random.sample(all_direct, num_group_interventions)
            else:
                # Select all direct groups/indices first
                selected_groups = all_direct.copy()
                # Add remaining groups/indices from indirect pool
                num_remaining = num_group_interventions - len(all_direct)
                selected_groups += random.sample(all_indirect, num_remaining)

            # Set the indices of the selected groups/indices to 1 in the mask
            for group in selected_groups:
                intervention_mask[group] = 1

            all_intervention_masks.append(intervention_mask)

        return torch.stack(all_intervention_masks)

    def _check_mutually_exclusive(self) -> None:
        """Method to check if the given softmax mask corresponds to mutually exclusive concepts."""
        # Flatten the list
        if self.mutually_exclusive_concepts is not None:
            all_indices = [
                index for group in self.mutually_exclusive_concepts for index in group
            ]

            # Convert to a set and check for non-unique values
            assert len(all_indices) == len(set(all_indices))
            # The concepts that are mutually exclusive should be less or equal to num of concepts
            assert len(all_indices) <= self.num_concepts

        if self.concept_representation in ("group_hard", "group_soft"):
            assert self.mutually_exclusive_concepts is not None
        else:
            assert self.mutually_exclusive_concepts is None

    def _straight_through(self, probabilities: Tensor) -> Tensor:
        """
        Applies straight-through estimation for both softmax (mutually exclusive groups)
        and sigmoid (binary independent variables).
        Assumes:
        1. Mutually exclusive groups have been processed with softmax.
        2. Remaining indices have been processed with sigmoid.

        Parameters:
            probabilities (Tensor): Input tensor after applying group softmax and remaining sigmoid.

        Returns:
            Tensor: Tensor with straight-through applied.
        """

        output = probabilities.clone()

        # Apply straight-through for mutually exclusive groups
        if self.mutually_exclusive_concepts:
            for group in self.mutually_exclusive_concepts:
                # Extract probabilities for the group
                group_probs = probabilities[:, group]

                group_hard_preds = torch.zeros_like(group_probs).scatter_(
                    1, torch.argmax(group_probs, dim=-1, keepdim=True), 1.0
                )
                # Straight-through: forward hard, backward soft
                output[:, group] = group_hard_preds - group_probs.detach() + group_probs

        # Apply straight-through for binary independent variables (remaining indices)
        if not self.mutually_exclusive_concepts:
            # No groups defined: treat all indices as binary independent
            # print(
            #     "No mutually exclusive groups defined. Applying binary straight-through to all indices."
            # )
            hard_preds = (probabilities > 0.5).float()
            output = hard_preds - probabilities.detach() + probabilities
        else:
            remaining_indices = self.non_mutually_exclusive_concepts

            if remaining_indices:
                # print(
                #     f"Applying binary straight-through to the following indices: {remaining_indices}"
                # )
                remaining_probs = probabilities[:, remaining_indices]
                hard_preds = (remaining_probs > 0.5).float()
                output[:, remaining_indices] = (
                    hard_preds - remaining_probs.detach() + remaining_probs
                )

        return output

    def _find_non_mutually_exclusive(self) -> list[int]:
        all_indices = set(range(self.num_concepts))
        ## If no mutex concepts are given, then return all concepts
        if self.mutually_exclusive_concepts is None:
            return list(all_indices)
        all_group_indices = set(
            idx for group in self.mutually_exclusive_concepts for idx in group
        )
        remaining_indices = list(all_indices - all_group_indices)
        print("Non mutually exclusive concept indexes:", remaining_indices)
        return remaining_indices

    def _split_into_in_direct_concepts_GROUP_INTERVENTIONS(
        self,
    ) -> tuple[list[int], list[int]]:
        if self.causal_graph is None:
            print("\n\nno causal graph given. Assuming CBM")
            return list(range(self.num_concepts)), []
        else:
            classes = self.causal_graph[self.num_concepts :]
            # Check for columns without a value of 1 (or True)
            columns_without_ones = ~classes.any(dim=0)

            # Get indices of such columns
            indirect_concept_indices = torch.where(columns_without_ones)[0]

            # Check for columns that have a value of 1 (or True) in the specified rows
            all_indices = set(range(self.num_concepts))
            # Given indices as a set
            # given_indices_set = set(indirect_concept_indices)
            # Remaining indices
            # all_concept_indices = list(all_indices - given_indices_set)
            all_concept_indices = list(all_indices)

            return indirect_concept_indices.tolist(), all_concept_indices

    def _split_into_in_direct_concepts(self) -> tuple[list[int], list[int]]:
        if self.causal_graph is None:
            print("\n\nno causal graph given. Assuming CBM")
            return list(range(self.num_concepts)), []
        else:
            classes = self.causal_graph[self.num_concepts :]

            # Check for columns without a value of 1 (or True)
            columns_without_ones = ~classes.any(dim=0)

            # Get indices of such columns
            indirect_concept_indices = torch.where(columns_without_ones)[0]

            # Check for columns that have a value of 1 (or True) in the specified rows
            all_indices = set(range(self.num_concepts))
            # Given indices as a set
            given_indices_set = set(indirect_concept_indices.tolist())
            # Remaining indices
            direct_concept_indices = list(all_indices - given_indices_set)

            return indirect_concept_indices.tolist(), direct_concept_indices

    def find_direct_and_indirect_groups(
        self, direct_concept_indices: set, indirect_concept_indices: set
    ) -> tuple:
        """
        Categorizes groups in self.mutex into direct and indirect groups,
        while also categorizing ungrouped indices.

        Args:
            direct_concept_indices (set): Set of direct concept indices.
            indirect_concept_indices (set): Set of indirect concept indices.

        Returns:
            tuple: A tuple (direct_groups, indirect_groups, ungrouped_direct, ungrouped_indirect), where:
                - direct_groups is a list of groups where all indices are direct.
                - indirect_groups is a list of groups where any index is indirect.
                - ungrouped_direct is a list of direct indices not part of any group.
                - ungrouped_indirect is a list of indirect indices not part of any group.
        """
        direct_groups = []
        indirect_groups = []

        assert self.mutually_exclusive_concepts is not None

        # Flatten self.mutex to find all grouped indices
        grouped_indices = set(
            index for group in self.mutually_exclusive_concepts for index in group
        )

        # Categorize each group in self.mutex
        for group in self.mutually_exclusive_concepts:
            group_set = set(group)

            # Check if the group is indirect or direct
            if group_set & indirect_concept_indices:
                # If any index in the group is indirect, the group is indirect
                indirect_groups.append(group)
            elif group_set <= direct_concept_indices:
                # If all indices are direct, the group is direct
                direct_groups.append(group)

        # Determine ungrouped\non mutex indices
        all_indices = set(range(self.num_concepts))
        ungrouped_indices = all_indices - grouped_indices

        # Categorize ungrouped indices
        ungrouped_direct = list(ungrouped_indices & direct_concept_indices)
        ungrouped_indirect = list(ungrouped_indices & indirect_concept_indices)

        return direct_groups, indirect_groups, ungrouped_direct, ungrouped_indirect


class C2Y_model(Template_MultiClass):
    def __init__(
        self,
        learning_rate: float,
        num_classes: int,
        num_concepts: int,
        linear_classifier: bool = True,
    ):
        super().__init__(learning_rate, num_classes)
        if linear_classifier:
            self.model = nn.Linear(num_concepts, num_classes)
        else:
            self.model = nn.Sequential(
                nn.Linear(num_concepts, 2 * num_concepts),
                nn.ReLU(),
                nn.Linear(2 * num_concepts, 2 * num_concepts),
                nn.ReLU(),
                nn.Linear(2 * num_concepts, num_classes),
            )
