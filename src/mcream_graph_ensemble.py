"""
mCREAM Graph Module Ensemble.

Subclasses UtoY_model directly — copies ALL of CREAM's code.
Only two things change:
  1. __init__:  builds u2c_models (list of M MaskedMLPs) instead of one u2c_model
  2. forward(): averages M concept predictions instead of using one

Everything else — u2u_model, side_channel, last_layer, concept_activation_function,
_get_preds_loss_accuracy, interventions, _replicate_columns — is CREAM code, unchanged.

Loss:
    task_loss    = CrossEntropy(y, y_true)                  same as CREAM
    concept_loss = sum( BCE(c_m, true_concepts) for m )     M terms, full gradient each
    total        = task_loss + lambda * concept_loss         same formula as CREAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor
from torchmetrics.functional import accuracy
from typing import Any, Optional, List

import pytorch_lightning as pl

from .models import UtoY_model, Template_CBM_MultiClass, freeze_model
from zuko.nn import MaskedMLP


class UtoY_MultiGraph(UtoY_model):
    """
    CREAM's UtoY_model with M Concept-Concept blocks instead of one.

    Only __init__ (building u2c_models), forward(), and
    _get_preds_loss_accuracy() are changed.
    All other methods are inherited from UtoY_model unchanged.
    """

    def __init__(
        self,
        expert_graphs: List[BoolTensor],   # M full (K+T)x(K+T) graphs
        ref_graph: BoolTensor,             # for last_layer (GT or union graph)
        # --- identical params as UtoY_model ---
        num_exogenous: int = 128,
        num_concepts: int = 11,
        num_side_channel: int = 40,
        num_classes: int = 10,
        learning_rate: float = 0.001,
        lambda_weight: float = 1.0,
        masking_algorithm: str = "zuko",
        num_hidden_layers_in_maskedmlp: int = 0,
        previous_model_output_size: Optional[int] = 128,
        last_layer_mask: bool = True,
        concept_representation: str = "group_soft",
        side_dropout: bool = True,
        dropout_prob: float = 0.9,
        mutually_exclusive_concepts: Optional[list] = None,
        **kwargs: Any,
    ) -> None:

        # Store expert graphs before calling parent (parent calls init_concept_concept)
        self._expert_graphs = expert_graphs
        self.num_experts    = len(expert_graphs)

        # Call parent with ref_graph — builds u2u_model, side_channel, last_layer
        # init_concept_concept() also runs here and creates self.u2c_model
        super().__init__(
            num_exogenous=num_exogenous,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel,
            num_classes=num_classes,
            learning_rate=learning_rate,
            lambda_weight=lambda_weight,
            causal_graph=ref_graph,
            masking_algorithm=masking_algorithm,
            num_hidden_layers_in_maskedmlp=num_hidden_layers_in_maskedmlp,
            previous_model_output_size=previous_model_output_size,
            last_layer_mask=last_layer_mask,
            concept_representation=concept_representation,
            side_dropout=side_dropout,
            dropout_prob=dropout_prob,
            mutually_exclusive_concepts=mutually_exclusive_concepts,
            **kwargs,
        )

        # Parent created self.u2c_model from ref_graph — replace with M experts
        del self.u2c_model
        input_per_concept = (num_exogenous - num_side_channel) // num_concepts

        self.u2c_models = nn.ModuleList([
            self._build_u2c_from_graph(g, input_per_concept, num_hidden_layers_in_maskedmlp)
            for g in expert_graphs
        ])

        # Learnable per-expert concept loss weights λ_m (Kavya: multi-task learning style)
        # Initialised to 1/M so total concept weight matches lambda_weight at start.
        # Trained end-to-end: experts whose concepts are harder to learn get higher λ.
        # Softmax ensures weights sum to 1 and stay positive.
        self._lambda_logits = nn.Parameter(
            torch.zeros(self.num_experts)   # softmax → uniform 1/M initially
        )

    def _build_u2c_from_graph(
        self,
        full_graph: BoolTensor,
        input_per_concept: int,
        num_hidden: int,
    ) -> nn.Module:
        """
        Exact copy of what init_concept_concept() does for one graph.
        Uses parent's _replicate_columns (already inherited).
        """
        u2c_graph = full_graph[:-self.num_classes, :-self.num_classes]

        multidim_concept_graph = self._replicate_columns(
            u2c_graph, input_per_concept
        )

        return MaskedMLP(
            multidim_concept_graph,
            hidden_features=[
                multidim_concept_graph.shape[0]
                for _ in range(num_hidden)
            ],
        )

    # ── Forward — only the u2c call changes, rest is CREAM ───────────────────

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        u = self.u2u_model(x)

        # Same split as CREAM
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]

        # ── CHANGED: run M concept blocks, average results ────────────────────
        all_c = []
        for u2c_m in self.u2c_models:
            c_m = self.concept_activation_function(u2c_m(Uc))
            all_c.append(c_m)
        c = torch.stack(all_c, dim=0).mean(dim=0)   # [B, K]
        # ─────────────────────────────────────────────────────────────────────

        # Below is identical to CREAM's forward
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

    @property
    def expert_weights(self):
        """Return learned λ_m weights for each expert (after softmax)."""
        return torch.softmax(self._lambda_logits, dim=0).detach()

    # ── Interventions — inherited logic, but uses averaged c ─────────────────

    def forward_with_interventions(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask=None,
    ) -> tuple[Tensor, Tensor]:
        """
        Same as CREAM's forward_with_interventions but uses c_avg from M experts.
        After computing c_avg, we replace intervened positions with true values —
        exactly what CREAM does with its single c.
        """
        u  = self.u2u_model(x)
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]

        # Compute c_avg from M experts (same as forward)
        all_c = []
        for u2c_m in self.u2c_models:
            c_m = self.concept_activation_function(u2c_m(Uc))
            all_c.append(c_m)
        c = torch.stack(all_c, dim=0).mean(dim=0)   # [B, K]
        c_predicted = c.clone()

        # Generate intervention mask (inherited from UtoY_model)
        if intervention_mask is None:
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

        # Replace predicted concepts with true values at intervened positions
        c_predicted[intervention_mask] = true_concepts[intervention_mask].type(
            c_predicted.dtype
        )
        c = c_predicted

        # Shared side channel + task head — identical to CREAM
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

    # ── Loss — adds per-expert concept supervision ────────────────────────────

    def _get_preds_loss_accuracy(self, batch: Tensor):
        """
        Same as CREAM's _get_preds_loss_accuracy except concept_loss
        is the SUM over M per-expert losses instead of one averaged loss.

        This gives each u2c_models[m] the full concept gradient (1.0)
        instead of CREAM's 1.0 / num_experts dilution.
        """
        x, target_concepts, y_true = batch

        y, c_avg = self(x)   # uses our new forward

        # ── Per-expert concept loss ───────────────────────────────────────────
        u  = self.u2u_model(x)
        Uc = u[:, : self.num_exogenous - self.num_side_channel]

        # Learnable per-expert weights λ_m (Kavya: multi-task learning style)
        # softmax → sum to 1, always positive
        lambdas = torch.softmax(self._lambda_logits, dim=0)   # [M]

        all_c_m = []
        per_expert_concept_loss = torch.tensor(0.0, device=x.device)
        for m_idx, u2c_m in enumerate(self.u2c_models):
            c_m = self.concept_activation_function(u2c_m(Uc))
            all_c_m.append(c_m)
            # λ_m · BCE(c_m, c_true)  — weighted per-expert supervision
            bce_m = F.binary_cross_entropy(
                c_m.clamp(1e-7, 1 - 1e-7), target_concepts.float()
            )
            per_expert_concept_loss = per_expert_concept_loss + lambdas[m_idx] * bce_m

        # Also supervise c_avg (the vector that actually enters last_layer)
        c_avg_supervised = torch.stack(all_c_m, dim=0).mean(dim=0)
        ensemble_concept_loss = F.binary_cross_entropy(
            c_avg_supervised.clamp(1e-7, 1 - 1e-7), target_concepts.float()
        )
        # Final concept loss = weighted individual + ensemble
        # concept_loss = Σ_m λ_m·BCE(c_m) + BCE(c_avg)
        per_expert_concept_loss = per_expert_concept_loss + ensemble_concept_loss

        # ── Task loss ─────────────────────────────────────────────────────────
        if self.num_classes == 1:
            # squeeze both to [B] to avoid shape mismatch on CelebA
            task_loss  = self.task_loss_function(y.squeeze(-1), y_true.float().squeeze(-1))
            task_preds = (torch.sigmoid(y) > 0.5).int().squeeze(-1)
            task_acc   = accuracy(y.squeeze(-1), y_true.squeeze(-1).int(), task="binary")
        else:
            task_loss  = self.task_loss_function(y, y_true)
            task_preds = y.argmax(dim=1)
            task_acc   = accuracy(task_preds, y_true.view(-1), task="multiclass",
                                  num_classes=self.num_classes)

        concept_acc   = accuracy(c_avg, target_concepts, task="multilabel",
                                 num_labels=self.num_concepts)
        concept_preds = (c_avg > 0.5).int()

        total_loss        = task_loss + self.lambda_weight * per_expert_concept_loss
        task_loss_percent = task_loss / total_loss * 100

        return (
            task_preds,
            concept_preds,
            per_expert_concept_loss,
            task_loss,
            concept_acc,
            task_acc,
            total_loss,
            task_loss_percent,
        )


# =============================================================================
# Full model: backbone + UtoY_MultiGraph
# Mirrors Template_CBM_MultiClass exactly — same interface
# =============================================================================

class mCREAM_GraphEnsemble(Template_CBM_MultiClass):
    """
    Full mCREAM Graph Ensemble = frozen backbone + UtoY_MultiGraph.
    Drop-in replacement for Template_CBM_MultiClass(backbone, UtoY_model).
    All training, validation, test, intervention code inherited unchanged.
    """

    def __init__(
        self,
        backbone: pl.LightningModule,
        expert_graphs: List[BoolTensor],
        ref_graph: BoolTensor,
        num_exogenous: int = 128,
        num_concepts: int = 11,
        num_side_channel: int = 40,
        num_classes: int = 10,
        learning_rate: float = 0.001,
        lambda_weight: float = 1.0,
        previous_model_output_size: Optional[int] = 128,
        concept_representation: str = "group_soft",
        side_dropout: bool = True,
        dropout_prob: float = 0.9,
        num_hidden_layers_in_maskedmlp: int = 0,
        mutually_exclusive_concepts: Optional[list] = None,
        frozen_backbone: bool = True,
    ):
        if frozen_backbone:
            freeze_model(backbone)

        u_to_CY = UtoY_MultiGraph(
            expert_graphs=expert_graphs,
            ref_graph=ref_graph,
            num_exogenous=num_exogenous,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel,
            num_classes=num_classes,
            learning_rate=learning_rate,
            lambda_weight=lambda_weight,
            previous_model_output_size=previous_model_output_size,
            concept_representation=concept_representation,
            side_dropout=side_dropout,
            dropout_prob=dropout_prob,
            num_hidden_layers_in_maskedmlp=num_hidden_layers_in_maskedmlp,
            mutually_exclusive_concepts=mutually_exclusive_concepts,
        )

        super().__init__(
            model1=backbone,
            model2=u_to_CY,
            num_exogenous=num_exogenous,
            num_classes=num_classes,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel,
            learning_rate=learning_rate,
        )

    def forward(self, x: Tensor) -> tuple:
        """
        Uses backbone.concept_extractor(x) directly to get [B, 128] features.
        FashionMNIST_for_CBM.forward() goes through the classifier and returns
        [B, 10] logits which would crash u2u_model = Linear(128, 128).
        Using concept_extractor directly matches previous_model_output_size=128.
        """
        exogenous_variables = self.x_to_u.concept_extractor(x)  # [B, 128]
        y, c = self.u_to_CY(exogenous_variables)
        return y, c

