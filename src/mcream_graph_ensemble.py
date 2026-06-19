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
        # Random init breaks symmetry so gradients can differentiate experts.
        # Softmax ensures weights sum to 1 and stay positive.
        # Using small random noise: experts start near-uniform but can diverge.
        self._lambda_logits = nn.Parameter(
            torch.randn(self.num_experts) * 0.1   # small random, softmax → near-uniform
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
        # Aggregate at LOGIT level, then activate once.
        # This produces sharper outputs matching CREAM's near-binary c.
        #
        # Averaging probabilities: mean([0.9, 0.1]) = 0.5  (soft, loses signal)
        # Averaging logits:        mean([+3, -3])   = 0.0  → sigmoid = 0.5
        #                          BUT mean([+5, -1]) = +2.0 → sigmoid = 0.88 (sharper)
        #
        # For group_soft: average the MaskedMLP outputs (logits) then apply
        # group_softmax once → same confident distribution as CREAM.
        all_c_logits = []
        for u2c_m in self.u2c_models:
            all_c_logits.append(u2c_m(Uc))   # raw logits [B, K]
        c_logits_stacked = torch.stack(all_c_logits, dim=0)           # [M, B, K]
        c_logits_agg = c_logits_stacked.max(dim=0).values             # [B, K] — most confident expert per concept
        c = self.concept_activation_function(c_logits_agg)            # activate once
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

    # Set to True externally to enable intervention debug logging
    _debug_interventions: bool = False
    _debug_log_path: str = "/home/dani00003/mCREAM/logs/intervention_debug_graph_ensemble.txt"

    def forward_with_interventions(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask=None,
    ) -> tuple[Tensor, Tensor]:
        """
        Same as CREAM's forward_with_interventions but uses c_avg from M experts.
        """
        u  = self.u2u_model(x)
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]

        # Aggregate at logit level then activate once — same as forward()
        all_c_logits = []
        for u2c_m in self.u2c_models:
            all_c_logits.append(u2c_m(Uc))
        c_logits_stacked = torch.stack(all_c_logits, dim=0)           # [M, B, K]
        c_logits_agg = c_logits_stacked.max(dim=0).values             # [B, K]
        c = self.concept_activation_function(c_logits_agg)
        c_predicted = c.clone()

        # Generate intervention mask
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

        # ── DEBUG LOGGING ─────────────────────────────────────────────────────
        if getattr(self, '_debug_interventions', False) and num_interventions > 0:
            import os
            log_path = getattr(self, '_debug_log_path',
                               '/home/dani00003/mCREAM/logs/intervention_debug_graph_ensemble.txt')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                s0 = c_predicted[0].detach().cpu().numpy()
                tc0 = true_concepts[0].detach().cpu().float().numpy()
                mask0 = intervention_mask[0].detach().cpu().numpy()
                f.write(f"\n{'='*60}\n")
                f.write(f"num_interventions={num_interventions}  group={self.group_interventions}\n")
                f.write(f"c_avg BEFORE intervention (sample 0):\n  {[f'{v:.4f}' for v in s0]}\n")
                f.write(f"true_concepts (sample 0):\n  {[f'{v:.4f}' for v in tc0]}\n")
                f.write(f"intervention_mask (sample 0):\n  {mask0.tolist()}\n")
                # Mutex group sums before
                if self.mutually_exclusive_concepts:
                    f.write("Group sums BEFORE replace:\n")
                    for g in self.mutually_exclusive_concepts:
                        f.write(f"  group{g}: {sum(s0[i] for i in g):.4f}  values={[f'{s0[i]:.4f}' for i in g]}\n")
        # ─────────────────────────────────────────────────────────────────────

        # Replace predicted concepts with true values at intervened positions.
        # true_concepts is already percentile-scaled by _convert_hard_interventions_to_soft.
        # Identical to CREAM's forward_with_interventions line 1022.
        c_predicted[intervention_mask] = true_concepts[intervention_mask].type(
            c_predicted.dtype
        )

        # ── DEBUG LOGGING AFTER ───────────────────────────────────────────────
        if getattr(self, '_debug_interventions', False) and num_interventions > 0:
            with open(log_path, 'a') as f:
                s1 = c_predicted[0].detach().cpu().numpy()
                f.write(f"c_avg AFTER intervention+renorm (sample 0):\n  {[f'{v:.4f}' for v in s1]}\n")
                if self.mutually_exclusive_concepts:
                    f.write("Group sums AFTER replace+renorm:\n")
                    for g in self.mutually_exclusive_concepts:
                        f.write(f"  group{g}: {sum(s1[i] for i in g):.4f}  values={[f'{s1[i]:.4f}' for i in g]}\n")
        # ─────────────────────────────────────────────────────────────────────

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

        # Run forward once — reuse u and logits for both task and concept loss
        u  = self.u2u_model(x)
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]

        # ── Collect per-expert logits and probabilities ───────────────────────
        lambdas = torch.softmax(self._lambda_logits, dim=0)   # [M]

        all_logits = []
        per_expert_concept_loss = torch.tensor(0.0, device=x.device)
        for m_idx, u2c_m in enumerate(self.u2c_models):
            logits_m = u2c_m(Uc)                                      # [B, K] raw logits
            all_logits.append(logits_m)
            c_m = self.concept_activation_function(logits_m)          # [B, K] probs
            bce_m = F.binary_cross_entropy(
                c_m.clamp(1e-7, 1 - 1e-7), target_concepts.float()
            )
            per_expert_concept_loss = per_expert_concept_loss + lambdas[m_idx] * bce_m

        # c_avg: same aggregation as forward() — supervise the actual c seen by last_layer
        c_logits_agg = torch.stack(all_logits, dim=0).max(dim=0).values  # [B, K]
        c_avg = self.concept_activation_function(c_logits_agg)            # [B, K]

        ensemble_concept_loss = F.binary_cross_entropy(
            c_avg.clamp(1e-7, 1 - 1e-7), target_concepts.float()
        )
        # concept_loss = Σ_m λ_m·BCE(c_m) + BCE(c_avg_exact)
        per_expert_concept_loss = per_expert_concept_loss + ensemble_concept_loss

        # ── Task prediction using same c_avg ──────────────────────────────────
        if self.num_side_channel > 0:
            s = self.side_channel(Uy)
            y = self.last_layer(torch.cat((c_avg, s), dim=1))
        else:
            y = self.last_layer(c_avg)

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

        # CREAM passes concept_extractor directly as model1 (not full backbone).
        # concept_extractor(x) → [B, 128]  (matches previous_model_output_size=128)
        # backbone.forward(x) → [B, 10]    (through classifier — WRONG for u2u_model)
        concept_extractor = backbone.concept_extractor

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
            model1=concept_extractor,   # same as CREAM: passes concept_extractor not full backbone
            model2=u_to_CY,
            num_exogenous=num_exogenous,
            num_classes=num_classes,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel,
            learning_rate=learning_rate,
        )

    # forward() and forward_with_interventions_cbm() are NOT overridden here.
    # Template_CBM_MultiClass.forward() calls self.x_to_u(x) which is now
    # concept_extractor (nn.Sequential) → returns [B, 128] directly.
    # This is identical to how CREAM works in simple_main.py.
    # All intervention logic (percentile scaling, group interventions) is
    # inherited unchanged from Template_CBM_MultiClass.

