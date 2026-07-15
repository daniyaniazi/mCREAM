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
        # ── Alpha soft-masking (new, off by default) ──────────────────────────
        use_alpha: bool = False,            # True → SoftMaskedLinear instead of MaskedMLP
        alpha_l1_weight: float = 0.0001,  # γ: L1 sparsity weight on sigmoid(alpha)
        **kwargs: Any,
    ) -> None:

        # Store expert graphs before calling parent (parent calls init_concept_concept)
        self._expert_graphs = expert_graphs
        self.num_experts    = len(expert_graphs)
        self.use_alpha      = use_alpha
        self.alpha_l1_weight = alpha_l1_weight

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

        if use_alpha:
            # ── Alpha path: shared learnable edge importance [K, K] ───────────
            # sigmoid(alpha_logits[i,j]) ∈ (0,1) = importance of edge i→j
            # Init positive (sigmoid(1.0)=0.73): assume edges matter, let L1 prune.
            # Zero init (0.5) lets L1 win too early before concept loss builds signal.
            self._alpha_logits = nn.Parameter(torch.ones(num_concepts, num_concepts))

            # SoftMaskedLinear uses alpha to gate each expert's binary mask softly
            self.u2c_models = nn.ModuleList([
                SoftMaskedLinear(
                    K=num_concepts,
                    D=input_per_concept,
                    binary_graph=g[:-num_classes, :-num_classes],
                    alpha_logits=self._alpha_logits,   # shared reference across all M
                )
                for g in expert_graphs
            ])
        else:
            # ── Original path: hard binary MaskedMLP per expert ───────────────
            self.u2c_models = nn.ModuleList([
                self._build_u2c_from_graph(g, input_per_concept, num_hidden_layers_in_maskedmlp)
                for g in expert_graphs
            ])

        # Learnable per-expert concept loss weights λ_m (Kavya: multi-task learning style)
        # Random init breaks symmetry so gradients can differentiate experts.
        # Softmax ensures weights sum to 1 and stay positive.
        self._lambda_logits = nn.Parameter(
            torch.randn(self.num_experts) * 0.1   # small random, softmax → near-uniform
        )

    @property
    def alpha_prob(self) -> Tensor:
        """Edge importance probabilities [K, K] in (0,1). Only valid when use_alpha=True."""
        if not self.use_alpha:
            raise AttributeError("alpha_prob only available when use_alpha=True")
        return torch.sigmoid(self._alpha_logits)

    def alpha_l1_loss(self) -> Tensor:
        """L1 sparsity penalty on alpha. Only valid when use_alpha=True."""
        return self.alpha_prob.sum()

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
        weights = torch.softmax(self._lambda_logits, dim=0)  # [M]
        all_c_logits = []
        for u2c_m in self.u2c_models:
            all_c_logits.append(u2c_m(Uc))   # raw logits [B, K]
        c_logits_stacked = torch.stack(all_c_logits, dim=0)                        # [M, B, K]
        c_logits_agg = (weights[:, None, None] * c_logits_stacked).sum(dim=0)      # [B, K] weighted mean
        c = self.concept_activation_function(c_logits_agg)                         # activate once
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
        weights = torch.softmax(self._lambda_logits, dim=0)  # [M]
        all_c_logits = []
        for u2c_m in self.u2c_models:
            all_c_logits.append(u2c_m(Uc))
        c_logits_stacked = torch.stack(all_c_logits, dim=0)                       # [M, B, K]
        c_logits_agg = (weights[:, None, None] * c_logits_stacked).sum(dim=0)     # [B, K] weighted mean
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
                diff0 = [f'{s0[i] - tc0[i]:+.4f}' for i in range(len(s0))]
                f.write(f"\n{'='*60}\n")
                f.write(f"num_interventions={num_interventions}  group={self.group_interventions}\n")
                f.write(f"c_avg BEFORE intervention (sample 0):\n  {[f'{v:.4f}' for v in s0]}\n")
                f.write(f"true_concepts        (sample 0):\n  {[f'{v:.4f}' for v in tc0]}\n")
                f.write(f"diff (c - true)      (sample 0):\n  {diff0}\n")
                f.write(f"intervention_mask    (sample 0):\n  {mask0.tolist()}\n")
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

    # ── _get_preds_loss_accuracy is NOT used — lives in mCREAM_GraphEnsemble ──
    # Kept here for reference only. mCREAM_GraphEnsemble._get_preds_loss_accuracy
    # is the one actually called (has access to task_loss_function, lambda_weight etc.)
    #
    # def _get_preds_loss_accuracy(self, batch: Tensor):
    #     u, target_concepts, y_true = batch   # u already computed by mCREAM_GraphEnsemble
    #     Uc = u[:, : self.num_exogenous - self.num_side_channel]
    #     Uy = u[:, self.num_exogenous - self.num_side_channel:]
    #     lambdas = torch.softmax(self._lambda_logits, dim=0)   # [M]
    #     all_logits = []
    #     per_expert_concept_loss = torch.tensor(0.0, device=u.device)
    #     for m_idx, u2c_m in enumerate(self.u2c_models):
    #         logits_m = u2c_m(Uc)
    #         all_logits.append(logits_m)
    #         c_m = self.concept_activation_function(logits_m)
    #         bce_m = F.binary_cross_entropy(c_m.clamp(1e-7, 1 - 1e-7), target_concepts.float())
    #         per_expert_concept_loss = per_expert_concept_loss + lambdas[m_idx] * bce_m
    #     c_logits_stacked = torch.stack(all_logits, dim=0)
    #     c_logits_agg = (lambdas[:, None, None] * c_logits_stacked).sum(dim=0)
    #     c_avg = self.concept_activation_function(c_logits_agg)
    #     ensemble_concept_loss = F.binary_cross_entropy(c_avg.clamp(1e-7, 1 - 1e-7), target_concepts.float())
    #     per_expert_concept_loss = per_expert_concept_loss + ensemble_concept_loss  # disabled in outer version
    #     ...


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
        use_alpha: bool = False,           # True → soft alpha masking
        alpha_l1_weight: float = 0.0001, # L1 weight on sigmoid(alpha)
        loss_type: str = "per_expert",   # "per_expert" | "ensemble" | "both"
    ):
        self._loss_type = loss_type
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
            use_alpha=use_alpha,
            alpha_l1_weight=alpha_l1_weight,
            loss_type=loss_type,
        )

        super().__init__(
            model1=concept_extractor,   # same as CREAM: passes concept_extractor not full backbone
            model2=u_to_CY,
            num_exogenous=num_exogenous,
            num_classes=num_classes,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel,
            learning_rate=learning_rate,
            lambda_weight=lambda_weight,   # BUG FIX: was missing → parent used default 0.01 → concept supervision ignored
        )

    # forward() and forward_with_interventions_cbm() are NOT overridden here.
    # Template_CBM_MultiClass.forward() calls self.x_to_u(x) which is now
    # concept_extractor (nn.Sequential) → returns [B, 128] directly.
    # This is identical to how CREAM works in simple_main.py.
    # All intervention logic (percentile scaling, group interventions) is
    # inherited unchanged from Template_CBM_MultiClass.

    def _get_preds_loss_accuracy(self, batch):
        """
        Override Template_CBM_MultiClass._get_preds_loss_accuracy.

        Inlines per-expert concept loss using self.task_loss_function etc.
        from the outer model — avoids AttributeError when delegating to inner.
        """
        x, target_concepts, y_true = batch

        if self.interventions is True:
            task_logits, concept_output = self.forward_with_interventions_cbm(
                x, target_concepts, y_true
            )
            from .models import calculate_mixed_loss
            return calculate_mixed_loss(
                task_logits=task_logits,
                concept_logits=concept_output,
                target_concepts=target_concepts,
                y=y_true,
                concept_loss_function=self.concept_loss_function,
                task_loss_function=self.task_loss_function,
                num_concepts=self.num_concepts,
                num_classes=self.num_classes,
                lambda_weight=self.lambda_weight,
            )

        # ── One forward pass through backbone + M expert blocks ──────────────
        u   = self.x_to_u(x)                                         # [B, 128]
        u2  = self.u_to_CY.u2u_model(u)                              # [B, 128]
        Uc  = u2[:, : self.u_to_CY.num_exogenous - self.u_to_CY.num_side_channel]
        Uy  = u2[:, self.u_to_CY.num_exogenous - self.u_to_CY.num_side_channel:]

        # ── Per-expert concept loss ───────────────────────────────────────────
        lambdas = torch.softmax(self.u_to_CY._lambda_logits, dim=0)  # [M]

        all_logits = []
        per_expert_concept_loss = torch.tensor(0.0, device=x.device)
        expert_bce_list = []
        for m_idx, u2c_m in enumerate(self.u_to_CY.u2c_models):
            logits_m = u2c_m(Uc)
            all_logits.append(logits_m)
            c_m = self.u_to_CY.concept_activation_function(logits_m)
            bce_m = F.binary_cross_entropy(
                c_m.clamp(1e-7, 1 - 1e-7), target_concepts.float()
            )
            expert_bce_list.append(bce_m.item())
            per_expert_concept_loss = per_expert_concept_loss + lambdas[m_idx] * bce_m

        # c_avg — identical aggregation to forward() and forward_with_interventions()
        c_logits_stacked = torch.stack(all_logits, dim=0)                          # [M, B, K]
        c_logits_agg     = (lambdas[:, None, None] * c_logits_stacked).sum(dim=0)  # [B, K]
        c_avg            = self.u_to_CY.concept_activation_function(c_logits_agg)  # [B, K]

        ensemble_concept_loss = F.binary_cross_entropy(
            c_avg.clamp(1e-7, 1 - 1e-7), target_concepts.float()
        )
        ensemble_bce = ensemble_concept_loss.item()

        loss_type = getattr(self, '_loss_type', 'per_expert')
        if loss_type == 'per_expert':
            concept_loss_for_backprop = per_expert_concept_loss
        elif loss_type == 'ensemble':
            concept_loss_for_backprop = ensemble_concept_loss
        else:  # 'both'
            concept_loss_for_backprop = per_expert_concept_loss + ensemble_concept_loss

        # ── Loss logging ──────────────────────────────────────────────────────
        if getattr(self, '_debug_loss', False):
            lambda_vals = torch.softmax(self.u_to_CY._lambda_logits, dim=0).detach().cpu().tolist()
            print("\n--- concept loss breakdown ---")
            for m_idx, (bce_m, lam_m) in enumerate(zip(expert_bce_list, lambda_vals)):
                print(f"  expert[{m_idx}]: BCE={bce_m:.4f}  λ={lam_m:.4f}  weighted={lam_m*bce_m:.4f}")
            print(f"  ensemble BCE(c_avg, true) = {ensemble_bce:.4f}  [NOT added to loss]")
            print(f"  per_expert_concept_loss   = {per_expert_concept_loss.item():.4f}")
            print(f"  combined concept_loss     = {per_expert_concept_loss.item():.4f}")

        # ── Task prediction ───────────────────────────────────────────────────
        if self.u_to_CY.side_dropout is True and self.u_to_CY.masking_algorithm == "none":
            s = self.u_to_CY.side_channel(Uc)
            y = self.u_to_CY.last_layer(torch.cat((c_avg, s), dim=1))
        elif self.u_to_CY.num_side_channel > 0:
            s = self.u_to_CY.side_channel(Uy)
            y = self.u_to_CY.last_layer(torch.cat((c_avg, s), dim=1))
        else:
            y = self.u_to_CY.last_layer(c_avg)

        # ── Task loss — uses self.task_loss_function from outer model ─────────
        if self.num_classes == 1:
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

        # ── Alpha L1 regularization (only when use_alpha=True) ───────────────
        if self.u_to_CY.use_alpha:
            alpha_reg  = self.u_to_CY.alpha_l1_loss() * self.u_to_CY.alpha_l1_weight
            total_loss = task_loss + self.lambda_weight * concept_loss_for_backprop + alpha_reg
        else:
            total_loss = task_loss + self.lambda_weight * concept_loss_for_backprop

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
# SoftMaskedLinear — used by UtoY_MultiGraph when use_alpha=True
# Defined after mCREAM_GraphEnsemble to keep existing classes uncluttered
# =============================================================================

class SoftMaskedLinear(nn.Module):
    """
    Linear layer where the binary expert mask is softened by a shared alpha matrix.

    For each directed edge i→j in the concept graph:
        effective_contribution = alpha[i,j] * G_m[i,j] * W[j, i_dims] * Uc[i_dims]

    alpha[i,j] = sigmoid(alpha_logits[i,j]) ∈ (0,1)
        - high → concept i strongly influences concept j
        - low  → edge is suppressed even if present in expert graph
    G_m[i,j] = binary expert graph (structural zeros never overridden by alpha)

    alpha_logits is a shared nn.Parameter owned by UtoY_AlphaGraph — passed in
    as a reference so all M experts share the same alpha.
    """

    def __init__(
        self,
        K: int,                         # num_concepts
        D: int,                         # input_per_concept (num_exogenous - num_side) // K
        binary_graph: BoolTensor,       # [K, K] expert binary u2c graph
        alpha_logits: nn.Parameter,     # [K, K] shared, owned externally
    ):
        super().__init__()
        self.K = K
        self.D = D
        self.alpha_logits_ref = alpha_logits   # shared reference — NOT owned here

        # Learnable weights — init larger than default 0.01 so alpha gets
        # meaningful gradient signal from concept loss before L1 suppresses it
        self.weight = nn.Parameter(torch.randn(K, K * D) * 0.1)
        self.bias   = nn.Parameter(torch.zeros(K))

        # Hard structural mask from expert graph — fixed, never trained
        #
        # CREAM convention: G[i,j]=1 means edge i→j (concept i is INPUT to concept j)
        # Weight matrix W shape: [K_out, K_in*D]
        #   row j = output concept j
        #   cols i*D:(i+1)*D = input dims of concept i
        #
        # So hard_mask[j, i*D:(i+1)*D] = G[i,j]  (is concept i an input to concept j?)
        # Build by transposing G first: G.T[j,i] = G[i,j]
        G = binary_graph.float()                           # [K, K]  G[i,j]=edge i→j
        Gt = G.T                                           # [K, K]  Gt[j,i]=G[i,j]
        hard = Gt.unsqueeze(-1).expand(K, K, D)           # [K, K, D]  hard[j,i,d]=G[i,j]
        hard = hard.reshape(K, K * D)                     # [K, K*D]  hard[j, i*D+d]=G[i,j]
        self.register_buffer('hard_mask', hard)

    def refresh_graph(self, new_binary_graph: BoolTensor) -> None:
        """Replace hard_mask with a new expert graph in-place.
        Called every N epochs by GraphRefreshCallback.
        alpha_logits and weight W are NOT reset — they continue accumulating signal.
        """
        G  = new_binary_graph.float()
        Gt = G.T
        hard = Gt.unsqueeze(-1).expand(self.K, self.K, self.D)
        hard = hard.reshape(self.K, self.K * self.D)
        self.hard_mask.copy_(hard)   # in-place update of registered buffer

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, K*D]  where x[:, i*D:(i+1)*D] = u-dims of concept i
        alpha_prob = torch.sigmoid(self.alpha_logits_ref)          # [K, K] alpha[i,j]=importance of i→j

        # Expand alpha to weight matrix shape [K_out, K_in*D]
        # alpha_exp[j, i*D:(i+1)*D] = alpha[i,j]  (importance of edge i→j)
        alpha_T = alpha_prob.T                                                 # [K, K]  alpha_T[j,i]=alpha[i,j]
        alpha_exp = alpha_T.unsqueeze(-1).expand(self.K, self.K, self.D)      # [K, K, D]
        alpha_exp = alpha_exp.reshape(self.K, self.K * self.D)                # [K, K*D]

        # Soft mask: alpha gates ONLY edges present in expert graph
        # G[i,j]=0 → hard_mask[j,i*D:]=0 → zero regardless of alpha ✓
        # G[i,j]=1 → soft_mask[j,i*D:] = alpha[i,j] ∈ (0,1) ✓
        soft_mask = alpha_exp * self.hard_mask                                 # [K, K*D]

        effective_W = self.weight * soft_mask                                  # [K, K*D]
        return x @ effective_W.T + self.bias                                   # [B, K]


# =============================================================================
# Graph Refresh Callback — dynamic expert graph augmentation
# =============================================================================

class GraphRefreshCallback(pl.Callback):
    """
    Every `refresh_every` epochs, regenerate M new random expert graphs and
    update the hard_mask in each SoftMaskedLinear — without touching alpha or W.

    Why this helps:
        Alpha accumulates gradient signal from MANY different random graphs.
        GT edges are present in every graph → consistent gradient → alpha grows.
        Noise-only edges differ every refresh → inconsistent gradient → L1 wins.

    Only active when use_alpha=True. No-op otherwise.

    Args:
        dag_path:        path to GT DAG CSV
        num_classes:     T — needed to extract u2c block from full graph
        p_base:          base noise for consensus generation (shared by all experts)
        p_private:       private noise per expert (controls pairwise consensus)
        refresh_every:   regenerate graphs every this many epochs
        base_seed:       starting seed; incremented each refresh for reproducibility
    """

    def __init__(
        self,
        dag_path: str,
        num_classes: int,
        p_base: float = 0.25,
        p_private: float = 0.05,
        refresh_every: int = 5,
        base_seed: int = 1000,
    ):
        super().__init__()
        self.dag_path      = dag_path
        self.num_classes   = num_classes
        self.p_base        = p_base
        self.p_private     = p_private
        self.refresh_every = refresh_every
        self.base_seed     = base_seed
        self._refresh_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1   # 1-indexed
        if epoch % self.refresh_every != 0:
            return

        # Only act when use_alpha is enabled on the inner model
        inner = getattr(pl_module, 'u_to_CY', None)
        if inner is None or not getattr(inner, 'use_alpha', False):
            return

        import numpy as np
        import pandas as pd
        from pathlib import Path

        # Load GT graph
        gt_df  = pd.read_csv(self.dag_path, index_col=0)
        gt_bool = (gt_df.values != 0).astype(int)
        K = gt_bool.shape[0] - self.num_classes
        G_star = gt_bool[:K, :K]

        # Generate new consensus expert graphs
        rng_base = np.random.default_rng(self.base_seed + self._refresh_count * 100)
        # Base flip
        total = K * K
        diag  = np.array([i * K + i for i in range(K)])
        pos   = np.setdiff1d(np.arange(total), diag)
        n_base = max(1, int(len(pos) * self.p_base))
        base_flip = rng_base.choice(pos, size=n_base, replace=False)
        G_base = G_star.copy()
        rows, cols = base_flip // K, base_flip % K
        G_base[rows, cols] = 1 - G_base[rows, cols]

        M = inner.num_experts
        new_graphs = []
        for m in range(M):
            rng_m  = np.random.default_rng(self.base_seed + self._refresh_count * 100 + m + 1)
            n_priv = max(1, int(len(pos) * self.p_private))
            priv_flip = rng_m.choice(pos, size=n_priv, replace=False)
            G_m = G_base.copy()
            pr, pc = priv_flip // K, priv_flip % K
            G_m[pr, pc] = 1 - G_m[pr, pc]
            new_graphs.append(torch.tensor(G_m.astype(bool), dtype=torch.bool))

        # Update hard_mask in each SoftMaskedLinear
        device = next(pl_module.parameters()).device
        for m, u2c_m in enumerate(inner.u2c_models):
            if hasattr(u2c_m, 'refresh_graph'):
                u2c_m.refresh_graph(new_graphs[m].to(device))

        self._refresh_count += 1
        print(f'[GraphRefresh] epoch={epoch}  refresh #{self._refresh_count}  '
              f'p_base={self.p_base}  p_private={self.p_private}')
