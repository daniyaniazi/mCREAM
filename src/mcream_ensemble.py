"""
mCREAM Ensemble: M separate CREAM models, one per expert graph.
Aggregation happens at the prediction level (y), not the graph level.

Architecture:
    Expert Graph G^(m) -> CREAM_m -> y_m    for m = 1..M
    [y_1, ..., y_M]   -> Ensemble aggregation (average / weighted π) -> y_final

Two expert modes control what happens INSIDE each expert CREAM:

    expert_mode='hard'
        Each expert is a standard CREAM (UtoY_model) with its graph baked in
        as a fixed hard binary mask. The graph never changes during training.
        This is the baseline ensemble.

        G^m (binary) ──► UtoY_model (hard mask) ──► y_m

    expert_mode='soft_edge'  (edge-level α)
        Each expert has its own learnable α^m per edge. The graph from G^m
        initialises α^m but training can increase/decrease each edge's weight.
        Internally uses SoftMaskedLinear from mCREAM_UtoC_Y.

        G^m (binary) ──► α^m (initialised from G^m, learnable) ──► soft mask
                     ──► mCREAM_UtoC_Y (soft mask) ──► y_m

The two are combined freely with any prediction aggregation:

    Prediction aggregation (ensemble level):
        average   — ŷ = (1/M) Σ y_m                    (no learned params)
        weighted  — ŷ = Σ softmax(π_m) * y_m            (π learned end-to-end)

Experiment matrix:
    hard   + average   →  baseline ensemble
    hard   + weighted  →  graph-level π  (H3: which expert to trust)
    soft   + average   →  edge-level α   (H3: which edges to trust)
    soft   + weighted  →  combined α+π   (full proposed method)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, List, Tuple

import pytorch_lightning as pl

from .models import UtoY_model, freeze_model
from .mcream_model import mCREAM_UtoC_Y


# =============================================================================
# Prediction-level ensemble aggregation
# =============================================================================

class AverageEnsemble(nn.Module):
    """ŷ = (1/M) Σ y_m  — no learned parameters."""

    def forward(self, logits_stack: Tensor) -> Tensor:
        # logits_stack: [M, B, T]
        return logits_stack.mean(dim=0)

    def get_expert_weights(self) -> Optional[Tensor]:
        return None


class WeightedEnsemble(nn.Module):
    """
    ŷ = Σ_m softmax(π)_m * y_m

    π_logits are M learnable scalars, initialised to zero (uniform).
    Trained end-to-end from task loss — experts with better predictions
    receive higher weight automatically.
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.pi_logits = nn.Parameter(torch.zeros(num_experts))

    def forward(self, logits_stack: Tensor) -> Tensor:
        # logits_stack: [M, B, T]
        pi = F.softmax(self.pi_logits, dim=0)           # [M]
        return torch.einsum('m,mbt->bt', pi, logits_stack)

    def get_expert_weights(self) -> Tensor:
        return F.softmax(self.pi_logits, dim=0).detach()


def _make_ensemble(ensemble_type: str, num_experts: int) -> nn.Module:
    if ensemble_type == "average":
        return AverageEnsemble()
    elif ensemble_type == "weighted":
        return WeightedEnsemble(num_experts)
    else:
        raise ValueError(
            f"Unknown ensemble_type '{ensemble_type}'. Choose from: average, weighted"
        )


# =============================================================================
# Expert unit builders
# =============================================================================

def _make_hard_expert(
    full_graph: Tensor,       # (K+T)×(K+T) bool
    num_exogenous: int,
    num_concepts: int,
    num_side_channel: int,
    num_classes: int,
    learning_rate: float,
    lambda_weight: float,
    previous_model_output_size: Optional[int],
    concept_representation: str,
    side_dropout: bool,
    dropout_prob: float,
    num_hidden_layers: int,
    mutually_exclusive_concepts: Optional[List],
) -> UtoY_model:
    """
    Standard CREAM expert with a fixed hard binary mask.
    The graph is baked in at construction and never updated.
    """
    return UtoY_model(
        num_exogenous=num_exogenous,
        num_concepts=num_concepts,
        num_side_channel=num_side_channel,
        num_classes=num_classes,
        learning_rate=learning_rate,
        lambda_weight=lambda_weight,
        causal_graph=full_graph.bool(),
        masking_algorithm="zuko",
        num_hidden_layers_in_maskedmlp=num_hidden_layers,
        previous_model_output_size=previous_model_output_size,
        last_layer_mask=True,
        concept_representation=concept_representation,
        side_dropout=side_dropout,
        dropout_prob=dropout_prob,
        mutually_exclusive_concepts=mutually_exclusive_concepts,
    )


def _make_soft_expert(
    u2c_graph: Tensor,        # [K×K] float
    c2y_graph: Tensor,        # [T×(K+T)] float
    num_exogenous: int,
    num_concepts: int,
    num_side_channel: int,
    num_classes: int,
    learning_rate: float,
    lambda_weight: float,
    previous_model_output_size: Optional[int],
    concept_representation: str,
    side_dropout: bool,
    dropout_prob: float,
    num_hidden_layers: int,
    mutually_exclusive_concepts: Optional[List],
) -> mCREAM_UtoC_Y:
    """
    Soft-edge expert: each edge has a learnable reliability α^m initialised
    from the expert graph's vote score. Training adjusts α^m so the expert
    can up-weight edges that help prediction and down-weight spurious ones.

    Uses mCREAM_UtoC_Y with aggregation_type='edge' (EdgeReliabilityModule)
    which wraps a single expert graph as a list of length 1.
    The EdgeReliabilityModule initialises α from the graph and makes it learnable.
    """
    return mCREAM_UtoC_Y(
        expert_u2c_graphs=[u2c_graph.float()],
        expert_c2y_graphs=[c2y_graph.float()],
        aggregation_type="edge",           # EdgeReliabilityModule: α per edge
        # No regularization at expert level — keep it simple
        prior_weight=0.0,
        sparsity_weight=0.0,
        acyclicity_weight=0.0,
        confidence_weight=0.0,
        graph_warmup_epochs=0,             # start learning α from epoch 0
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
        num_hidden_layers_in_maskedmlp=num_hidden_layers,
        mutually_exclusive_concepts=mutually_exclusive_concepts,
    )


# =============================================================================
# mCREAM_Ensemble
# =============================================================================
# Activation proxy for save_intermediate_values compatibility
# =============================================================================

class _LastLayerProxy(nn.Module):
    """
    Fake last_layer that the LogIntermediateLayerCallback can hook into.
    During the ensemble forward pass we write the averaged [c, s] concatenation
    into self._last_input, then call this module so the hook fires with the
    correct ensemble-averaged activations — not just expert_0's activations.
    """
    def __init__(self):
        super().__init__()
        self._last_input: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self._last_input = x
        return x  # pass-through, output unused


class _EnsembleActivationProxy(nn.Module):
    """
    Mimics the pl_module.u_to_CY interface expected by LogIntermediateLayerCallback.

    The callback hooks into:
        pl_module.u_to_CY.u2u_model   → to capture exogenous features u
        pl_module.u_to_CY.last_layer  → to capture [c, s] input to task layer

    mCREAM_Ensemble.forward() calls proxy.record(u, c_avg, s_avg) each batch,
    which runs the proxy's u2u_model and last_layer so the hooks fire with the
    ensemble-averaged values rather than any single expert's values.
    """

    def __init__(
        self,
        u2u_model: nn.Module,
        num_concepts: int,
        num_side_channel: int,
        side_dropout: bool,
    ):
        super().__init__()
        self.u2u_model = u2u_model   # shared with expert_0 — same splitter
        self.last_layer = _LastLayerProxy()
        self.side_dropout = side_dropout
        self.num_concepts = num_concepts
        self.num_side_channel = num_side_channel

    def record(self, u: Tensor, c_avg: Tensor, s_avg: Optional[Tensor]) -> None:
        """
        Called from mCREAM_Ensemble.forward() with ensemble-averaged activations.
        Runs u through u2u_model (fires exogenous hook) and runs [c,s] through
        last_layer proxy (fires concept/side-channel hook).
        """
        with torch.no_grad():
            # Fire the u hook — same u for all experts (shared backbone)
            self.u2u_model(u)
            # Fire the last_layer hook with ensemble-averaged [c, s]
            if s_avg is not None:
                last_input = torch.cat([c_avg, s_avg], dim=1)
            else:
                last_input = c_avg
            self.last_layer(last_input)


# =============================================================================

class mCREAM_Ensemble(pl.LightningModule):
    """
    M expert CREAM models, each with its own graph, predictions aggregated at y level.

    expert_mode controls what happens inside each expert:
        'hard'       — fixed binary mask (standard CREAM, no graph learning)
        'soft_edge'  — learnable α^m per edge (each expert refines its own graph)

    ensemble_type controls how predictions are combined:
        'average'    — equal weight, no learned params
        'weighted'   — learnable π_m weights trained end-to-end

    Experiment matrix:
        hard  + average   →  baseline ensemble
        hard  + weighted  →  graph-level π   (learn which expert to trust)
        soft  + average   →  edge-level α    (learn which edges to trust)
        soft  + weighted  →  combined α + π  (full proposed method)
    """

    def __init__(
        self,
        backbone: pl.LightningModule,

        # Expert graphs — two formats depending on expert_mode
        expert_full_graphs: Optional[List[Tensor]] = None,   # for hard mode: (K+T)×(K+T)
        expert_u2c_graphs: Optional[List[Tensor]] = None,    # for soft mode: K×K
        expert_c2y_graphs: Optional[List[Tensor]] = None,    # for soft mode: T×(K+T)

        expert_mode: str = "hard",         # 'hard' | 'soft_edge'
        ensemble_type: str = "weighted",   # 'average' | 'weighted'

        # CREAM hyperparameters — identical for all M experts
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
        mutually_exclusive_concepts: Optional[List] = None,
        frozen_backbone: bool = True,

        **kwargs: Any,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.num_side_channel = num_side_channel
        self.lambda_weight = lambda_weight
        self.learning_rate = learning_rate
        self.expert_mode = expert_mode
        self.ensemble_type = ensemble_type

        # Shared frozen backbone
        self.backbone = backbone
        if frozen_backbone:
            freeze_model(self.backbone)

        # ------------------------------------------------------------------
        # Build M expert models
        # ------------------------------------------------------------------
        shared_kwargs = dict(
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
            mutually_exclusive_concepts=mutually_exclusive_concepts,
        )

        if expert_mode == "hard":
            assert expert_full_graphs is not None, \
                "expert_full_graphs required for expert_mode='hard'"
            self.num_experts = len(expert_full_graphs)
            self.experts = nn.ModuleList([
                _make_hard_expert(
                    full_graph=g,
                    num_hidden_layers=num_hidden_layers_in_maskedmlp,
                    **shared_kwargs,
                )
                for g in expert_full_graphs
            ])

        elif expert_mode == "soft_edge":
            assert expert_u2c_graphs is not None and expert_c2y_graphs is not None, \
                "expert_u2c_graphs and expert_c2y_graphs required for expert_mode='soft_edge'"
            self.num_experts = len(expert_u2c_graphs)
            self.experts = nn.ModuleList([
                _make_soft_expert(
                    u2c_graph=u2c,
                    c2y_graph=c2y,
                    num_hidden_layers=num_hidden_layers_in_maskedmlp,
                    **shared_kwargs,
                )
                for u2c, c2y in zip(expert_u2c_graphs, expert_c2y_graphs)
            ])

        else:
            raise ValueError(
                f"Unknown expert_mode '{expert_mode}'. Choose from: hard, soft_edge"
            )

        # Prediction-level aggregation
        self.ensemble = _make_ensemble(ensemble_type, self.num_experts)

        # Compatibility shim for save_intermediate_values / LogIntermediateLayerCallback.
        # That callback expects pl_module.u_to_CY.u2u_model and pl_module.u_to_CY.last_layer.
        # We create a thin pass-through object that holds the AVERAGED activations
        # (averaged across all M experts) rather than expert_0's activations.
        # The forward pass writes to _proxy.last_input and _proxy.u_output each step,
        # so the hooks see the true ensemble-level concept representation.
        self.u_to_CY = _EnsembleActivationProxy(
            u2u_model=self.experts[0].u2u_model if hasattr(self.experts[0], 'u2u_model') else self.experts[0].utoy.u2u_model,
            num_concepts=num_concepts,
            num_side_channel=num_side_channel if num_side_channel else 0,
            side_dropout=side_dropout,
        )

        # Loss functions (same as CREAM's Template_CBM_MultiClass)
        if num_classes == 1:
            self.task_loss_function = nn.BCEWithLogitsLoss()
        else:
            self.task_loss_function = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=[
            "backbone", "expert_full_graphs",
            "expert_u2c_graphs", "expert_c2y_graphs",
        ])

    # -------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------

    def _run_expert(self, expert: nn.Module, u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run one expert on backbone features u.
        Returns (y_logits, c_activations) regardless of expert type.

        hard expert (UtoY_model)        returns (y, c)
        soft expert (mCREAM_UtoC_Y)     returns (y, c, c_logits) — we drop c_logits
        """
        out = expert(u)
        if len(out) == 3:          # mCREAM_UtoC_Y returns (y, c, c_logits)
            return out[0], out[1]
        return out                 # UtoY_model returns (y, c)

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            y: Ensemble task logits [B, T]
            c: Mean concept activations across experts [B, K]
        """
        u = self.backbone.concept_extractor(x)   # [B, backbone_dim]

        all_y, all_c = [], []
        for expert in self.experts:
            y_m, c_m = self._run_expert(expert, u)
            all_y.append(y_m)
            all_c.append(c_m)

        logits_stack = torch.stack(all_y, dim=0)    # [M, B, T]
        y = self.ensemble(logits_stack)             # [B, T]
        c = torch.stack(all_c, dim=0).mean(dim=0)  # [B, K]

        # Fire proxy hooks so LogIntermediateLayerCallback captures the
        # ensemble-averaged c (and s) rather than any single expert's values.
        # u_to_CY.record() triggers both hooks: u2u_model hook with u,
        # and last_layer hook with [c_avg, s_avg].
        if self.num_side_channel > 0:
            # Use expert_0's side channel on the same u — all experts share the
            # same backbone output u and the same side channel architecture.
            e0 = self.experts[0]
            utoy = e0 if hasattr(e0, 'u2u_model') else e0.utoy
            u_split = utoy.u2u_model(u)
            Uy = u_split[:, self.num_concepts:]
            s_avg = utoy.side_channel(Uy)
        else:
            s_avg = None
        self.u_to_CY.record(u, c, s_avg)

        return y, c

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def _compute_loss(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        stage: str,
    ) -> Tuple[Tensor, dict]:
        x, true_concepts, y_true = batch

        u = self.backbone.concept_extractor(x)

        all_y, all_c = [], []
        total_concept_loss = torch.tensor(0.0, device=x.device)

        for expert in self.experts:
            y_m, c_m = self._run_expert(expert, u)
            all_y.append(y_m)
            all_c.append(c_m)
            # Each expert individually predicts concepts — each pays its own loss
            total_concept_loss = total_concept_loss + F.binary_cross_entropy(
                c_m.clamp(1e-7, 1 - 1e-7), true_concepts.float()
            )

        avg_concept_loss = total_concept_loss / self.num_experts

        # Ensemble prediction → task loss
        logits_stack = torch.stack(all_y, dim=0)
        y_pred = self.ensemble(logits_stack)

        if self.num_classes == 1:
            task_loss = F.binary_cross_entropy_with_logits(
                y_pred.squeeze(), y_true.float()
            )
            task_preds = (torch.sigmoid(y_pred) > 0.5).int().squeeze()
        else:
            task_loss = F.cross_entropy(y_pred, y_true)
            task_preds = y_pred.argmax(dim=1)

        task_acc = (task_preds == y_true).float().mean()
        c_avg = torch.stack(all_c, dim=0).mean(dim=0)
        concept_acc = ((c_avg > 0.5) == true_concepts).float().mean()

        total_loss = task_loss + self.lambda_weight * avg_concept_loss

        metrics = {
            f"{stage}_task_loss":        task_loss.detach(),
            f"{stage}_concept_loss":     avg_concept_loss.detach(),
            f"{stage}_task_accuracy":    task_acc.detach(),
            f"{stage}_concept_accuracy": concept_acc.detach(),
        }

        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch, "train")
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch, "val")
        self.log_dict(metrics, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._compute_loss(batch, "test")
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.learning_rate)

    # -------------------------------------------------------------------------
    # Analysis helpers
    # -------------------------------------------------------------------------

    def get_expert_weights(self) -> Optional[Tensor]:
        """Learned π weights (weighted ensemble only)."""
        return self.ensemble.get_expert_weights()

    def get_edge_reliabilities(self) -> Optional[List[Tuple[Tensor, Tensor]]]:
        """
        Per-expert learned edge reliabilities (soft_edge mode only).
        Returns list of (u2c_reliability, c2y_reliability) per expert.
        """
        if self.expert_mode != "soft_edge":
            return None
        result = []
        for expert in self.experts:
            r_u2c, r_c2y = expert.get_edge_reliabilities()
            result.append((r_u2c, r_c2y))
        return result

    def get_individual_predictions(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Per-expert predictions without aggregation, for post-hoc analysis."""
        self.eval()
        u = self.backbone.concept_extractor(x)
        all_y, all_c = [], []
        with torch.no_grad():
            for expert in self.experts:
                y_m, c_m = self._run_expert(expert, u)
                all_y.append(y_m)
                all_c.append(c_m)
        return all_y, all_c

    def collect_expert_predictions(
        self,
        dataloader,
        device: Optional[torch.device] = None,
    ) -> "pd.DataFrame":
        """
        Run the full test set and collect per-expert predictions.

        Returns a DataFrame with one row per sample:
            y_true                  — ground truth label
            y_ensemble              — ensemble hard prediction (argmax)
            y_m_logits_{t}          — raw logit for class t from expert m
            y_m_prob_{t}            — softmax probability for class t from expert m
            y_m_pred                — argmax prediction from expert m
            y_m_correct             — whether expert m was correct
            ensemble_correct        — whether ensemble was correct

        This lets you analyse:
            - Which experts were right/wrong on which samples
            - Whether the ensemble fixed individual expert errors
            - Per-class accuracy per expert
        """
        import pandas as pd

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        rows = []

        with torch.no_grad():
            for batch in dataloader:
                x, true_concepts, y_true = batch
                x, y_true = x.to(device), y_true.to(device)

                u = self.backbone.concept_extractor(x)

                # Per-expert predictions
                all_y, all_c = [], []
                for expert in self.experts:
                    y_m, c_m = self._run_expert(expert, u)
                    all_y.append(y_m)    # [B, T]
                    all_c.append(c_m)

                logits_stack = torch.stack(all_y, dim=0)   # [M, B, T]
                y_ensemble = self.ensemble(logits_stack)    # [B, T]

                probs_stack = torch.softmax(logits_stack, dim=-1)  # [M, B, T]
                ensemble_pred = y_ensemble.argmax(dim=1)           # [B]

                B = x.size(0)
                for i in range(B):
                    row = {
                        "y_true": y_true[i].item(),
                        "y_ensemble": ensemble_pred[i].item(),
                        "ensemble_correct": (ensemble_pred[i] == y_true[i]).item(),
                    }
                    for m in range(self.num_experts):
                        pred_m = logits_stack[m, i].argmax().item()
                        row[f"y_{m}_pred"] = pred_m
                        row[f"y_{m}_correct"] = (pred_m == y_true[i].item())
                        for t in range(self.num_classes):
                            row[f"y_{m}_logit_{t}"] = logits_stack[m, i, t].item()
                            row[f"y_{m}_prob_{t}"] = probs_stack[m, i, t].item()
                    rows.append(row)

        return pd.DataFrame(rows)

    def forward_with_interventions(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Intervention: replace predicted concepts with ground truth in each
        expert independently, then ensemble the results.
        """
        u = self.backbone.concept_extractor(x)
        all_y, all_c = [], []

        for expert in self.experts:
            if self.expert_mode == "hard":
                y_m, c_m = expert.forward_with_interventions(
                    u, true_concepts, num_interventions, intervention_mask
                )
            else:  # soft_edge — mCREAM_UtoC_Y
                y_m, c_m, _ = expert.forward_with_interventions(
                    u, true_concepts, num_interventions, intervention_mask
                )
            all_y.append(y_m)
            all_c.append(c_m)

        logits_stack = torch.stack(all_y, dim=0)
        y = self.ensemble(logits_stack)
        c = torch.stack(all_c, dim=0).mean(dim=0)
        return y, c
