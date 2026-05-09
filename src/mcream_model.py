"""
mCREAM Model: Multi-Expert Graph Extension of CREAM.

This module extends CREAM to handle multiple expert graphs
with learned edge/expert reliability.

Key differences from CREAM:
1. Takes multiple expert graphs instead of single DAG
2. Learns which edges/experts to trust
3. Uses soft masks instead of hard boolean masks
4. Adds graph regularization losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics.functional import accuracy
from torchvision.ops import StochasticDepth
from zuko.nn import MaskedMLP, MaskedLinear
import pytorch_lightning as pl
from typing import Any, Optional, List, Tuple

from .expert_graphs.aggregation import (
    create_aggregation_module,
    EdgeReliabilityModule,
    GraphAttentionModule,
    CombinedReliabilityModule,
)
from .expert_graphs.losses import GraphRegularizationLoss


class SoftMaskedLinear(nn.Module):
    """
    Linear layer with soft (differentiable) mask.
    
    Unlike zuko's MaskedLinear which uses hard boolean masks,
    this layer accepts soft masks in [0, 1] for gradient flow.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input [batch, in_features]
            mask: Soft mask [out_features, in_features], values in [0,1]
        
        Returns:
            Output [batch, out_features]
        """
        # Apply soft mask to weights
        masked_weight = self.weight * mask
        return F.linear(x, masked_weight, self.bias)


class SoftMaskedMLP(nn.Module):
    """
    MLP with soft masking for the first layer.
    
    Structure: SoftMaskedLinear → ReLU → Linear → ... → Linear
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [],
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # First layer is soft-masked
        layers = []
        
        if len(hidden_features) == 0:
            # Single layer
            self.first_layer = SoftMaskedLinear(in_features, out_features)
            self.hidden_layers = None
        else:
            # Multiple layers
            self.first_layer = SoftMaskedLinear(in_features, hidden_features[0])
            
            hidden_layers = [nn.ReLU()]
            prev_dim = hidden_features[0]
            for h_dim in hidden_features[1:]:
                hidden_layers.append(nn.Linear(prev_dim, h_dim))
                hidden_layers.append(nn.ReLU())
                prev_dim = h_dim
            hidden_layers.append(nn.Linear(prev_dim, out_features))
            
            self.hidden_layers = nn.Sequential(*hidden_layers)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Input [batch, in_features]
            mask: Soft mask for first layer [hidden_features[0] or out_features, in_features]
        """
        out = self.first_layer(x, mask)
        
        if self.hidden_layers is not None:
            out = self.hidden_layers(out)
        
        return out


class mCREAM_UtoC_Y(pl.LightningModule):
    """
    mCREAM: Multi-Expert Concept Reasoning with Uncertainty-Aware Masks.
    
    Extends CREAM's UtoY_model to handle multiple expert graphs
    and learn which edges/experts to trust.
    """
    
    def __init__(
        self,
        # Expert graph settings
        expert_u2c_graphs: List[Tensor],
        expert_c2y_graphs: List[Tensor],
        aggregation_type: str = "edge",  # 'edge', 'graph', 'combined', 'union', etc.
        
        # Graph regularization
        prior_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        acyclicity_weight: float = 0.0,
        
        # CREAM parameters (same as UtoY_model)
        num_exogenous: int = 76,
        num_concepts: int = 8,
        num_side_channel: int = 20,
        num_classes: int = 10,
        learning_rate: float = 0.001,
        lambda_weight: float = 1.0,
        previous_model_output_size: Optional[int] = 128,
        concept_representation: str = "soft",
        side_dropout: bool = True,
        dropout_prob: float = 0.9,
        num_hidden_layers_in_maskedmlp: int = 0,
        mutually_exclusive_concepts: Optional[List] = None,
        
        **kwargs: Any,
    ):
        super().__init__()
        
        # Store parameters
        self.num_exogenous = num_exogenous
        self.num_concepts = num_concepts
        self.num_side_channel = num_side_channel
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight
        self.previous_model_output_size = previous_model_output_size
        self.concept_representation = concept_representation
        self.side_dropout = side_dropout
        self.dropout_prob = dropout_prob
        self.mutually_exclusive_concepts = mutually_exclusive_concepts
        self.aggregation_type = aggregation_type
        
        # Regularization weights
        self.prior_weight = prior_weight
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        
        # =====================================================================
        # Graph Aggregation Modules (NEW in mCREAM)
        # =====================================================================
        self.graph_agg_u2c = create_aggregation_module(
            aggregation_type, expert_u2c_graphs
        )
        self.graph_agg_c2y = create_aggregation_module(
            aggregation_type, expert_c2y_graphs
        )
        
        # Graph regularization loss
        self.graph_reg_loss = GraphRegularizationLoss(
            prior_weight=prior_weight,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight,
        )
        
        # =====================================================================
        # Model Architecture (similar to CREAM)
        # =====================================================================
        
        # 1. Representation splitter: u → [Uc, Uy]
        if previous_model_output_size is not None:
            self.u2u_model = nn.Sequential(
                nn.Linear(previous_model_output_size, num_exogenous),
                nn.ReLU(),
            )
        else:
            self.u2u_model = nn.Identity()
        
        # 2. Concept prediction: Uc → concepts (with soft mask)
        self._init_concept_model(num_hidden_layers_in_maskedmlp)
        
        # 3. Side channel (with dropout)
        self._init_side_channel()
        
        # 4. Task prediction: [concepts, side] → task (with soft mask)
        self._init_task_model()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['expert_u2c_graphs', 'expert_c2y_graphs'])
    
    def _init_concept_model(self, num_hidden_layers: int):
        """
        Initialize concept prediction with soft masking.
        
        Layer: Uc [in_dim] → concepts [K]
        Where in_dim = num_exogenous - num_side_channel
        
        The mask shape will be [K × in_dim] after expansion.
        """
        in_dim = self.num_exogenous - self.num_side_channel
        out_dim = self.num_concepts
        
        # Verify dimensions are compatible
        assert in_dim % out_dim == 0, \
            f"Input dim ({in_dim}) must be divisible by num_concepts ({out_dim})"
        
        self.input_per_concept = in_dim // out_dim
        
        if num_hidden_layers == 0:
            self.u2c_model = SoftMaskedLinear(in_dim, out_dim)
        else:
            hidden = [out_dim] * num_hidden_layers
            self.u2c_model = SoftMaskedMLP(in_dim, out_dim, hidden)
    
    def _init_side_channel(self):
        """Initialize side channel with optional dropout."""
        if self.num_side_channel > 0:
            self.side_channel = nn.Sequential(
                nn.Identity(),
                nn.Linear(self.num_side_channel, self.num_classes),
                nn.ReLU(),
            )
            
            if self.side_dropout:
                self.side_channel = nn.Sequential(
                    *self.side_channel,
                    StochasticDepth(p=self.dropout_prob, mode="batch"),
                )
        else:
            self.side_channel = None
    
    def _init_task_model(self):
        """Initialize task prediction with soft masking."""
        if self.num_side_channel > 0 or self.side_dropout:
            in_dim = self.num_concepts + self.num_classes  # concepts + side channel output
        else:
            in_dim = self.num_concepts
        
        out_dim = self.num_classes
        
        self.last_layer = SoftMaskedLinear(in_dim, out_dim)
    
    def _build_u2c_mask(self, A_soft: Tensor) -> Tensor:
        """
        Build mask for concept prediction layer (Uc → concepts).
        
        === What is u2c? ===
        - Input: Uc features [batch, in_dim] where in_dim = num_exogenous - num_side_channel
        - Output: Concept logits [batch, K] where K = num_concepts
        - The mask controls: which input dimensions can influence which concept output
        
        === What is A_soft? ===
        - Shape: [K × K] (concept → concept adjacency)
        - A_soft[i, j] = probability that concept i influences concept j
        - Example: A_soft[0, 1] = 0.8 means c0 → c1 with 80% confidence
        
        === Why expand? ===
        - Each concept has `input_per_concept` input dimensions
        - Example: K=8 concepts, in_dim=56 → input_per_concept = 56/8 = 7
        - If c0 → c1, then ALL 7 dimensions of c0's input can affect c1's output
        
        === The transformation ===
        1. A_soft[i,j] means "ci causes cj" (row=cause, col=effect)
        2. For Linear layer, we need mask[out, in] where out receives from in
        3. So mask[j, dims_of_i] = A_soft[i, j] (transpose relationship)
        4. Then expand each concept column to input_per_concept columns
        
        Returns:
            mask: [K × in_dim] soft mask for the u2c layer
        """
        K = self.num_concepts
        in_dim = self.num_exogenous - self.num_side_channel
        input_per_concept = in_dim // K
        
        # Step 1: Transpose - convert (cause→effect) to (output←input)
        # A_soft[i,j] means i→j, we need mask[j,i] for output j seeing input i
        A_T = A_soft.T  # [K × K]
        
        # Step 2: Expand columns using Kronecker product
        # Each concept column becomes input_per_concept columns
        # This replicates the connectivity for all dimensions of each concept
        mask = A_T.repeat_interleave(input_per_concept, dim=1)  # [K × (K * input_per_concept)]
        
        return mask  # Shape: [K, in_dim]
    
    def _build_c2y_mask(self, A_soft: Tensor) -> Tensor:
        """
        Build mask for task prediction layer ([concepts, side] → tasks).
        
        === What is c2y? ===
        - Input: [concepts, side_channel_output] = [K + T] dimensions
        - Output: Task logits [batch, T] where T = num_classes
        - The mask controls: which concepts/side-channel can influence which task
        
        === What is A_soft? ===
        - Shape: [T × (K + T)] (task rows, all columns from original DAG)
        - A_soft[t, c] = probability that concept c influences task t
        - A_soft[t, K+t2] = probability that task t2 influences task t (usually 0)
        
        === Why NO expansion needed (unlike u2c)? ===
        - In u2c: Each concept has 7 input dims → need Kronecker expansion
        - In c2y: Each concept is 1 dim (after activation), side_channel is T dims
        - Input to last_layer: [batch, K + T] where K=concepts, T=side_output
        - This ALREADY matches the DAG shape [T × (K + T)]!
        
        === The transformation ===
        1. Take first K columns: A_soft[:, :K] → concept→task connections [T × K]
        2. For side channel: allow full connectivity (ones) [T × T]
        3. Concatenate: [T × (K + T)]
        
        Note: CREAM also uses _replicate_columns with num_replicates=1 which 
        effectively keeps columns as-is (no expansion).
        
        Returns:
            mask: [T × (K + T)] soft mask for the c2y layer
        """
        T = self.num_classes
        K = self.num_concepts
        
        # Step 1: Extract concept→task connections (first K columns)
        # A_soft[:, c] tells us how much concept c influences each task
        concept_mask = A_soft[:, :K]  # [T, K]
        
        # Step 2: Handle side channel
        # Side channel is processed through: Uy [num_side_channel] → Linear → [T] 
        # We allow full connectivity from side channel to tasks
        if self.num_side_channel > 0 or self.side_dropout:
            side_mask = torch.ones(
                T, T,  # [num_tasks, num_tasks] for side channel output
                device=A_soft.device, 
                dtype=A_soft.dtype
            )
            full_mask = torch.cat([concept_mask, side_mask], dim=1)  # [T, K + T]
        else:
            full_mask = concept_mask  # [T, K]
        
        return full_mask
    
    def concept_activation_function(self, c: Tensor) -> Tensor:
        """Apply appropriate activation to concept logits."""
        if self.concept_representation == "logits":
            return c
        elif self.concept_representation == "hard":
            c = torch.sigmoid(c)
            return (c > 0.5).float() + c - c.detach()  # Straight-through
        elif self.concept_representation == "soft":
            return torch.sigmoid(c)
        elif self.concept_representation == "group_soft":
            return self._apply_group_softmax(c)
        elif self.concept_representation == "group_hard":
            c = self._apply_group_softmax(c)
            return (c > 0.5).float() + c - c.detach()
        else:
            return torch.sigmoid(c)
    
    def _apply_group_softmax(self, c: Tensor) -> Tensor:
        """Apply softmax to mutually exclusive concept groups."""
        if self.mutually_exclusive_concepts is None:
            return torch.sigmoid(c)
        
        temp_c = c.clone()
        for group in self.mutually_exclusive_concepts:
            group_outputs = c[:, group]
            softmaxed = F.softmax(group_outputs, dim=1)
            temp_c[:, group] = softmaxed
        
        return temp_c
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features from backbone [batch, previous_model_output_size]
        
        Returns:
            y: Task logits [batch, num_classes]
            c: Concept activations [batch, num_concepts] (after activation function)
            c_logits: Concept logits [batch, num_concepts] (before activation, for loss)
        """
        # Get soft adjacency matrices from aggregation modules
        A_soft_u2c = self.graph_agg_u2c()
        A_soft_c2y = self.graph_agg_c2y()
        
        # Build masks
        mask_u2c = self._build_u2c_mask(A_soft_u2c)
        mask_c2y = self._build_c2y_mask(A_soft_c2y)
        
        # 1. Split representation
        u = self.u2u_model(x)
        Uc = u[:, :self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel:]
        
        # 2. Predict concepts with soft mask
        c_logits = self.u2c_model(Uc, mask_u2c)
        c = self.concept_activation_function(c_logits)
        
        # 3. Side channel
        if self.side_channel is not None:
            s = self.side_channel(Uy)
            last_input = torch.cat([c, s], dim=1)
        else:
            last_input = c
        
        # 4. Predict task with soft mask
        y = self.last_layer(last_input, mask_c2y)
        
        return y, c, c_logits
    
    def _compute_loss(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        stage: str = "train",
    ) -> Tuple[Tensor, dict]:
        """
        Compute total loss.
        
        Follows CREAM's loss calculation:
        - Concept loss: BCEWithLogitsLoss on concept LOGITS (not activations)
        - Task loss: CrossEntropy on task logits
        
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of individual metrics
        """
        x, true_concepts, y_true = batch
        
        # Forward pass - now returns logits too
        y_pred, c_pred, c_logits = self(x)
        
        # Task loss (same as CREAM)
        if self.num_classes == 1:
            task_loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_true.float())
            task_preds = (torch.sigmoid(y_pred) > 0.5).int().squeeze()
            task_acc = (task_preds == y_true).float().mean()
        else:
            task_loss = F.cross_entropy(y_pred, y_true)
            task_preds = y_pred.argmax(dim=1)
            task_acc = (task_preds == y_true).float().mean()
        
        # Concept loss (CREAM-compatible)
        # CREAM uses BCEWithLogitsLoss on the logits, not the activations
        # This is important because CREAM's calculate_mixed_loss uses concept_logits
        concept_loss = F.binary_cross_entropy_with_logits(
            c_logits,  # Use logits, not activated predictions!
            true_concepts.float()
        )
        
        # Concept accuracy (threshold at 0.5 on activations)
        concept_acc = ((c_pred > 0.5) == true_concepts).float().mean()
        
        # Base loss: L = L_task + λ * L_concept
        base_loss = task_loss + self.lambda_weight * concept_loss
        
        # Graph regularization loss (only during training)
        if stage == "train":
            graph_loss_u2c = self.graph_reg_loss(self.graph_agg_u2c, self.graph_agg_u2c())
            graph_loss_c2y = self.graph_reg_loss(self.graph_agg_c2y, self.graph_agg_c2y())
            graph_loss = graph_loss_u2c + graph_loss_c2y
        else:
            graph_loss = torch.tensor(0.0, device=x.device)
        
        total_loss = base_loss + graph_loss
        
        metrics = {
            f"{stage}_task_loss": task_loss,
            f"{stage}_concept_loss": concept_loss,
            f"{stage}_graph_loss": graph_loss,
            f"{stage}_task_accuracy": task_acc,
            f"{stage}_concept_accuracy": concept_acc,
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # =========================================================================
    # Logging and Analysis Methods
    # =========================================================================
    
    def get_learned_graphs(self) -> Tuple[Tensor, Tensor]:
        """Get current learned soft adjacency matrices."""
        return self.graph_agg_u2c(), self.graph_agg_c2y()
    
    def get_expert_weights(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get expert weights if using graph-level aggregation."""
        w_u2c = self.graph_agg_u2c.get_expert_weights() if hasattr(self.graph_agg_u2c, 'get_expert_weights') else None
        w_c2y = self.graph_agg_c2y.get_expert_weights() if hasattr(self.graph_agg_c2y, 'get_expert_weights') else None
        return w_u2c, w_c2y
    
    def get_edge_reliabilities(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get edge reliabilities if using edge-level aggregation."""
        r_u2c = self.graph_agg_u2c.get_edge_reliabilities() if hasattr(self.graph_agg_u2c, 'get_edge_reliabilities') else None
        r_c2y = self.graph_agg_c2y.get_edge_reliabilities() if hasattr(self.graph_agg_c2y, 'get_edge_reliabilities') else None
        return r_u2c, r_c2y


# =============================================================================
# Full mCREAM Model (with backbone) - Similar to Template_CBM_MultiClass
# =============================================================================

def freeze_model(model: nn.Module) -> None:
    """Freeze model parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


class mCREAM_Full(pl.LightningModule):
    """
    Full mCREAM model combining backbone (x→u) with mCREAM_UtoC_Y (u→c,y).
    
    This is analogous to CREAM's Template_CBM_MultiClass.
    
    Architecture:
        x (image) → backbone (x2u) → u (features) → mCREAM_UtoC_Y → (y, c)
    """
    
    def __init__(
        self,
        backbone: pl.LightningModule,
        mcream_model: mCREAM_UtoC_Y,
        frozen_backbone: bool = True,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            backbone: Pretrained backbone model (e.g., FashionMNIST_for_CBM)
            mcream_model: The mCREAM_UtoC_Y model
            frozen_backbone: Whether to freeze the backbone
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        
        self.x_to_u = backbone
        self.u_to_CY = mcream_model
        self.frozen_backbone = frozen_backbone
        self.learning_rate = learning_rate
        
        if frozen_backbone:
            freeze_model(self.x_to_u)
        
        # Copy some attributes from mcream_model for convenience
        self.num_classes = mcream_model.num_classes
        self.num_concepts = mcream_model.num_concepts
        self.num_side_channel = mcream_model.num_side_channel
        self.lambda_weight = mcream_model.lambda_weight
        
        # Loss functions (same as CREAM)
        if self.num_classes == 1:
            self.task_loss_function = nn.BCEWithLogitsLoss()
        else:
            self.task_loss_function = nn.CrossEntropyLoss()
        self.concept_loss_function = nn.BCEWithLogitsLoss()
        
        self.save_hyperparameters(ignore=['backbone', 'mcream_model'])
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass: image → features → concepts/task.
        
        Args:
            x: Input images [batch, channels, height, width]
        
        Returns:
            y: Task logits [batch, num_classes]
            c: Concept activations [batch, num_concepts]
            c_logits: Concept logits [batch, num_concepts]
        """
        # Extract features from backbone
        u = self.x_to_u.concept_extractor(x)  # [batch, feature_dim]
        
        # Pass through mCREAM
        y, c, c_logits = self.u_to_CY(u)
        
        return y, c, c_logits
    
    def training_step(self, batch, batch_idx):
        """Training step using full forward pass (x → u → c,y)."""
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
    
    def _compute_loss(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        stage: str = "train",
    ) -> Tuple[Tensor, dict]:
        """Compute loss using full forward pass (same pattern as CREAM)."""
        x, true_concepts, y_true = batch
        
        # Full forward pass
        y_pred, c_pred, c_logits = self(x)
        
        # Task loss (using loss class like CREAM - handles shapes automatically)
        task_loss = self.task_loss_function(y_pred, y_true)
        
        if self.num_classes == 1:
            task_preds = (torch.sigmoid(y_pred) > 0.5).int()
            task_acc = (task_preds == y_true).float().mean()
        else:
            task_preds = y_pred.argmax(dim=1)
            task_acc = (task_preds == y_true).float().mean()
        
        # Concept loss (using loss class like CREAM)
        concept_loss = self.concept_loss_function(c_logits, true_concepts.float())
        concept_acc = ((c_pred > 0.5) == true_concepts).float().mean()
        
        # Base loss
        base_loss = task_loss + self.lambda_weight * concept_loss
        
        # Graph regularization (only during training)
        if stage == "train":
            graph_loss_u2c = self.u_to_CY.graph_reg_loss(
                self.u_to_CY.graph_agg_u2c, 
                self.u_to_CY.graph_agg_u2c()
            )
            graph_loss_c2y = self.u_to_CY.graph_reg_loss(
                self.u_to_CY.graph_agg_c2y,
                self.u_to_CY.graph_agg_c2y()
            )
            graph_loss = graph_loss_u2c + graph_loss_c2y
        else:
            graph_loss = torch.tensor(0.0, device=x.device)
        
        total_loss = base_loss + graph_loss
        
        metrics = {
            f"{stage}_task_loss": task_loss,
            f"{stage}_concept_loss": concept_loss,
            f"{stage}_graph_loss": graph_loss,
            f"{stage}_task_accuracy": task_acc,
            f"{stage}_concept_accuracy": concept_acc,
        }
        
        return total_loss, metrics
    
    def configure_optimizers(self):
        # Only optimize non-frozen parameters
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(params, lr=self.learning_rate)
    
    # Delegate graph analysis methods to inner model
    def get_learned_graphs(self) -> Tuple[Tensor, Tensor]:
        return self.u_to_CY.get_learned_graphs()
    
    def get_expert_weights(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self.u_to_CY.get_expert_weights()
    
    def get_edge_reliabilities(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self.u_to_CY.get_edge_reliabilities()
