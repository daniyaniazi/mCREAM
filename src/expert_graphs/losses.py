"""
Graph Regularization Losses for mCREAM.

These losses help:
1. Keep learned graph close to expert consensus (prior loss)
2. Encourage sparse graphs (sparsity loss)
3. Ensure DAG property (acyclicity loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def prior_consistency_loss(
    alpha: Tensor,
    vote_score: Tensor,
    use_sigmoid: bool = True,
) -> Tensor:
    """
    Keep learned edge reliabilities close to expert consensus.
    
    L_prior = MSE(sigmoid(α), vote_score)
    
    This prevents the model from diverging too far from expert knowledge.
    
    Args:
        alpha: Learned edge reliability logits [n_rows, n_cols]
        vote_score: Expert voting score [n_rows, n_cols], values in [0,1]
        use_sigmoid: If True, apply sigmoid to alpha before comparison
    
    Returns:
        Scalar loss value
    """
    if use_sigmoid:
        alpha_soft = torch.sigmoid(alpha)
    else:
        alpha_soft = alpha
    
    return F.mse_loss(alpha_soft, vote_score)


def sparsity_loss(
    alpha: Tensor,
    use_sigmoid: bool = True,
) -> Tensor:
    """
    Encourage sparse graph by penalizing total edge weight.
    
    L_sparse = sum(sigmoid(α))
    
    This prevents the model from adding too many edges.
    
    Args:
        alpha: Edge reliability logits or soft values
        use_sigmoid: If True, apply sigmoid to alpha
    
    Returns:
        Scalar loss value (sum of all edge weights)
    """
    if use_sigmoid:
        alpha_soft = torch.sigmoid(alpha)
    else:
        alpha_soft = alpha
    
    return alpha_soft.sum()


def l1_sparsity_loss(
    alpha: Tensor,
    use_sigmoid: bool = True,
) -> Tensor:
    """
    L1 sparsity: mean absolute value.
    
    More commonly used than sum for regularization.
    """
    if use_sigmoid:
        alpha_soft = torch.sigmoid(alpha)
    else:
        alpha_soft = alpha
    
    return alpha_soft.abs().mean()


def acyclicity_loss(
    A_soft: Tensor,
    method: str = "trace_exp",
) -> Tensor:
    """
    Penalize cycles in the graph to maintain DAG property.
    
    For a DAG: tr(e^A) - n = 0
    
    Args:
        A_soft: Soft adjacency matrix [n, n], values in [0,1]
        method: 'trace_exp' (default) or 'polynomial'
    
    Returns:
        Scalar loss value (0 for perfect DAG)
    
    Note: Only applicable to square matrices (concept→concept graphs)
    """
    n = A_soft.shape[0]
    
    if A_soft.shape[0] != A_soft.shape[1]:
        # Non-square matrix (e.g., c2y graph), skip acyclicity
        return torch.tensor(0.0, device=A_soft.device)
    
    if method == "trace_exp":
        # tr(e^A) - n should be 0 for DAG
        # Using matrix exponential
        return torch.trace(torch.matrix_exp(A_soft)) - n
    
    elif method == "polynomial":
        # Polynomial approximation (faster but less accurate)
        # h(A) = tr((I + A/n)^n) - n
        I = torch.eye(n, device=A_soft.device)
        M = I + A_soft / n
        M_power = M
        for _ in range(n - 1):
            M_power = M_power @ M
        return torch.trace(M_power) - n
    
    else:
        raise ValueError(f"Unknown acyclicity method: {method}")


def entropy_regularization(
    pi: Tensor,
) -> Tensor:
    """
    Entropy regularization for expert weights.
    
    Encourages diverse use of experts (high entropy)
    or focuses on few experts (low entropy, use negative weight).
    
    H(π) = -Σ π_m * log(π_m)
    
    Args:
        pi: Expert weights [M], should sum to 1
    
    Returns:
        Negative entropy (minimize to maximize entropy)
    """
    # Add small epsilon for numerical stability
    pi_stable = pi + 1e-8
    return (pi_stable * torch.log(pi_stable)).sum()


class GraphRegularizationLoss(nn.Module):
    """
    Combined graph regularization loss module.
    
    L_graph = β * L_prior + γ * L_sparse + δ * L_acyclic + ε * L_entropy
    
    Designed to work with EdgeReliabilityModule, GraphAttentionModule,
    or CombinedReliabilityModule.
    """
    
    def __init__(
        self,
        prior_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        acyclicity_weight: float = 0.0,
        entropy_weight: float = 0.0,
        sparsity_type: str = "sum",  # 'sum' or 'l1'
    ):
        """
        Args:
            prior_weight: Weight for prior consistency loss (β)
            sparsity_weight: Weight for sparsity loss (γ)
            acyclicity_weight: Weight for acyclicity loss (δ)
            entropy_weight: Weight for entropy regularization (ε)
            sparsity_type: 'sum' or 'l1'
        """
        super().__init__()
        self.prior_weight = prior_weight
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.entropy_weight = entropy_weight
        self.sparsity_type = sparsity_type
    
    def forward(
        self,
        aggregation_module: nn.Module,
        A_soft: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute total graph regularization loss.
        
        Args:
            aggregation_module: The aggregation module (EdgeReliability, etc.)
            A_soft: Optional soft adjacency matrix (if not provided, computed from module)
        
        Returns:
            Total regularization loss
        """
        total_loss = torch.tensor(0.0, device=next(aggregation_module.parameters()).device 
                                  if list(aggregation_module.parameters()) 
                                  else torch.device('cpu'))
        
        # Prior consistency loss
        if self.prior_weight > 0:
            if hasattr(aggregation_module, 'alpha') and hasattr(aggregation_module, 'vote_score'):
                loss_prior = prior_consistency_loss(
                    aggregation_module.alpha,
                    aggregation_module.vote_score
                )
                total_loss = total_loss + self.prior_weight * loss_prior
        
        # Sparsity loss
        if self.sparsity_weight > 0:
            if hasattr(aggregation_module, 'alpha'):
                if self.sparsity_type == "sum":
                    loss_sparse = sparsity_loss(aggregation_module.alpha)
                else:
                    loss_sparse = l1_sparsity_loss(aggregation_module.alpha)
                total_loss = total_loss + self.sparsity_weight * loss_sparse
        
        # Acyclicity loss (only for square matrices)
        if self.acyclicity_weight > 0 and A_soft is not None:
            if A_soft.shape[0] == A_soft.shape[1]:
                loss_acyclic = acyclicity_loss(A_soft)
                total_loss = total_loss + self.acyclicity_weight * loss_acyclic
        
        # Entropy regularization for expert weights
        if self.entropy_weight > 0:
            if hasattr(aggregation_module, 'get_expert_weights'):
                pi = aggregation_module.get_expert_weights()
                if pi is not None:
                    loss_entropy = entropy_regularization(pi)
                    total_loss = total_loss + self.entropy_weight * loss_entropy
        
        return total_loss
    
    def get_individual_losses(
        self,
        aggregation_module: nn.Module,
        A_soft: Optional[Tensor] = None,
    ) -> dict:
        """
        Get individual loss components for logging.
        
        Returns:
            Dictionary of individual loss values
        """
        losses = {}
        
        if hasattr(aggregation_module, 'alpha') and hasattr(aggregation_module, 'vote_score'):
            losses['prior'] = prior_consistency_loss(
                aggregation_module.alpha,
                aggregation_module.vote_score
            ).item()
        
        if hasattr(aggregation_module, 'alpha'):
            losses['sparsity'] = sparsity_loss(aggregation_module.alpha).item()
        
        if A_soft is not None and A_soft.shape[0] == A_soft.shape[1]:
            losses['acyclicity'] = acyclicity_loss(A_soft).item()
        
        if hasattr(aggregation_module, 'get_expert_weights'):
            pi = aggregation_module.get_expert_weights()
            if pi is not None:
                losses['entropy'] = entropy_regularization(pi).item()
        
        return losses
