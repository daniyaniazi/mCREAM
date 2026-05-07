"""
Graph Aggregation Modules for mCREAM.

This module provides:
1. Baseline aggregation methods (Union, Intersection, Majority Vote)
2. Learnable aggregation methods (Edge-level, Graph-level, Combined)

All modules are designed to be drop-in replacements that output soft adjacency matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List


# =============================================================================
# Baseline (Non-Learnable) Aggregation Methods
# =============================================================================

class BaseAggregation(nn.Module):
    """Base class for graph aggregation methods."""
    
    def __init__(self, expert_graphs: List[Tensor]):
        """
        Args:
            expert_graphs: List of M adjacency matrices
        """
        super().__init__()
        stacked = torch.stack(expert_graphs)
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
    
    def forward(self) -> Tensor:
        """Returns aggregated soft adjacency matrix."""
        raise NotImplementedError
    
    def get_expert_weights(self) -> Optional[Tensor]:
        """Returns expert weights if applicable (for logging)."""
        return None
    
    def get_edge_reliabilities(self) -> Optional[Tensor]:
        """Returns edge reliabilities if applicable (for logging)."""
        return None


class UnionAggregation(BaseAggregation):
    """
    Union (OR) of all expert graphs.
    
    A_union[i,j] = 1 if ANY expert has edge (i,j)
    
    Properties:
    - High recall (unlikely to miss true edges)
    - Low precision (many spurious edges)
    """
    
    def forward(self) -> Tensor:
        return torch.any(self.expert_graphs.bool(), dim=0).float()


class IntersectionAggregation(BaseAggregation):
    """
    Intersection (AND) of all expert graphs.
    
    A_inter[i,j] = 1 only if ALL experts have edge (i,j)
    
    Properties:
    - High precision (clean graph)
    - Low recall (misses edges)
    """
    
    def forward(self) -> Tensor:
        return torch.all(self.expert_graphs.bool(), dim=0).float()


class MajorityVoteAggregation(BaseAggregation):
    """
    Majority vote: edges that >50% of experts include.
    
    A_majority[i,j] = 1 if vote_count[i,j] > M/2
    """
    
    def forward(self) -> Tensor:
        vote_count = self.expert_graphs.sum(dim=0)
        return (vote_count > self.num_experts / 2).float()


class AverageAggregation(BaseAggregation):
    """
    Simple average of expert graphs.
    
    A_avg[i,j] = mean of all expert values (soft output in [0,1])
    """
    
    def forward(self) -> Tensor:
        return self.expert_graphs.mean(dim=0)


# =============================================================================
# Learnable Aggregation Methods
# =============================================================================

class EdgeReliabilityModule(nn.Module):
    """
    Edge-level reliability learning (α).
    
    Learns a reliability parameter α_ij for each possible edge.
    Output: Ã = sigmoid(α)
    
    Initialization: α is set so that sigmoid(α) ≈ expert vote score
    
    How learning happens:
    - Forward: ŷ = f(ĉ, Ã) where Ã = sigmoid(α)
    - Loss: L = CrossEntropy(ŷ, y)
    - Backward: ∂L/∂α → update α
    - If edge helps prediction → α increases → edge stronger
    - If edge hurts prediction → α decreases → edge weaker
    """
    
    def __init__(
        self,
        expert_graphs: List[Tensor],
        init_from_vote: bool = True,
        learnable: bool = True,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices [n_rows, n_cols]
            init_from_vote: If True, initialize α from expert voting
            learnable: If True, α is learnable; if False, fixed to vote
        """
        super().__init__()
        
        stacked = torch.stack(expert_graphs)
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        
        # Compute vote score
        vote_score = stacked.mean(dim=0)  # [n_rows, n_cols]
        
        if init_from_vote:
            # Initialize α so sigmoid(α) ≈ vote_score
            # logit(x) = log(x / (1-x))
            alpha_init = torch.logit(vote_score.clamp(0.01, 0.99))
        else:
            # Random initialization
            alpha_init = torch.zeros_like(vote_score)
        
        if learnable:
            self.alpha = nn.Parameter(alpha_init)
        else:
            self.register_buffer('alpha', alpha_init)
        
        # Store vote score for regularization
        self.register_buffer('vote_score', vote_score)
    
    def forward(self) -> Tensor:
        """Returns soft adjacency matrix Ã = sigmoid(α)."""
        return torch.sigmoid(self.alpha)
    
    def get_edge_reliabilities(self) -> Tensor:
        """Returns current edge reliability values."""
        return torch.sigmoid(self.alpha).detach()
    
    def get_vote_score(self) -> Tensor:
        """Returns expert voting score for regularization."""
        return self.vote_score


class GraphAttentionModule(nn.Module):
    """
    Graph-level attention (π).
    
    Learns a weight π_m for each expert.
    Output: Ã = Σ π_m * A^(m)
    
    π is computed via softmax so weights sum to 1.
    """
    
    def __init__(
        self,
        expert_graphs: List[Tensor],
        init_uniform: bool = True,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices
            init_uniform: If True, initialize with uniform weights
        """
        super().__init__()
        
        stacked = torch.stack(expert_graphs)
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        
        # Initialize expert weight logits
        if init_uniform:
            pi_init = torch.zeros(self.num_experts)
        else:
            pi_init = torch.randn(self.num_experts) * 0.1
        
        self.pi_logits = nn.Parameter(pi_init)
        
        # Store vote score for reference
        vote_score = stacked.mean(dim=0)
        self.register_buffer('vote_score', vote_score)
    
    def forward(self) -> Tensor:
        """Returns weighted average graph Ã = Σ π_m * A^(m)."""
        pi = F.softmax(self.pi_logits, dim=0)  # [M], sums to 1
        # Weighted sum: einsum('m,mij->ij', pi, expert_graphs)
        A_soft = torch.einsum('m,mij->ij', pi, self.expert_graphs)
        return A_soft
    
    def get_expert_weights(self) -> Tensor:
        """Returns current expert weights π."""
        return F.softmax(self.pi_logits, dim=0).detach()
    
    def get_vote_score(self) -> Tensor:
        """Returns expert voting score for reference."""
        return self.vote_score


class CombinedReliabilityModule(nn.Module):
    """
    Combined edge-level + graph-level learning.
    
    Two-stage process:
    1. Graph-level: A_weighted = Σ π_m * A^(m)
    2. Edge-level: Ã = A_weighted * sigmoid(α)
    
    This allows both:
    - Trusting certain experts more (π)
    - Refining individual edge reliabilities (α)
    """
    
    def __init__(
        self,
        expert_graphs: List[Tensor],
        init_from_vote: bool = True,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices
            init_from_vote: If True, initialize from expert voting
        """
        super().__init__()
        
        stacked = torch.stack(expert_graphs)
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        
        # Graph-level weights
        self.pi_logits = nn.Parameter(torch.zeros(self.num_experts))
        
        # Edge-level refinement
        vote_score = stacked.mean(dim=0)
        
        if init_from_vote:
            alpha_init = torch.logit(vote_score.clamp(0.01, 0.99))
        else:
            alpha_init = torch.zeros_like(vote_score)
        
        self.alpha = nn.Parameter(alpha_init)
        self.register_buffer('vote_score', vote_score)
    
    def forward(self) -> Tensor:
        """Returns combined aggregated graph."""
        # Step 1: Graph-level aggregation
        pi = F.softmax(self.pi_logits, dim=0)
        A_weighted = torch.einsum('m,mij->ij', pi, self.expert_graphs)
        
        # Step 2: Edge-level refinement
        A_soft = A_weighted * torch.sigmoid(self.alpha)
        
        return A_soft
    
    def get_expert_weights(self) -> Tensor:
        """Returns expert weights π."""
        return F.softmax(self.pi_logits, dim=0).detach()
    
    def get_edge_reliabilities(self) -> Tensor:
        """Returns edge reliabilities α."""
        return torch.sigmoid(self.alpha).detach()
    
    def get_vote_score(self) -> Tensor:
        """Returns expert voting score."""
        return self.vote_score


# =============================================================================
# Factory Function
# =============================================================================

def create_aggregation_module(
    aggregation_type: str,
    expert_graphs: List[Tensor],
    **kwargs,
) -> nn.Module:
    """
    Factory function to create aggregation module by name.
    
    Args:
        aggregation_type: One of:
            - 'union': UnionAggregation
            - 'intersection': IntersectionAggregation
            - 'majority': MajorityVoteAggregation
            - 'average': AverageAggregation
            - 'edge': EdgeReliabilityModule
            - 'graph': GraphAttentionModule
            - 'combined': CombinedReliabilityModule
        expert_graphs: List of M adjacency matrices
        **kwargs: Additional arguments for the module
    
    Returns:
        Aggregation module
    """
    modules = {
        'union': UnionAggregation,
        'intersection': IntersectionAggregation,
        'majority': MajorityVoteAggregation,
        'average': AverageAggregation,
        'edge': EdgeReliabilityModule,
        'graph': GraphAttentionModule,
        'combined': CombinedReliabilityModule,
    }
    
    if aggregation_type not in modules:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}. "
                        f"Choose from {list(modules.keys())}")
    
    return modules[aggregation_type](expert_graphs, **kwargs)
