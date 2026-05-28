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
# Shared Utility
# =============================================================================

def _gumbel_sigmoid(logits: Tensor, temperature: float) -> Tensor:
    """
    Gumbel-Sigmoid: differentiable approximation to hard Bernoulli sampling.

    Adds Gumbel noise to logits before sigmoid, pushing outputs bimodally
    toward 0 or 1. Provides gradient signal even when task loss is flat.

        p = sigmoid((logit + Gumbel_noise) / T)

    Args:
        logits: Raw logits (any shape)
        temperature: Lower → more binary output (0.1 = nearly hard)
    """
    eps = 1e-8
    u = torch.rand_like(logits).clamp(eps, 1 - eps)
    gumbel_noise = -torch.log(-torch.log(u))
    return torch.sigmoid((logits + gumbel_noise) / temperature)


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
    
    Initialization: Agreement-weighted — edges with high expert agreement
    start with amplified confidence (further from 0.5), while disputed edges
    start closer to the decision boundary.
    
    How learning happens:
    - Forward: ŷ = f(ĉ, Ã) where Ã = sigmoid(α)
    - Loss: L = CrossEntropy(ŷ, y) + confidence_loss
    - Backward: ∂L/∂α → update α
    - If edge helps prediction → α increases → edge stronger
    - If edge hurts prediction → α decreases → edge weaker
    """
    
    def __init__(
        self,
        expert_graphs: List[Tensor],
        init_from_vote: bool = True,
        learnable: bool = True,
        agreement_amplification: float = 2.0,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        anneal_temperature: bool = True,
        min_temperature: float = 0.1,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices [n_rows, n_cols]
            init_from_vote: If True, initialize α from expert voting
            learnable: If True, α is learnable; if False, fixed to vote
            agreement_amplification: Scale factor for high-agreement edges.
                Higher = more confident initialization on agreed-upon edges.
            temperature: Gumbel-Sigmoid temperature (lower = more binary)
            use_gumbel: If True, use Gumbel-Sigmoid during training
            anneal_temperature: If True, temperature decays toward min_temperature
            min_temperature: Minimum temperature after annealing
        """
        super().__init__()
        
        stacked = torch.stack(expert_graphs)
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        
        # Compute vote score
        vote_score = stacked.mean(dim=0)  # [n_rows, n_cols]
        
        if init_from_vote:
            # Agreement-weighted initialization:
            # High agreement (vote near 0 or 1) → amplified logit (more confident)
            # Low agreement (vote near 0.5) → logit stays near 0 (uncertain)
            alpha_init = torch.logit(vote_score.clamp(0.01, 0.99))
            
            # Amplify: edges with strong agreement get pushed further from 0
            # agreement = |vote - 0.5| * 2, ranges from 0 (total disagreement) to 1 (unanimous)
            agreement = (vote_score - 0.5).abs() * 2
            alpha_init = alpha_init * (1.0 + (agreement_amplification - 1.0) * agreement)
        else:
            # Random initialization
            alpha_init = torch.zeros_like(vote_score)
        
        if learnable:
            self.alpha = nn.Parameter(alpha_init)
        else:
            self.register_buffer('alpha', alpha_init)
        
        # Gumbel-Sigmoid temperature
        self.use_gumbel = use_gumbel
        self.anneal_temperature = anneal_temperature
        self.min_temperature = min_temperature
        self.register_buffer('temperature', torch.tensor(temperature))

        # Store vote score and agreement for regularization
        self.register_buffer('vote_score', vote_score)
        self.register_buffer('agreement', (vote_score - 0.5).abs() * 2)

    def anneal(self, epoch: int, total_epochs: int) -> None:
        """Anneal Gumbel temperature linearly from 1.0 → min_temperature."""
        if self.anneal_temperature:
            progress = epoch / max(total_epochs - 1, 1)
            new_temp = 1.0 * (1 - progress) + self.min_temperature * progress
            self.temperature.fill_(new_temp)

    def forward(self) -> Tensor:
        """Returns soft adjacency matrix. Gumbel-Sigmoid during training."""
        if self.training and self.use_gumbel:
            return _gumbel_sigmoid(self.alpha, self.temperature.item())
        return torch.sigmoid(self.alpha)
    
    def get_edge_reliabilities(self) -> Tensor:
        """Returns current edge reliability values."""
        return torch.sigmoid(self.alpha).detach()
    
    def get_vote_score(self) -> Tensor:
        """Returns expert voting score for regularization."""
        return self.vote_score


class GraphAttentionModule(nn.Module):
    """
    Graph-level attention with per-edge expert weighting.
    
    Instead of learning only M scalar weights (too few parameters),
    this learns per-edge expert attention: for each edge (i,j), which
    experts to trust most.
    
    Two levels:
    1. Global expert weights π_m (soft baseline, M params)
    2. Per-edge expert attention w_m(i,j) (rich, M×n_rows×n_cols params)
    
    Output: Ã[i,j] = Σ_m softmax(π_m + w_m[i,j]) * A^(m)[i,j]
    
    This allows the model to trust expert 1 on edge (0,3) but expert 5 on edge (2,7).
    """
    
    def __init__(
        self,
        expert_graphs: List[Tensor],
        init_uniform: bool = True,
        per_edge_attention: bool = True,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        anneal_temperature: bool = True,
        min_temperature: float = 0.1,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices
            init_uniform: If True, initialize with uniform weights
            per_edge_attention: If True, learn per-edge expert weights
            temperature: Gumbel-Sigmoid temperature (lower = more binary)
            use_gumbel: If True, apply Gumbel-Sigmoid to final output
            anneal_temperature: If True, temperature decays toward min_temperature
            min_temperature: Minimum temperature after annealing
        """
        super().__init__()
        
        stacked = torch.stack(expert_graphs)  # [M, n_rows, n_cols]
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        self.per_edge_attention = per_edge_attention
        
        n_rows, n_cols = expert_graphs[0].shape
        
        # Global expert weight logits
        if init_uniform:
            pi_init = torch.zeros(self.num_experts)
        else:
            pi_init = torch.randn(self.num_experts) * 0.1
        self.pi_logits = nn.Parameter(pi_init)
        
        if per_edge_attention:
            # Per-edge expert attention logits
            # Initialize from agreement: experts that agree with majority get higher weight
            vote_score = stacked.mean(dim=0)  # [n_rows, n_cols]
            # Agreement of each expert with vote: high if expert matches consensus
            expert_agreement = 1.0 - (stacked - vote_score.unsqueeze(0)).abs()  # [M, n_rows, n_cols]
            # Initialize: experts that agree more get slightly higher logits
            w_init = (expert_agreement - 0.5) * 0.5  # small init, centered near 0
            self.edge_logits = nn.Parameter(w_init)  # [M, n_rows, n_cols]
        
        # Gumbel-Sigmoid temperature
        self.use_gumbel = use_gumbel
        self.anneal_temperature = anneal_temperature
        self.min_temperature = min_temperature
        self.register_buffer('temperature', torch.tensor(temperature))

        # Store vote score for reference
        vote_score = stacked.mean(dim=0)
        self.register_buffer('vote_score', vote_score)
        self.register_buffer('agreement', (vote_score - 0.5).abs() * 2)

    def anneal(self, epoch: int, total_epochs: int) -> None:
        """Anneal Gumbel temperature linearly from 1.0 → min_temperature."""
        if self.anneal_temperature:
            progress = epoch / max(total_epochs - 1, 1)
            new_temp = 1.0 * (1 - progress) + self.min_temperature * progress
            self.temperature.fill_(new_temp)

    def forward(self) -> Tensor:
        """Returns weighted average graph. Gumbel-Sigmoid applied during training."""
        if self.per_edge_attention:
            combined_logits = self.pi_logits.view(-1, 1, 1) + self.edge_logits
            weights = F.softmax(combined_logits, dim=0)  # [M, n_rows, n_cols]
            A_soft = (weights * self.expert_graphs).sum(dim=0)  # [n_rows, n_cols]
        else:
            pi = F.softmax(self.pi_logits, dim=0)
            A_soft = torch.einsum('m,mij->ij', pi, self.expert_graphs)

        if self.training and self.use_gumbel:
            # Convert soft output to logit space, then apply Gumbel-Sigmoid
            eps = 1e-6
            logits = torch.logit(A_soft.clamp(eps, 1 - eps))
            return _gumbel_sigmoid(logits, self.temperature.item())
        return A_soft
    
    def get_expert_weights(self) -> Tensor:
        """Returns global expert weights π."""
        return F.softmax(self.pi_logits, dim=0).detach()
    
    def get_edge_reliabilities(self) -> Optional[Tensor]:
        """Returns the learned soft adjacency (for logging)."""
        return self.forward().detach()
    
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
        temperature: float = 1.0,
        use_gumbel: bool = True,
        anneal_temperature: bool = True,
        min_temperature: float = 0.1,
    ):
        """
        Args:
            expert_graphs: List of M adjacency matrices
            init_from_vote: If True, initialize from expert voting
            temperature: Gumbel-Sigmoid temperature (lower = more binary)
            use_gumbel: If True, apply Gumbel-Sigmoid to alpha logits
            anneal_temperature: If True, temperature decays toward min_temperature
            min_temperature: Minimum temperature after annealing
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

        # Gumbel-Sigmoid temperature
        self.use_gumbel = use_gumbel
        self.anneal_temperature = anneal_temperature
        self.min_temperature = min_temperature
        self.register_buffer('temperature', torch.tensor(temperature))

    def anneal(self, epoch: int, total_epochs: int) -> None:
        """Anneal Gumbel temperature linearly from 1.0 → min_temperature."""
        if self.anneal_temperature:
            progress = epoch / max(total_epochs - 1, 1)
            new_temp = 1.0 * (1 - progress) + self.min_temperature * progress
            self.temperature.fill_(new_temp)

    def forward(self) -> Tensor:
        """Returns combined aggregated graph. Gumbel-Sigmoid on edge logits during training."""
        # Step 1: Graph-level aggregation
        pi = F.softmax(self.pi_logits, dim=0)
        A_weighted = torch.einsum('m,mij->ij', pi, self.expert_graphs)

        # Step 2: Edge-level refinement with Gumbel-Sigmoid
        if self.training and self.use_gumbel:
            edge_probs = _gumbel_sigmoid(self.alpha, self.temperature.item())
        else:
            edge_probs = torch.sigmoid(self.alpha)

        return A_weighted * edge_probs
    
    def get_expert_weights(self) -> Tensor:
        """Returns expert weights π."""
        return F.softmax(self.pi_logits, dim=0).detach()
    
    def get_edge_reliabilities(self) -> Tensor:
        """Returns edge reliabilities α."""
        return torch.sigmoid(self.alpha).detach()
    
    def get_vote_score(self) -> Tensor:
        """Returns expert voting score."""
        return self.vote_score


class GraphLearningMLP(nn.Module):
    """
    MLP-based graph learning: learns edge probabilities from M expert votes.

    Instead of a single learnable scalar α per edge (EdgeReliabilityModule),
    this uses a small shared MLP that takes M expert votes for each edge as
    input and outputs a probability for that edge.

    Architecture per edge (i,j):
        [vote_1, vote_2, ..., vote_M]  →  MLP  →  logit  →  Gumbel-Sigmoid  →  p_ij

    The MLP shares weights across ALL edges, so it learns a general rule:
        "given this pattern of M votes, how confident should we be in this edge?"

    Advantages over EdgeReliabilityModule (α per edge):
    - Generalizes across edges: learns a function, not independent scalars
    - Can capture complex patterns (e.g., "trust edge if experts 1,3,5 agree")
    - More parameter efficient for large graphs

    Gumbel-Sigmoid (during training):
        p = sigmoid((logit + Gumbel_noise) / temperature)
        → bimodal output, pushes edges toward 0 or 1
        → still fully differentiable (unlike hard thresholding)
    At inference: hard threshold at 0.5 → clean binary graph.
    """

    def __init__(
        self,
        expert_graphs: List[Tensor],
        hidden_dim: int = 32,
        temperature: float = 1.0,
        use_gumbel: bool = True,
        anneal_temperature: bool = True,
        min_temperature: float = 0.1,
    ):
        """
        Args:
            expert_graphs: List of M binary adjacency matrices [n_rows, n_cols]
            hidden_dim: Hidden layer size of the MLP
            temperature: Gumbel-Sigmoid temperature (lower = more binary)
            use_gumbel: If True, use Gumbel-Sigmoid during training
            anneal_temperature: If True, temperature decays toward min_temperature
            min_temperature: Minimum temperature after annealing
        """
        super().__init__()

        stacked = torch.stack(expert_graphs)  # [M, n_rows, n_cols]
        self.register_buffer('expert_graphs', stacked)
        self.num_experts = len(expert_graphs)
        self.use_gumbel = use_gumbel
        self.anneal_temperature = anneal_temperature
        self.min_temperature = min_temperature
        self.register_buffer('temperature', torch.tensor(temperature))

        # MLP: M votes → 1 edge logit (shared across all edges)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_experts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # No sigmoid here — raw logits, Gumbel-Sigmoid applied in forward()
        )

        # Initialize MLP last layer bias from vote score so we start near majority vote
        vote_score = stacked.mean(dim=0)  # [n_rows, n_cols]
        mean_vote = vote_score.mean().clamp(0.01, 0.99)
        with torch.no_grad():
            self.mlp[-1].bias.fill_(torch.logit(mean_vote).item())

        # Store vote score for reference / regularization
        self.register_buffer('vote_score', vote_score)
        self.register_buffer('agreement', (vote_score - 0.5).abs() * 2)

    def forward(self) -> Tensor:
        """
        Returns soft (or near-binary) adjacency matrix.

        During training with Gumbel-Sigmoid: values bimodal near 0 or 1.
        During eval / inference: hard threshold at 0.5 → binary float.
        """
        # expert_graphs: [M, n_rows, n_cols]
        # Permute to [n_rows, n_cols, M] so MLP sees M votes per edge
        edge_votes = self.expert_graphs.permute(1, 2, 0).float()  # [n_rows, n_cols, M]

        # MLP maps M votes → 1 logit per edge
        logits = self.mlp(edge_votes).squeeze(-1)  # [n_rows, n_cols]

        if self.training and self.use_gumbel:
            # Gumbel-Sigmoid: add Gumbel noise for bimodal, decisive output
            eps = 1e-8
            u = torch.rand_like(logits).clamp(eps, 1 - eps)
            gumbel_noise = -torch.log(-torch.log(u))
            return torch.sigmoid((logits + gumbel_noise) / self.temperature)
        else:
            # Eval: deterministic hard threshold → binary graph
            return (torch.sigmoid(logits) > 0.5).float()

    def get_soft_probs(self) -> Tensor:
        """Returns deterministic soft probabilities (no Gumbel noise)."""
        edge_votes = self.expert_graphs.permute(1, 2, 0).float()
        logits = self.mlp(edge_votes).squeeze(-1)
        return torch.sigmoid(logits).detach()

    def get_edge_reliabilities(self) -> Tensor:
        """Returns current edge probability values (for logging)."""
        return self.get_soft_probs()

    def get_expert_weights(self) -> Optional[Tensor]:
        """MLP has no expert-level weights — return None."""
        return None

    def get_vote_score(self) -> Tensor:
        """Returns expert voting score for reference."""
        return self.vote_score

    def anneal(self, epoch: int, total_epochs: int) -> None:
        """
        Anneal temperature from initial value toward min_temperature.
        Call this at the start of each epoch.

        Temperature schedule: linear decay
            T(epoch) = T_init * (1 - epoch/total_epochs) + T_min * (epoch/total_epochs)
        """
        if self.anneal_temperature:
            progress = epoch / max(total_epochs - 1, 1)
            T_init = 1.0  # always start from 1.0
            new_temp = T_init * (1 - progress) + self.min_temperature * progress
            self.temperature.fill_(new_temp)


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
        'mlp': GraphLearningMLP,          # MLP-based: M votes → edge prob (Gumbel-Sigmoid)
    }
    
    if aggregation_type not in modules:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}. "
                        f"Choose from {list(modules.keys())}")
    
    return modules[aggregation_type](expert_graphs, **kwargs)
