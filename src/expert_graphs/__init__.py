# Expert Graph Generation and Aggregation Modules for mCREAM
from .generation import (
    generate_expert_graph,
    generate_expert_graphs_from_dag,
    load_and_split_dag,
    save_expert_graphs,
    load_expert_graphs,
)
from .aggregation import (
    EdgeReliabilityModule,
    GraphAttentionModule,
    CombinedReliabilityModule,
    UnionAggregation,
    IntersectionAggregation,
    MajorityVoteAggregation,
)
from .losses import (
    prior_consistency_loss,
    sparsity_loss,
    acyclicity_loss,
    GraphRegularizationLoss,
)

__all__ = [
    # Generation
    "generate_expert_graph",
    "generate_expert_graphs_from_dag",
    "load_and_split_dag",
    "save_expert_graphs",
    "load_expert_graphs",
    # Aggregation
    "EdgeReliabilityModule",
    "GraphAttentionModule",
    "CombinedReliabilityModule",
    "UnionAggregation",
    "IntersectionAggregation",
    "MajorityVoteAggregation",
    # Losses
    "prior_consistency_loss",
    "sparsity_loss",
    "acyclicity_loss",
    "GraphRegularizationLoss",
]
