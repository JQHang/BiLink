"""
Join Edges Module

This module provides functionality to generate join edges from a heterogeneous graph.
Join edges unify multiple edge types into standardized schemas for efficient graph operations.

Two types of join edges:
- Inner Join Edge: Union of all edge types with complete information
- Outer Join Edge: Sampled neighborhoods from inner join edges

Main Functions:
- generate_join_edges: Main entry point to generate both inner and outer join edges
- generate_inner_join_edge: Generate inner join edges (union of all edges)
- generate_outer_join_edge: Generate outer join edges (sampled neighborhoods)
- add_path_to_path: Join two paths with deduplication (supports multi-hop)
- _unique_edge_join: Generic duplicate-free join for arbitrary path lengths
- compute_k_hop_bipaths: Compute bipaths by joining forward and backward paths
- match_k_hop_bipaths: Match bipaths with target pairs and calculate statistics
- extract_u_explore_paths: Extract u_node side exploration paths from bipaths
- extract_v_explore_paths: Extract v_node side exploration paths from bipaths
- generate_bipath_pairs_for_hop: Generate candidate pairs from bipaths for a specific hop
"""

from .join_edge import generate_join_edges
from .inner_join_edge import generate_inner_join_edge
from .outer_join_edge import generate_outer_join_edge
from .add_path import add_path_to_path
from .unique_edge_join import _unique_edge_join
from .bipath import compute_k_hop_bipaths, _get_matched_bipaths_schema, match_k_hop_bipaths
from .bipath_types import extract_u_explore_paths, extract_v_explore_paths
from .bipath_pair import generate_bipath_pairs_for_hop

__all__ = [
    'generate_join_edges',
    'generate_inner_join_edge',
    'generate_outer_join_edge',
    'add_path_to_path',
    '_unique_edge_join',
    'compute_k_hop_bipaths',
    '_get_matched_bipaths_schema',
    'match_k_hop_bipaths',
    'extract_u_explore_paths',
    'extract_v_explore_paths',
    'generate_bipath_pairs_for_hop',
]
