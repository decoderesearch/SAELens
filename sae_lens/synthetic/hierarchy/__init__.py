"""
Hierarchical feature modifier for activation generators.

This module provides HierarchyNode, which enforces hierarchical dependencies
on feature activations. Child features are deactivated when their parent is inactive,
and children can optionally be mutually exclusive.

Based on Noa Nabeshima's Matryoshka SAEs:
https://github.com/noanabeshima/matryoshka-saes/blob/main/toy_model.py
"""

from sae_lens.synthetic.hierarchy.config import HierarchyConfig
from sae_lens.synthetic.hierarchy.hierarchy import Hierarchy, generate_hierarchy
from sae_lens.synthetic.hierarchy.modifier import (
    ActivationsModifier,
    hierarchy_modifier,
)
from sae_lens.synthetic.hierarchy.node import HierarchyNode

__all__ = [
    "ActivationsModifier",
    "Hierarchy",
    "HierarchyConfig",
    "HierarchyNode",
    "generate_hierarchy",
    "hierarchy_modifier",
]
