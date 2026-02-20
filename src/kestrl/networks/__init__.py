"""Neural network architectures for KestRL.

Provides clean, modular network implementations following CleanRL patterns
with proper initialization and performance optimizations.
"""

from .mlp import MLP
from .multi_head_mlp import MultiHeadMLP

__all__ = ['MultiHeadMLP',
           'MLP']