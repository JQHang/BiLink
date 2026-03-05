"""
Context mapping layer for element table construction.

This module provides abstractions for mapping context tables to element tables
with different strategies (static, time_window).
"""

from .base import ContextMapper
from .static import StaticContextMapper
from .time_window import TimeWindowContextMapper

__all__ = [
    'ContextMapper',
    'StaticContextMapper',
    'TimeWindowContextMapper',
]
