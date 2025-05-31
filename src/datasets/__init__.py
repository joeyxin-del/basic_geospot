"""
数据集模块，包含数据集基类和具体实现。
"""

from .base import DatasetBase
from .spotgeov2 import SpotGEOv2Dataset

__all__ = ['DatasetBase', 'SpotGEOv2Dataset'] 