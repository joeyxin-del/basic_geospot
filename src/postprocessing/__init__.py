"""
后处理模块
用于将模型输出转换为最终结果
"""

from .heatmap_to_coords import heatmap_to_coords

__all__ = ['heatmap_to_coords'] 