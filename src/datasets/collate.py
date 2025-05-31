"""
数据集的collate函数模块。
用于处理批处理时的数据整理，特别是处理不同长度的序列数据。
"""

import torch
from typing import Dict, List, Any
from torch.utils.data.dataloader import default_collate
import numpy as np
from PIL import Image

def spotgeo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义的collate函数，用于处理SpotGEOv2数据集的批次数据。
    
    Args:
        batch: 批次数据列表，每个元素是一个字典，包含：
            - images: List[PIL.Image]，序列图像列表
            - labels: List[Dict]，序列标注列表
            - sequence_name: str，序列名称
            
    Returns:
        处理后的批次数据字典，包含：
            - images: List[List[PIL.Image]]，批次图像列表
            - labels: List[List[Dict]]，批次标注列表
            - sequence_name: List[str]，批次序列名称列表
    """
    # 收集批次数据
    batch_images = []
    batch_labels = []
    batch_sequence_names = []
    
    for sample in batch:
        batch_images.append(sample['images'])
        batch_labels.append(sample['labels'])
        batch_sequence_names.append(sample['sequence_name'])
        
    return {
        'images': batch_images,  # List[List[PIL.Image]]
        'labels': batch_labels,  # List[List[Dict]]
        'sequence_name': batch_sequence_names  # List[str]
    }

# 导出函数
__all__ = ['spotgeo_collate_fn'] 