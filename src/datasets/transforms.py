from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
from src.utils import get_logger

logger = get_logger('transforms')

class SpotGEOTransform:
    """
    SpotGEO数据集的数据转换类。
    处理序列化的多目标数据，包括图像预处理和数据增强。
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 480),
        normalize: bool = True,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化转换类。
        
        Args:
            image_size: 输出图像大小 (width, height)
            normalize: 是否进行归一化
            augment: 是否进行数据增强
            augment_config: 数据增强配置
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.augment_config = augment_config or {}
        
        # 基础转换
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]) if normalize else T.Lambda(lambda x: x)
        ])
        
        # 数据增强转换
        if augment:
            self.augment_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5) if self.augment_config.get('horizontal_flip', False) else T.Lambda(lambda x: x),
                T.RandomVerticalFlip(p=0.5) if self.augment_config.get('vertical_flip', False) else T.Lambda(lambda x: x),
                T.RandomRotation(self.augment_config.get('rotation', 0)) if self.augment_config.get('rotation', 0) > 0 else T.Lambda(lambda x: x),
                T.ColorJitter(
                    brightness=self.augment_config.get('brightness', 0),
                    contrast=self.augment_config.get('contrast', 0)
                ) if any([self.augment_config.get('brightness', 0) > 0,
                         self.augment_config.get('contrast', 0) > 0]) else T.Lambda(lambda x: x)
            ])
        else:
            self.augment_transform = T.Lambda(lambda x: x)
            
    def _transform_coordinates(self, coords: List[List[float]], 
                             old_size: Tuple[int, int], 
                             new_size: Tuple[int, int]) -> List[List[float]]:
        """
        转换坐标以适应新的图像大小。
        
        Args:
            coords: 原始坐标列表 [[x1, y1], [x2, y2], ...]
            old_size: 原始图像大小 (width, height)
            new_size: 新图像大小 (width, height)
            
        Returns:
            转换后的坐标列表
        """
        old_w, old_h = old_size
        new_w, new_h = new_size
        
        # 计算缩放比例
        scale_x = new_w / old_w
        scale_y = new_h / old_h
        
        # 转换坐标
        transformed_coords = []
        for x, y in coords:
            new_x = x * scale_x
            new_y = y * scale_y
            transformed_coords.append([new_x, new_y])
            
        return transformed_coords
        
    def _create_target_tensors(self, labels: List[Dict[str, Any]], 
                              image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        创建目标张量。
        
        Args:
            labels: 标注列表，每个元素包含frame、num_objects和object_coords
            image_size: 图像大小 (width, height)
            
        Returns:
            包含分类和回归目标的字典
        """
        # 创建分类和回归目标张量
        cls_target = torch.zeros((len(labels), 1, image_size[1], image_size[0]))
        reg_target = torch.zeros((len(labels), 2, image_size[1], image_size[0]))
        
        # 填充目标张量
        for i, label in enumerate(labels):
            coords = label['object_coords']
            for x, y in coords:
                # 将坐标转换为整数索引
                x_idx = int(x)
                y_idx = int(y)
                
                # 确保索引在有效范围内
                if 0 <= x_idx < image_size[0] and 0 <= y_idx < image_size[1]:
                    # 设置分类目标
                    cls_target[i, 0, y_idx, x_idx] = 1.0
                    
                    # 设置回归目标（相对于网格中心的偏移）
                    reg_target[i, 0, y_idx, x_idx] = x - x_idx  # x偏移
                    reg_target[i, 1, y_idx, x_idx] = y - y_idx  # y偏移
                    
        return {
            'cls': cls_target,
            'reg': reg_target
        }
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用转换。
        
        Args:
            sample: 输入样本字典，包含：
                - images: List[PIL.Image]，序列图像列表
                - labels: List[Dict]，序列标注列表
                - sequence_name: str，序列名称
                
        Returns:
            转换后的样本字典
        """
        images = sample['images']
        labels = sample['labels']
        sequence_name = sample['sequence_name']
        
        # 转换图像
        transformed_images = []
        for img in images:
            # 获取原始图像大小
            old_size = img.size
            
            # 应用基础转换
            img_tensor = self.base_transform(img)
            
            # 应用数据增强
            if self.augment:
                img_tensor = self.augment_transform(img_tensor)
                
            transformed_images.append(img_tensor)
            
        # 转换标签
        transformed_labels = []
        for i, label in enumerate(labels):
            # 转换坐标
            coords = self._transform_coordinates(
                label['object_coords'],
                old_size,
                self.image_size
            )
            
            # 更新标签
            transformed_label = label.copy()
            transformed_label['object_coords'] = coords
            transformed_labels.append(transformed_label)
            
        # 创建目标张量
        targets = self._create_target_tensors(transformed_labels, self.image_size)
        
        return {
            'images': transformed_images,
            'labels': transformed_labels,
            'sequence_name': sequence_name,
            'targets': targets
        } 