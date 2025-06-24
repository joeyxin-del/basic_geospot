import random
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A

from .base import BaseTransform
from .factory import TransformFactory


class AlbumentationsWrap(BaseTransform):
    """
    Albumentations包装基类，处理输入输出格式转换
    """
    def to_numpy(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
        """将输入图像转换为numpy数组"""
        if isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, torch.Tensor):
            return image.cpu().numpy().transpose(1, 2, 0)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")

    def from_numpy(self, image: np.ndarray, original: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """将numpy数组转换回原始类型"""
        if isinstance(original, Image.Image):
            return Image.fromarray(image)
        elif isinstance(original, torch.Tensor):
            return torch.from_numpy(image.transpose(2, 0, 1))
        elif isinstance(original, np.ndarray):
            return image
        else:
            raise TypeError(f"不支持的图像类型: {type(original)}")


@TransformFactory.register('advanced_augmentation')
class AdvancedAugmentation(AlbumentationsWrap):
    """
    高级数据增强组合。
    使用albumentations的OneOf来组织数据增强，包括：
    - 水平翻转
    - 垂直翻转
    - 随机旋转
    - 高斯噪声
    - 饱和度调整
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        
        # 创建变换流水线
        self.transform = A.Compose([
            # 几何变换组
            A.OneOf([
                A.HorizontalFlip(p=1.0),  # 水平翻转
                A.VerticalFlip(p=1.0),    # 垂直翻转
                A.Rotate(
                    limit=self.config.get('rotate_limit', 30),  # 旋转角度范围
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ], p=self.config.get('geometric_prob', 0.5)),  # 几何变换组的整体概率
            
            # 图像增强组
            A.OneOf([
                A.GaussNoise(
                    var_limit=self.config.get('noise_var_limit', (10.0, 50.0)),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0,  # 不调整色调
                    sat_shift_limit=self.config.get('saturation_limit', (-30, 30)),
                    val_shift_limit=0,  # 不调整明度
                    p=1.0
                ),
            ], p=self.config.get('enhance_prob', 0.3)),  # 图像增强组的整体概率
        ])
        
    def __call__(self, data: Union[Dict[str, Any], Image.Image]) -> Union[Dict[str, Any], Image.Image]:
        """
        应用数据增强。
        
        Args:
            data: 可以是以下两种格式之一：
                1. 单个图像 (PIL.Image)
                2. 数据字典，包含：
                   - image: PIL.Image，输入图像
                   - label: Dict，标注信息
                   - sequence_name: str，序列名称
                   - frame_idx: int，帧索引
        
        Returns:
            增强后的数据，保持与输入相同的格式
        """
        if isinstance(data, dict):
            # 处理字典格式输入
            data = data.copy()
            image = data['image']
            
            # 转换图像并应用增强
            np_image = self.to_numpy(image)
            transformed = self.transform(image=np_image)['image']
            data['image'] = self.from_numpy(transformed, image)
            
            return data
        else:
            # 处理单个图像输入
            np_image = self.to_numpy(data)
            transformed = self.transform(image=np_image)['image']
            return self.from_numpy(transformed, data)


# 导出所有数据增强类
__all__ = ['AdvancedAugmentation'] 