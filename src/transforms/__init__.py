from typing import Any, Dict, List, Optional, Type, Union

from .base import BaseTransform, Compose, ToTensor, Normalize, Resize
from .image import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from .sequence import SequenceCrop, SequenceNoise, SequenceMasking, SequenceMixup, SequenceCutout
from .factory import TransformFactory, TransformChain


class TransformFactory:
    """
    数据增强工厂类，负责数据增强的注册和实例化。
    使用工厂模式管理不同的数据增强实现。
    """
    _transforms: Dict[str, Type[BaseTransform]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """
        注册数据增强的装饰器。
        
        Args:
            name: 数据增强名称
            
        Returns:
            装饰器函数
        """
        def decorator(transform_cls: Type[BaseTransform]) -> Type[BaseTransform]:
            if name in cls._transforms:
                raise ValueError(f"Transform {name} already registered")
            cls._transforms[name] = transform_cls
            return transform_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseTransform:
        """
        创建数据增强实例。
        
        Args:
            name: 数据增强名称
            config: 数据增强配置字典
            
        Returns:
            数据增强实例
            
        Raises:
            ValueError: 如果指定的数据增强名称未注册
        """
        if name not in cls._transforms:
            raise ValueError(f"Transform {name} not registered. "
                           f"Available transforms: {list(cls._transforms.keys())}")
            
        # 创建数据增强实例
        transform_cls = cls._transforms[name]
        return transform_cls(config)
    
    @classmethod
    def create_chain(cls, transforms: List[Dict[str, Any]]) -> 'TransformChain':
        """
        创建数据增强链。
        
        Args:
            transforms: 数据增强配置列表，每个元素是一个字典，包含：
                - name: 数据增强名称
                - config: 数据增强配置字典（可选）
                
        Returns:
            数据增强链实例
        """
        transform_list = []
        for transform_config in transforms:
            name = transform_config['name']
            config = transform_config.get('config')
            transform = cls.create(name, config)
            transform_list.append(transform)
        return TransformChain(transform_list)
    
    @classmethod
    def list_transforms(cls) -> list:
        """
        列出所有已注册的数据增强名称。
        
        Returns:
            数据增强名称列表
        """
        return list(cls._transforms.keys())


class TransformChain(BaseTransform):
    """
    数据增强链，按顺序应用多个数据增强。
    """
    def __init__(self, transforms: List[BaseTransform]):
        """
        初始化数据增强链。
        
        Args:
            transforms: 数据增强列表
        """
        super().__init__()
        self.transforms = transforms
        
    def __call__(self, data: Any) -> Any:
        """
        按顺序应用数据增强。
        
        Args:
            data: 输入数据
            
        Returns:
            增强后的数据
        """
        for transform in self.transforms:
            data = transform(data)
        return data


# 导出工厂类、链类和常用增强类
__all__ = [
    'TransformFactory', 'TransformChain', 'BaseTransform',
    'Compose', 'ToTensor', 'Normalize', 'Resize',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRotation',
    'SequenceCrop', 'SequenceNoise', 'SequenceMasking', 'SequenceMixup', 'SequenceCutout'
] 