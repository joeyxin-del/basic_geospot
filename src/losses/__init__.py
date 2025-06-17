from typing import Any, Dict, Optional, Type

from .base import BaseLoss


class LossFactory:
    """
    损失函数工厂类，负责损失函数的注册和实例化。
    使用工厂模式管理不同的损失函数实现。
    """
    _losses: Dict[str, Type[BaseLoss]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """
        注册损失函数的装饰器。
        
        Args:
            name: 损失函数名称
            
        Returns:
            装饰器函数
        """
        def decorator(loss_cls: Type[BaseLoss]) -> Type[BaseLoss]:
            if name in cls._losses:
                raise ValueError(f"Loss {name} already registered")
            cls._losses[name] = loss_cls
            return loss_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseLoss:
        """
        创建损失函数实例。
        
        Args:
            name: 损失函数名称
            config: 损失函数配置字典
            
        Returns:
            损失函数实例
            
        Raises:
            ValueError: 如果指定的损失函数名称未注册
        """
        if name not in cls._losses:
            raise ValueError(f"Loss {name} not registered. "
                           f"Available losses: {list(cls._losses.keys())}")
        return cls._losses[name](config)
    
    @classmethod
    def list_losses(cls) -> list:
        """
        列出所有已注册的损失函数名称。
        
        Returns:
            损失函数名称列表
        """
        return list(cls._losses.keys())


# 导出工厂类
__all__ = ['LossFactory', 'BaseLoss']

# 导入损失函数以触发注册
from .spotgeo import ClassificationLoss, RegressionLoss, SpotGEOLoss 