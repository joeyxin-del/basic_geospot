from typing import Any, Dict, Optional, Type

from .base import BaseModel


class ModelRegistry:
    """
    模型注册器类，负责模型的注册和实例化。
    使用工厂模式管理不同的模型实现。
    """
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """
        注册模型的装饰器。
        
        Args:
            name: 模型名称
            
        Returns:
            装饰器函数
        """
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._models:
                raise ValueError(f"Model {name} already registered")
            cls._models[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        创建模型实例。
        
        Args:
            name: 模型名称
            config: 模型配置字典
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果指定的模型名称未注册
        """
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered. "
                           f"Available models: {list(cls._models.keys())}")
        return cls._models[name](config)
    
    @classmethod
    def list_models(cls) -> list:
        """
        列出所有已注册的模型名称。
        
        Returns:
            模型名称列表
        """
        return list(cls._models.keys())


# 创建全局模型注册器实例
model_registry = ModelRegistry() 