from typing import Any, Dict, Optional, Type

from .base import BaseModel


class ModelFactory:
    """
    模型工厂类，负责模型的注册和实例化。
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


def get_model(name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    获取模型实例的便捷函数。
    
    Args:
        name: 模型名称
        config: 模型配置字典
        
    Returns:
        模型实例
    """
    return ModelFactory.create(name, config)

def list_models() -> list:
    """
    列出所有已注册的模型名称的便捷函数。
    
    Returns:
        模型名称列表
    """
    return ModelFactory.list_models()

# 更新导出列表
__all__ = ['ModelFactory', 'BaseModel', 'get_model', 'list_models'] 