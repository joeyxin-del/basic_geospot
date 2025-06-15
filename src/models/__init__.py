from typing import Dict, Any, Type
from .base import BaseModel
from .registry import model_registry

# 导出模型工厂
ModelFactory = model_registry

# 导出模型列表函数
def list_models() -> list:
    """列出所有可用的模型"""
    return model_registry.list_models()

# 导出模型创建函数
def get_model(name: str, config: Dict[str, Any] = None) -> BaseModel:
    """创建模型实例"""
    return model_registry.create(name, config)

# 导入模型以触发注册
from .spotgeo import SpotGEOModel  # 导入SpotGEO模型
from .spotgeo_res import SpotGEOModelRes  # 导入SpotGEO残差模型

__all__ = ['ModelFactory', 'list_models', 'get_model', 'SpotGEOModel', 'SpotGEOModelRes'] 