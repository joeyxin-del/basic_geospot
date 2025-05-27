from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim


class OptimizerFactory:
    """
    优化器工厂类，负责优化器的创建和配置。
    支持PyTorch内置的优化器，以及自定义优化器。
    """
    _optimizers: Dict[str, Type[torch.optim.Optimizer]] = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
    }
    
    @classmethod
    def register(cls, name: str) -> callable:
        """
        注册优化器的装饰器。
        
        Args:
            name: 优化器名称
            
        Returns:
            装饰器函数
        """
        def decorator(optim_cls: Type[torch.optim.Optimizer]) -> Type[torch.optim.Optimizer]:
            if name in cls._optimizers:
                raise ValueError(f"Optimizer {name} already registered")
            cls._optimizers[name] = optim_cls
            return optim_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, params: Union[nn.Module, List[nn.Parameter]], 
               config: Optional[Dict[str, Any]] = None) -> torch.optim.Optimizer:
        """
        创建优化器实例。
        
        Args:
            name: 优化器名称
            params: 模型参数或参数列表
            config: 优化器配置字典
            
        Returns:
            优化器实例
            
        Raises:
            ValueError: 如果指定的优化器名称未注册
        """
        if name not in cls._optimizers:
            raise ValueError(f"Optimizer {name} not registered. "
                           f"Available optimizers: {list(cls._optimizers.keys())}")
            
        # 获取模型参数
        if isinstance(params, nn.Module):
            params = params.parameters()
            
        # 创建优化器
        optimizer_cls = cls._optimizers[name]
        config = config or {}
        return optimizer_cls(params, **config)
    
    @classmethod
    def list_optimizers(cls) -> list:
        """
        列出所有已注册的优化器名称。
        
        Returns:
            优化器名称列表
        """
        return list(cls._optimizers.keys())
    
    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """
        获取优化器的默认配置。
        
        Args:
            name: 优化器名称
            
        Returns:
            默认配置字典
            
        Raises:
            ValueError: 如果指定的优化器名称未注册
        """
        if name not in cls._optimizers:
            raise ValueError(f"Optimizer {name} not registered")
            
        # 不同优化器的默认配置
        default_configs = {
            'sgd': {
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0001,
                'nesterov': False
            },
            'adam': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0001,
                'amsgrad': False
            },
            'adamw': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.01,
                'amsgrad': False
            },
            'rmsprop': {
                'lr': 0.01,
                'alpha': 0.99,
                'eps': 1e-8,
                'weight_decay': 0.0001,
                'momentum': 0,
                'centered': False
            }
        }
        
        return default_configs.get(name, {})


# 导出工厂类
__all__ = ['OptimizerFactory'] 