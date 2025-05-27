from typing import Any, Dict, Optional, Type

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class SchedulerFactory:
    """
    学习率调度器工厂类，负责调度器的创建和配置。
    支持PyTorch内置的调度器，以及自定义调度器。
    """
    _schedulers: Dict[str, Type[torch.optim.lr_scheduler._LRScheduler]] = {
        'step': lr_scheduler.StepLR,
        'multi_step': lr_scheduler.MultiStepLR,
        'exponential': lr_scheduler.ExponentialLR,
        'cosine': lr_scheduler.CosineAnnealingLR,
        'cosine_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'reduce_on_plateau': lr_scheduler.ReduceLROnPlateau,
        'one_cycle': lr_scheduler.OneCycleLR,
        'cyclic': lr_scheduler.CyclicLR,
    }
    
    @classmethod
    def register(cls, name: str) -> callable:
        """
        注册调度器的装饰器。
        
        Args:
            name: 调度器名称
            
        Returns:
            装饰器函数
        """
        def decorator(scheduler_cls: Type[torch.optim.lr_scheduler._LRScheduler]) -> Type[torch.optim.lr_scheduler._LRScheduler]:
            if name in cls._schedulers:
                raise ValueError(f"Scheduler {name} already registered")
            cls._schedulers[name] = scheduler_cls
            return scheduler_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, optimizer: torch.optim.Optimizer, 
               config: Optional[Dict[str, Any]] = None) -> torch.optim.lr_scheduler._LRScheduler:
        """
        创建调度器实例。
        
        Args:
            name: 调度器名称
            optimizer: 优化器实例
            config: 调度器配置字典
            
        Returns:
            调度器实例
            
        Raises:
            ValueError: 如果指定的调度器名称未注册
        """
        if name not in cls._schedulers:
            raise ValueError(f"Scheduler {name} not registered. "
                           f"Available schedulers: {list(cls._schedulers.keys())}")
            
        # 创建调度器
        scheduler_cls = cls._schedulers[name]
        config = config or {}
        return scheduler_cls(optimizer, **config)
    
    @classmethod
    def list_schedulers(cls) -> list:
        """
        列出所有已注册的调度器名称。
        
        Returns:
            调度器名称列表
        """
        return list(cls._schedulers.keys())
    
    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """
        获取调度器的默认配置。
        
        Args:
            name: 调度器名称
            
        Returns:
            默认配置字典
            
        Raises:
            ValueError: 如果指定的调度器名称未注册
        """
        if name not in cls._schedulers:
            raise ValueError(f"Scheduler {name} not registered")
            
        # 不同调度器的默认配置
        default_configs = {
            'step': {
                'step_size': 30,
                'gamma': 0.1
            },
            'multi_step': {
                'milestones': [30, 60, 90],
                'gamma': 0.1
            },
            'exponential': {
                'gamma': 0.95
            },
            'cosine': {
                'T_max': 100,
                'eta_min': 0
            },
            'cosine_warm_restarts': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 0
            },
            'reduce_on_plateau': {
                'mode': 'min',
                'factor': 0.1,
                'patience': 10,
                'verbose': False
            },
            'one_cycle': {
                'max_lr': 0.1,
                'total_steps': 100,
                'pct_start': 0.3,
                'div_factor': 25.0,
                'final_div_factor': 10000.0
            },
            'cyclic': {
                'base_lr': 0.001,
                'max_lr': 0.1,
                'step_size_up': 2000,
                'step_size_down': None,
                'mode': 'triangular',
                'gamma': 1.0
            }
        }
        
        return default_configs.get(name, {})


# 导出工厂类
__all__ = ['SchedulerFactory'] 