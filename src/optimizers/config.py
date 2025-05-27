from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn


@dataclass
class OptimizerConfig:
    """
    优化器配置类。
    用于管理和验证优化器的配置参数。
    """
    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    eps: float = 1e-8
    amsgrad: bool = False
    nesterov: bool = False
    alpha: Optional[float] = None
    centered: bool = False
    
    # 参数组配置
    param_groups: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """
        验证配置参数的有效性。
        """
        # 验证优化器名称
        valid_names = ['sgd', 'adam', 'adamw', 'rmsprop']
        if self.name not in valid_names:
            raise ValueError(f"Invalid optimizer name: {self.name}. "
                           f"Must be one of {valid_names}")
            
        # 验证学习率
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
            
        # 验证权重衰减
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
            
        # 验证优化器特定参数
        if self.name == 'sgd':
            if self.momentum is not None and self.momentum < 0:
                raise ValueError(f"Momentum must be non-negative, got {self.momentum}")
        elif self.name in ['adam', 'adamw']:
            if self.betas is not None:
                beta1, beta2 = self.betas
                if not (0 < beta1 < 1 and 0 < beta2 < 1):
                    raise ValueError(f"Betas must be in (0, 1), got {self.betas}")
        elif self.name == 'rmsprop':
            if self.alpha is not None and not (0 < self.alpha < 1):
                raise ValueError(f"Alpha must be in (0, 1), got {self.alpha}")
                
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典。
        
        Returns:
            配置字典
        """
        config = {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'eps': self.eps
        }
        
        # 添加优化器特定参数
        if self.name == 'sgd':
            if self.momentum is not None:
                config['momentum'] = self.momentum
            config['nesterov'] = self.nesterov
        elif self.name in ['adam', 'adamw']:
            if self.betas is not None:
                config['betas'] = self.betas
            config['amsgrad'] = self.amsgrad
        elif self.name == 'rmsprop':
            if self.alpha is not None:
                config['alpha'] = self.alpha
            if self.momentum is not None:
                config['momentum'] = self.momentum
            config['centered'] = self.centered
            
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptimizerConfig':
        """
        从字典创建配置实例。
        
        Args:
            config: 配置字典
            
        Returns:
            配置实例
        """
        return cls(**config)
    
    def create_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        创建参数组。
        
        Args:
            model: 模型实例
            
        Returns:
            参数组列表
        """
        if not self.param_groups:
            return [{'params': model.parameters()}]
            
        param_groups = []
        for group_config in self.param_groups:
            # 获取参数
            if 'module' in group_config:
                module = getattr(model, group_config['module'])
                params = module.parameters()
            else:
                params = model.parameters()
                
            # 创建参数组
            param_group = {
                'params': params,
                'lr': group_config.get('lr', self.lr),
                'weight_decay': group_config.get('weight_decay', self.weight_decay)
            }
            
            # 添加优化器特定参数
            if self.name == 'sgd':
                if 'momentum' in group_config:
                    param_group['momentum'] = group_config['momentum']
                param_group['nesterov'] = group_config.get('nesterov', self.nesterov)
            elif self.name in ['adam', 'adamw']:
                if 'betas' in group_config:
                    param_group['betas'] = group_config['betas']
                param_group['amsgrad'] = group_config.get('amsgrad', self.amsgrad)
            elif self.name == 'rmsprop':
                if 'alpha' in group_config:
                    param_group['alpha'] = group_config['alpha']
                if 'momentum' in group_config:
                    param_group['momentum'] = group_config['momentum']
                param_group['centered'] = group_config.get('centered', self.centered)
                
            param_groups.append(param_group)
            
        return param_groups


# 导出配置类
__all__ = ['OptimizerConfig'] 