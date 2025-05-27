from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseLoss(ABC):
    """
    损失函数基类，定义所有损失函数必须实现的接口。
    所有具体的损失函数实现都应该继承这个基类。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化损失函数基类。
        
        Args:
            config: 损失函数配置字典，包含损失函数的各种超参数
        """
        self.config = config or {}
        
    @abstractmethod
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失。
        
        Args:
            predictions: 模型预测结果字典
            targets: 目标值字典
            
        Returns:
            包含损失值的字典，至少包含 'loss' 键
        """
        pass
    
    def __call__(self, predictions: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        调用损失函数。
        
        Args:
            predictions: 模型预测结果字典
            targets: 目标值字典
            
        Returns:
            包含损失值的字典
        """
        return self.forward(predictions, targets) 