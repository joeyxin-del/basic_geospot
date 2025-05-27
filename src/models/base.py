from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    模型基类，定义所有模型必须实现的接口和通用方法。
    所有具体的模型实现都应该继承这个基类。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型基类。
        
        Args:
            config: 模型配置字典，包含模型的各种超参数
        """
        super().__init__()
        self.config = config or {}
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        模型前向传播。
        
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
            
        Returns:
            包含模型输出的字典，至少包含以下键：
            - 'predictions': 模型预测结果
            - 'features': 中间特征图（可选）
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算模型损失。
        
        Args:
            predictions: 模型预测结果字典
            targets: 目标值字典
            
        Returns:
            包含各项损失的字典，至少包含 'total_loss' 键
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """
        保存模型检查点。
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> None:
        """
        加载模型检查点。
        
        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """
        获取模型的可训练参数。
        
        Returns:
            参数字典，键为参数名，值为参数张量
        """
        return {name: param for name, param in self.named_parameters() 
                if param.requires_grad}
    
    def freeze_layers(self, layer_names: list) -> None:
        """
        冻结指定层的参数。
        
        Args:
            layer_names: 要冻结的层名称列表
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                
    def unfreeze_layers(self, layer_names: list) -> None:
        """
        解冻指定层的参数。
        
        Args:
            layer_names: 要解冻的层名称列表
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True 