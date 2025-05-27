from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LossFactory
from .base import BaseLoss


class ClassificationLoss(BaseLoss):
    """
    分类损失函数。
    使用带掩码的二元交叉熵损失。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分类损失函数。
        
        Args:
            config: 损失函数配置字典，包含以下键：
                - pos_weight: 正样本权重（可选）
                - reduction: 损失归约方式（可选，默认为'none'）
        """
        super().__init__(config)
        self.pos_weight = self.config.get('pos_weight', None)
        if self.pos_weight is not None:
            self.pos_weight = torch.tensor(self.pos_weight)
        self.reduction = self.config.get('reduction', 'none')
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算分类损失。
        
        Args:
            predictions: 模型预测结果字典，包含：
                - cls: 分类预测 [B, num_classes, H, W]
            targets: 目标值字典，包含：
                - cls: 分类标签 [B, num_classes, H, W]
                - mask: 有效区域掩码 [B, 1, H, W]
                
        Returns:
            包含以下键的损失字典：
            - loss: 分类损失
        """
        cls_pred = predictions['cls']
        cls_target = targets['cls']
        mask = targets['mask']
        
        # 计算带掩码的二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            cls_pred, cls_target, 
            pos_weight=self.pos_weight.to(cls_pred.device) if self.pos_weight is not None else None,
            reduction=self.reduction
        )
        
        # 应用掩码并计算平均损失
        if self.reduction == 'none':
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            
        return {'loss': loss}


class RegressionLoss(BaseLoss):
    """
    回归损失函数。
    使用带掩码的L1损失。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化回归损失函数。
        
        Args:
            config: 损失函数配置字典，包含以下键：
                - reduction: 损失归约方式（可选，默认为'none'）
        """
        super().__init__(config)
        self.reduction = self.config.get('reduction', 'none')
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算回归损失。
        
        Args:
            predictions: 模型预测结果字典，包含：
                - reg: 回归预测 [B, 2, H, W]
            targets: 目标值字典，包含：
                - reg: 回归标签 [B, 2, H, W]
                - mask: 有效区域掩码 [B, 1, H, W]
                
        Returns:
            包含以下键的损失字典：
            - loss: 回归损失
        """
        reg_pred = predictions['reg']
        reg_target = targets['reg']
        mask = targets['mask']
        
        # 计算带掩码的L1损失
        loss = F.l1_loss(reg_pred, reg_target, reduction=self.reduction)
        
        # 应用掩码并计算平均损失
        if self.reduction == 'none':
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            
        return {'loss': loss}


class SpotGEOLoss(BaseLoss):
    """
    SpotGEO检测模型的组合损失函数。
    结合分类损失和回归损失。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化SpotGEO损失函数。
        
        Args:
            config: 损失函数配置字典，包含以下键：
                - cls_weight: 分类损失权重（可选，默认为1.0）
                - reg_weight: 回归损失权重（可选，默认为1.0）
                - cls_config: 分类损失配置（可选）
                - reg_config: 回归损失配置（可选）
        """
        super().__init__(config)
        self.cls_weight = self.config.get('cls_weight', 1.0)
        self.reg_weight = self.config.get('reg_weight', 1.0)
        
        # 创建分类和回归损失函数
        self.cls_loss = ClassificationLoss(self.config.get('cls_config'))
        self.reg_loss = RegressionLoss(self.config.get('reg_config'))
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算组合损失。
        
        Args:
            predictions: 模型预测结果字典，包含：
                - cls: 分类预测 [B, num_classes, H, W]
                - reg: 回归预测 [B, 2, H, W]
            targets: 目标值字典，包含：
                - cls: 分类标签 [B, num_classes, H, W]
                - reg: 回归标签 [B, 2, H, W]
                - mask: 有效区域掩码 [B, 1, H, W]
                
        Returns:
            包含以下键的损失字典：
            - cls_loss: 分类损失
            - reg_loss: 回归损失
            - total_loss: 总损失
        """
        # 计算分类损失
        cls_loss_dict = self.cls_loss(predictions, targets)
        cls_loss = cls_loss_dict['loss']
        
        # 计算回归损失
        reg_loss_dict = self.reg_loss(predictions, targets)
        reg_loss = reg_loss_dict['loss']
        
        # 计算总损失
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }


# 注册损失函数
LossFactory.register('classification')(ClassificationLoss)
LossFactory.register('regression')(RegressionLoss)
LossFactory.register('spotgeo')(SpotGEOLoss) 