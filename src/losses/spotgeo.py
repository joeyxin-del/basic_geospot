from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LossFactory
from .base import BaseLoss


class BinaryCrossEntropyLoss(nn.Module):
    """
    二元交叉熵损失实现。
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        初始化二元交叉熵损失。
        
        Args:
            pos_weight: 正样本权重
            reduction: 损失归约方式
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算二元交叉熵损失。
        
        Args:
            inputs: 模型预测的 logits [B, C, H, W]
            targets: 目标标签 [B, C, H, W]
            
        Returns:
            二元交叉熵损失值
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss 实现。
    使用 PyTorch 官方的 Focal Loss。
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化 Focal Loss。
        
        Args:
            alpha: 平衡正负样本的权重因子
            gamma: 聚焦参数，用于降低易分样本的权重
            reduction: 损失归约方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal Loss。
        
        Args:
            inputs: 模型预测的 logits [B, C, H, W]
            targets: 目标标签 [B, C, H, W]
            
        Returns:
            Focal Loss 值
        """
        # 计算 sigmoid
        probs = torch.sigmoid(inputs)
        
        # 计算二元交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 pt (预测概率与真实标签的匹配程度)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算 focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用 alpha 权重
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算 focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # 应用归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class ClassificationLoss(BaseLoss):
    """
    分类损失函数。
    支持二元交叉熵损失和 Focal Loss，可通过配置选择。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分类损失函数。
        
        Args:
            config: 损失函数配置字典，包含以下键：
                - loss_type: 损失类型，'binary' 或 'focal'（可选，默认为'focal'）
                - alpha: Focal Loss 的 alpha 参数（可选，默认为1.0）
                - gamma: Focal Loss 的 gamma 参数（可选，默认为2.0）
                - pos_weight: 二元交叉熵的正样本权重（可选）
                - reduction: 损失归约方式（可选，默认为'none'）
        """
        super().__init__(config)
        self.loss_type = self.config.get('loss_type', 'focal')
        self.reduction = self.config.get('reduction', 'none')
        
        # 根据损失类型创建相应的损失函数
        if self.loss_type == 'focal':
            alpha = self.config.get('alpha', 1.0)
            gamma = self.config.get('gamma', 2.0)
            self.loss_fn = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                reduction='none'  # 我们会在 forward 中手动处理归约
            )
        elif self.loss_type == 'binary':
            pos_weight = self.config.get('pos_weight', None)
            if pos_weight is not None:
                pos_weight = torch.tensor(pos_weight)
            self.loss_fn = BinaryCrossEntropyLoss(
                pos_weight=pos_weight,
                reduction='none'  # 我们会在 forward 中手动处理归约
            )
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}。支持的类型: 'binary', 'focal'")
        
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
        
        # 计算损失
        loss = self.loss_fn(cls_pred, cls_target)
        
        # 应用掩码并计算平均损失
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