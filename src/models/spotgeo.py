from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ..utils import get_logger

from .base import BaseModel

logger = get_logger('spotgeo_model')


class ConvBlock(nn.Module):
    """
    卷积块，包含卷积、批归一化和激活函数。
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpotGEOModel(BaseModel):
    """
    SpotGEO检测模型。
    使用CNN提取特征，然后通过检测头预测目标位置和置信度。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化SpotGEO模型。
        
        Args:
            config: 模型配置字典，包含以下键：
                - backbone_channels: 骨干网络通道数列表
                - detection_channels: 检测头通道数列表
                - num_classes: 类别数量
                - use_bn: 是否使用批归一化
                - dropout: Dropout比率
        """
        super().__init__(config)
        self.config = config or {}
        
        # 配置参数
        base_backbone_channels = self.config.get('backbone_channels', [64, 128, 256, 512])
        base_detection_channels = self.config.get('detection_channels', [256, 128, 64])
        scale_factor = self.config.get('scale_factor', 0.25)
        
        # 应用缩放因子
        backbone_channels = [int(ch * scale_factor) for ch in base_backbone_channels]
        detection_channels = [int(ch * scale_factor) for ch in base_detection_channels]
        self.num_classes = self.config.get('num_classes', 1)
        use_bn = self.config.get('use_bn', True)
        self.dropout = self.config.get('dropout', 0.1)
        
        # 骨干网络
        self.backbone = nn.ModuleList()
        in_channels = 3  # RGB输入
        for out_channels in backbone_channels:
            self.backbone.append(ConvBlock(in_channels, out_channels, 
                                         use_bn=use_bn))
            in_channels = out_channels
            
        # 检测头
        self.detection_head = nn.ModuleList()
        in_channels = backbone_channels[-1]
        for out_channels in detection_channels:
            self.detection_head.append(ConvBlock(in_channels, out_channels, 
                                               use_bn=use_bn))
            in_channels = out_channels
            
        # 输出层
        self.cls_head = nn.Conv2d(detection_channels[-1], self.num_classes, 1)
        self.reg_head = nn.Conv2d(detection_channels[-1], 2, 1)  # x, y坐标
        
        # 分类器（用于测试）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(detection_channels[-1], self.num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _preprocess_images(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        预处理输入图像。
        
        Args:
            images: PIL.Image列表或张量
            
        Returns:
            预处理后的张量，形状为 [batch_size, 3, height, width]
            
        Raises:
            ValueError: 当输入为空列表或无效图像时
        """
        if isinstance(images, list):
            if not images:  # 检查空列表
                raise ValueError("输入图像列表不能为空")
            # 转换为张量
            tensors = []
            for img in images:
                if not isinstance(img, Image.Image):
                    raise ValueError("输入必须是PIL.Image列表")
                # 转换为numpy数组
                img_array = np.array(img)
                if img_array.ndim == 2:  # 灰度图
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[..., :3]
                # 转换为张量并归一化
                img_tensor = torch.from_numpy(img_array).float()
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                img_tensor = img_tensor / 255.0  # 归一化到[0,1]
                tensors.append(img_tensor)
            # 堆叠批次
            x = torch.stack(tensors)
        elif isinstance(images, torch.Tensor):
            x = images
            if x.dim() == 3:  # 单张图像
                x = x.unsqueeze(0)
            if x.dim() == 4:  # 批次图像
                if x.shape[1] == 1:  # 灰度图
                    x = x.repeat(1, 3, 1, 1)
                elif x.shape[1] == 4:  # RGBA
                    x = x[:, :3]
            else:
                raise ValueError("输入张量维度不正确")
        else:
            raise ValueError("输入类型必须是PIL.Image列表或张量")
            
        return x
        
    def forward(self, x: Union[Dict[str, Any], List[Image.Image], torch.Tensor]) -> Dict[str, Any]:
        """
        模型前向传播。
        
        Args:
            x: 输入数据，可以是：
                - 字典，包含'images'键
                - PIL.Image列表
                - 张量，形状为 [batch_size, channels, height, width]
            
        Returns:
            包含以下键的字典：
            - predictions: 包含分类和回归预测的字典
            - features: 中间特征图列表
            
        Raises:
            ValueError: 当输入数据无效时
        """
        # 处理输入
        if isinstance(x, dict):
            if 'images' not in x:
                raise ValueError("输入字典必须包含'images'键")
            if not x['images']:  # 检查空列表
                raise ValueError("输入图像列表不能为空")
            if 'labels' in x and not isinstance(x['labels'], torch.Tensor):
                raise ValueError("标签必须是torch.Tensor类型")
            if 'labels' in x and torch.any(x['labels'] < 0):
                raise ValueError("标签值不能为负数")
            x = self._preprocess_images(x['images'])
        else:
            x = self._preprocess_images(x)
            
        features = []
        
        # 骨干网络
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
            x = F.max_pool2d(x, 2)
            
        # 检测头
        for layer in self.detection_head:
            x = layer(x)
            features.append(x)
            
        # 输出预测
        cls_pred = self.cls_head(x)
        reg_pred = self.reg_head(x)
        
        # 分类预测（用于测试）
        cls_logits = self.classifier(x)
        cls_probs = F.softmax(cls_logits, dim=1)
        
        return {
            'predictions': {
                'cls': cls_pred,  # [B, num_classes, H, W]
                'reg': reg_pred,  # [B, 2, H, W]
                'logits': cls_logits,  # [B, num_classes]
                'probs': cls_probs,  # [B, num_classes]
            },
            'features': features
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算模型损失。
        
        Args:
            predictions: 模型预测结果字典，包含：
                - predictions.cls: 分类预测 [B, num_classes, H, W]
                - predictions.reg: 回归预测 [B, 2, H, W]
                - predictions.logits: 分类logits [B, num_classes]
            targets: 目标值字典，包含：
                - cls: 分类标签 [B, num_classes, H, W] 或 [B, num_classes]
                - reg: 回归标签 [B, 2, H, W]
                - mask: 有效区域掩码 [B, 1, H, W]（可选）
                
        Returns:
            包含以下键的损失字典：
            - cls_loss: 分类损失
            - reg_loss: 回归损失
            - total_loss: 总损失
        """
        pred_dict = predictions['predictions']
        cls_pred = pred_dict['cls']  # [B, num_classes, H, W]
        reg_pred = pred_dict['reg']  # [B, 2, H, W]
        cls_logits = pred_dict['logits']  # [B, num_classes]
        
        # 处理分类标签
        if 'cls' in targets:
            cls_target = targets['cls']
            if cls_target.dim() == 2:  # [B, num_classes]
                # 使用交叉熵损失
                cls_loss = F.cross_entropy(cls_logits, cls_target)
            else:  # [B, num_classes, H, W]
                # 使用带掩码的二元交叉熵
                if 'mask' in targets:
                    mask = targets['mask']
                else:
                    # 创建一个全1的掩码，形状与cls_target匹配
                    mask = torch.ones_like(cls_target[:, 0:1])  # 只取第一个通道作为掩码
                
                # 确保预测和目标具有相同的形状
                if cls_pred.shape != cls_target.shape:
                    cls_pred = cls_pred.view(cls_target.shape)
                
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_pred, cls_target, reduction='none'
                )
                cls_loss = (cls_loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)
        
        # 处理回归标签
        if 'reg' in targets:
            reg_target = targets['reg']
            if 'mask' in targets:
                mask = targets['mask']
            else:
                # 创建一个全1的掩码，形状与reg_target匹配
                mask = torch.ones_like(reg_target[:, 0:1])  # 只取第一个通道作为掩码
            
            # 确保预测和目标具有相同的形状
            if reg_pred.shape != reg_target.shape:
                reg_pred = reg_pred.view(reg_target.shape)
            
            reg_loss = F.l1_loss(reg_pred, reg_target, reduction='none')
            reg_loss = (reg_loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            reg_loss = torch.tensor(0.0, device=reg_pred.device)
        
        # 总损失
        total_loss = cls_loss + reg_loss
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }

# 在文件末尾注册模型
from src.models.registry import model_registry
model_registry.register('spotgeo')(SpotGEOModel) 
