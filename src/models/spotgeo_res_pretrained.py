from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from ..utils import get_logger
from ..losses.spotgeo import SpotGEOLoss

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


class SpotGEOModelResPretrained(BaseModel):
    """
    SpotGEO检测模型。
    使用预训练的ResNet18作为backbone，冻结其权重，只训练检测头。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化SpotGEO模型。
        
        Args:
            config: 模型配置字典，包含以下键：
                - detection_channels: 检测头通道数列表
                - num_classes: 类别数量
                - use_bn: 是否使用批归一化
                - dropout: Dropout比率
        """
        super().__init__(config)
        self.config = config or {}
        
        # 配置参数
        base_detection_channels = self.config.get('detection_channels', [256, 128, 64])
        scale_factor = self.config.get('scale_factor', 0.25)
        
        # 应用缩放因子
        detection_channels = [int(ch * scale_factor) for ch in base_detection_channels]
        self.num_classes = self.config.get('num_classes', 1)
        use_bn = self.config.get('use_bn', True)
        self.dropout = self.config.get('dropout', 0.1)
        
        # 加载预训练的ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结backbone的权重
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 移除最后的全连接层和平均池化层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 检测头
        self.detection_head = nn.ModuleList()
        in_channels = 512  # ResNet18最后一层的通道数
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
        
        # 初始化检测头权重
        self._initialize_detection_head()
        
        # 初始化损失函数
        loss_config = self.config.get('loss', {})
        self.loss_fn = SpotGEOLoss(loss_config)
        
    def _initialize_detection_head(self):
        """初始化检测头权重"""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 初始化输出层
        for m in [self.cls_head, self.reg_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        
    def _preprocess_images(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        预处理输入图像。
        
        Args:
            images: PIL.Image列表或张量
            
        Returns:
            预处理后的张量，形状为 [batch_size, 3, height, width]
        """
        if isinstance(images, list):
            if not images:
                raise ValueError("输入图像列表不能为空")
            tensors = []
            for img in images:
                if not isinstance(img, Image.Image):
                    raise ValueError("输入必须是PIL.Image列表")
                img_array = np.array(img)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = img_array[..., :3]
                img_tensor = torch.from_numpy(img_array).float()
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = img_tensor / 255.0
                # 应用ImageNet标准化
                img_tensor = self._normalize_imagenet(img_tensor)
                tensors.append(img_tensor)
            x = torch.stack(tensors)
        elif isinstance(images, torch.Tensor):
            x = images
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.dim() == 4:
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                elif x.shape[1] == 4:
                    x = x[:, :3]
                # 如果输入在[0,1]范围内，应用ImageNet标准化
                if x.max() <= 1.0:
                    x = self._normalize_imagenet(x)
            else:
                raise ValueError("输入张量维度不正确")
        else:
            raise ValueError("输入类型必须是PIL.Image列表或张量")
            
        return x
    
    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """应用ImageNet标准化"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if x.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (x - mean.to(x.device)) / std.to(x.device)
        
    def forward(self, x: Union[Dict[str, Any], List[Image.Image], torch.Tensor]) -> Dict[str, Any]:
        """
        模型前向传播。
        
        Args:
            x: 输入数据
            
        Returns:
            包含预测结果的字典
        """
        # 处理输入
        if isinstance(x, dict):
            if 'images' not in x:
                raise ValueError("输入字典必须包含'images'键")
            x = self._preprocess_images(x['images'])
        else:
            x = self._preprocess_images(x)
            
        # 提取特征
        features = []
        x = self.backbone(x)
        features.append(x)
            
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
                'cls': cls_pred,
                'reg': reg_pred,
                'logits': cls_logits,
                'probs': cls_probs,
            },
            'features': features
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算模型损失。
        
        Args:
            predictions: 模型预测结果字典
            targets: 目标值字典
                
        Returns:
            损失字典
        """
        pred_dict = predictions['predictions']
        cls_pred = pred_dict['cls']
        reg_pred = pred_dict['reg']
        
        loss_predictions = {
            'cls': cls_pred,
            'reg': reg_pred
        }
        
        loss_targets = {}
        if 'cls' in targets:
            loss_targets['cls'] = targets['cls']
        if 'reg' in targets:
            loss_targets['reg'] = targets['reg']
        
        if 'mask' not in loss_targets:
            mask = torch.ones_like(cls_pred[:, 0:1])
            loss_targets['mask'] = mask
        
        loss_dict = self.loss_fn(loss_predictions, loss_targets)
        
        return loss_dict

# 注册模型
from src.models.registry import model_registry
model_registry.register('spotgeo_res_pretrained')(SpotGEOModelResPretrained) 