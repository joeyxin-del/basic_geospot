from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ..utils import get_logger
from ..losses.spotgeo import SpotGEOLoss

from .base import BaseModel

logger = get_logger('spotgeo_model')


class ConvBlock(nn.Module):
    """
    深度可分离卷积块，包含：
    1. 深度卷积（Depthwise Convolution）
    2. 逐点卷积（Pointwise Convolution）
    3. 批归一化和激活函数
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_bn: bool = True):
        super().__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                  stride=stride, padding=padding, groups=in_channels,
                                  bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1,
                                  stride=1, padding=0, bias=not use_bn)
        # 批归一化和激活函数
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, use_bn)
        
        # 当输入输出通道数不同时，需要投影层
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if use_bn:
                self.shortcut_bn = nn.BatchNorm2d(out_channels)
            else:
                self.shortcut_bn = nn.Identity()
        else:
            self.shortcut = nn.Identity()
            self.shortcut_bn = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut_bn(self.shortcut(x))
        return self.conv(x) + residual


class SpotGEOModelResFusion(BaseModel):
    """
    SpotGEO检测模型。
    使用CNN提取特征，然后通过FPN融合特征，最后使用P2尺度进行检测。
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
        # detection_channels = [int(ch * scale_factor) for ch in base_detection_channels]
        self.num_classes = self.config.get('num_classes', 1)
        use_bn = self.config.get('use_bn', True)
        self.dropout = self.config.get('dropout', 0.1)
        
        # 保存缩放后的通道数作为类属性
        self.scaled_channels = backbone_channels
        
        # FPN通道数
        self.fpn_channels = self.config.get('fpn_channels', 256)
        
        # 骨干网络
        self.backbone = nn.ModuleList()
        self.backbone_features = {}  # 存储不同层级的特征
        in_channels = 1
        
        if self.config.get('backbone_type') == 'resnet':
            # 残差网络backbone
            res_blocks_per_layer = self.config.get('res_blocks_per_layer', [2, 2, 2, 2])
            assert len(backbone_channels) == len(res_blocks_per_layer)
            
            for i, (out_channels, num_blocks) in enumerate(zip(backbone_channels, res_blocks_per_layer)):
                layer_blocks = []
                for block_idx in range(num_blocks):
                    stride = 2 if block_idx == 0 else 1
                    layer_blocks.append(ResBlock(in_channels, out_channels, 
                                              stride=stride, use_bn=use_bn))
                    in_channels = out_channels
                self.backbone.append(nn.Sequential(*layer_blocks))
        
        # FPN层
        # 获取实际的backbone通道数（考虑scale_factor的影响）
        scaled_channels = [int(ch) for ch in backbone_channels]
        
        # 计算每个层级的实际通道数
        p2_in_channels = scaled_channels[-3]  # 64
        p3_in_channels = scaled_channels[-2]  # 128
        p4_in_channels = scaled_channels[-1]  # 256
        
        self.fpn = nn.ModuleDict({
            # 横向连接，将backbone特征转换为相同的通道数
            'lateral_p4': nn.Conv2d(p4_in_channels, self.fpn_channels, 1),  # 256->128
            'lateral_p3': nn.Conv2d(p3_in_channels, self.fpn_channels, 1),  # 128->128
            'lateral_p2': nn.Conv2d(p2_in_channels, self.fpn_channels, 1),  # 64->128
            
            # 3x3卷积融合特征
            'smooth_p2': ConvBlock(self.fpn_channels, self.fpn_channels)
        })
        
        # 检测头
        # 分类头：fpn_channels -> 8 -> 1
        self.cls_head = nn.Sequential(
            ConvBlock(self.fpn_channels, 8, kernel_size=3, padding=1, use_bn=use_bn),
            # ConvBlock(8, 8, kernel_size=3, padding=1, use_bn=use_bn),
            nn.Conv2d(8, self.num_classes, 1)
        )
        
        # 回归头：fpn_channels -> 8 -> 2
        self.reg_head = nn.Sequential(
            ConvBlock(self.fpn_channels, 8, kernel_size=3, padding=1, use_bn=use_bn),
            # ConvBlock(8, 8, kernel_size=3, padding=1, use_bn=use_bn),
            nn.Conv2d(8, 2, 1)
        )
        
        # 打印模型结构信息
        logger.info(f"Backbone type: {self.config.get('backbone_type', 'conv')}")
        logger.info(f"Backbone channels: {backbone_channels}")
        if self.config.get('backbone_type') == 'resnet':
            logger.info(f"Res blocks per layer: {self.config.get('res_blocks_per_layer', [2, 2, 2, 2])}")
        logger.info(f"Scale factor: {scale_factor}")
        logger.info(f"Total backbone layers: {len(self.backbone)}")
        
        # 初始化权重
        self._initialize_weights()
        
        # 初始化损失函数
        loss_config = self.config.get('loss', {})
        self.loss_fn = SpotGEOLoss(loss_config)
        
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
                if img_array.ndim == 2:  # 已经是灰度图
                    img_array = img_array[np.newaxis, ...]  # 添加通道维度
                elif img_array.ndim == 3:  # RGB或RGBA
                    if img_array.shape[2] == 4:  # RGBA
                        img_array = img_array[..., :3]  # 去掉alpha通道
                    img_array = np.mean(img_array, axis=2, keepdims=True).transpose(2, 0, 1)  # 转换为灰度
                # 转换为张量并归一化
                img_tensor = torch.from_numpy(img_array).float()
                img_tensor = img_tensor / 255.0  # 归一化到[0,1]
                tensors.append(img_tensor)
            # 堆叠批次
            x = torch.stack(tensors)
        elif isinstance(images, torch.Tensor):
            x = images
            if x.dim() == 3:  # 单张图像
                if x.size(0) == 3 or x.size(0) == 4:  # RGB或RGBA
                    x = x[:3].mean(dim=0, keepdim=True)  # 转换为灰度
                x = x.unsqueeze(0)  # 添加批次维度
            if x.dim() == 4:  # 批次图像
                if x.size(1) == 3 or x.size(1) == 4:  # RGB或RGBA
                    x = x[:, :3].mean(dim=1, keepdim=True)  # 转换为灰度
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
        """
        if isinstance(x, dict):
            x = self._preprocess_images(x['images'])
        elif isinstance(x, list):
            x = self._preprocess_images(x)
        features = []
        multi_scale_features = {}
        
        # 骨干网络前向传播
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            features.append(x)
            # 存储P2, P3, P4的特征
            if i >= len(self.backbone) - 3:
                # 从backbone的倒数第三层开始，依次对应P2, P3, P4
                scale_idx = i - (len(self.backbone) - 3)  # 0, 1, 2 -> p2, p3, p4
                scale_name = f'p{scale_idx + 2}'
                multi_scale_features[scale_name] = x
        
        # 从最深层开始处理
        # 首先处理P4
        p4 = self.fpn['lateral_p4'](multi_scale_features['p4'])
        
        # 然后处理P3，加上上采样的P4
        p3 = self.fpn['lateral_p3'](multi_scale_features['p3'])
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        
        # 最后处理P2，加上上采样的P3
        p2 = self.fpn['lateral_p2'](multi_scale_features['p2'])
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        
        # 特征平滑
        p2 = self.fpn['smooth_p2'](p2)
        
        # 只使用P2尺度进行预测
        predictions = {
            'cls': self.cls_head(p2),
            'reg': self.reg_head(p2)
        }
        
        return {
            'predictions': predictions,
            'features': features
        }
        
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算模型损失。
        
        Args:
            predictions: 模型预测结果字典
                - predictions.cls: 分类预测 [B, num_classes, H, W]
                - predictions.reg: 回归预测 [B, 2, H, W]
            targets: 目标值字典
                - cls: 分类标签 [B, num_classes, H, W]
                - reg: 回归标签 [B, 2, H, W]
                - mask: 有效区域掩码 [B, 1, H, W]
                
        Returns:
            包含以下键的损失字典：
            - cls_loss: 分类损失
            - reg_loss: 回归损失
            - total_loss: 总损失
        """
        # 如果输入是多尺度格式，只取P2尺度的目标
        if 'multi_scale' in targets:
            p2_targets = targets['multi_scale'][0]  # P2是第一个尺度
            targets = {
                'cls': p2_targets['cls'],
                'reg': p2_targets['reg'],
                'mask': p2_targets['mask']
            }
        
        return self.loss_fn(predictions['predictions'], targets)

# 在文件末尾注册模型
from src.models.registry import model_registry
model_registry.register('spotgeo_res_fusion')(SpotGEOModelResFusion) 