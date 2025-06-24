#!/usr/bin/env python3
"""
测试backbone初始化脚本
"""

import torch
import yaml
from src.models.spotgeo_res_fusion import SpotGEOModelResFusion

def test_backbone_initialization():
    """测试不同配置下的backbone初始化"""
    
    # 测试配置1: ResNet backbone
    config1 = {
        'backbone_type': 'resnet',
        'res_blocks_per_layer': [2, 2, 2, 2],
        'backbone_channels': [64, 128, 256, 512],
        'detection_channels': [256, 128, 64],
        'num_classes': 1,
        'use_bn': True,
        'dropout': 0.1,
        'scale_factor': 0.5
    }
    
    # 测试配置2: 普通卷积backbone
    config2 = {
        'backbone_type': 'conv',
        'backbone_channels': [64, 128, 256, 512],
        'detection_channels': [256, 128, 64],
        'num_classes': 1,
        'use_bn': True,
        'dropout': 0.1,
        'scale_factor': 0.5
    }
    
    print("=" * 50)
    print("测试ResNet backbone初始化")
    print("=" * 50)
    
    try:
        model1 = SpotGEOModelResFusion(config1)
        print(f"✓ ResNet backbone初始化成功")
        print(f"  - Backbone层数: {len(model1.backbone)}")
        print(f"  - 预期层数: {sum(config1['res_blocks_per_layer'])}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 224, 224)
        output1 = model1(test_input)
        print(f"  - 前向传播成功，输出形状: {output1['predictions']['cls'].shape}")
        
    except Exception as e:
        print(f"✗ ResNet backbone初始化失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试普通卷积backbone初始化")
    print("=" * 50)
    
    try:
        model2 = SpotGEOModelResFusion(config2)
        print(f"✓ 普通卷积backbone初始化成功")
        print(f"  - Backbone层数: {len(model2.backbone)}")
        print(f"  - 预期层数: {len(config2['backbone_channels'])}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 224, 224)
        output2 = model2(test_input)
        print(f"  - 前向传播成功，输出形状: {output2['predictions']['cls'].shape}")
        
    except Exception as e:
        print(f"✗ 普通卷积backbone初始化失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试配置文件加载")
    print("=" * 50)
    
    try:
        # 加载配置文件
        with open('configs/singleframe_res_fusion_focal.yaml', 'r') as f:
            full_config = yaml.safe_load(f)
        
        model_config = full_config['model']['config']
        model3 = SpotGEOModelResFusion(model_config)
        print(f"✓ 配置文件加载成功")
        print(f"  - Backbone类型: {model_config.get('backbone_type')}")
        print(f"  - Backbone层数: {len(model3.backbone)}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 224, 224)
        output3 = model3(test_input)
        print(f"  - 前向传播成功，输出形状: {output3['predictions']['cls'].shape}")
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")

if __name__ == "__main__":
    test_backbone_initialization() 