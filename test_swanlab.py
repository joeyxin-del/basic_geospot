#!/usr/bin/env python3
"""
SwanLab集成测试脚本
演示如何在训练过程中使用SwanLab记录详细的训练曲线
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.trainer_singleframe import TrainerSingleFrame
from src.models.spotgeo_res import SpotGeoRes
from src.datasets.spotgeov2_singleframe import SpotGeoV2SingleFrame
from src.transforms.factory import create_transform

def test_swanlab_integration():
    """
    测试SwanLab集成功能
    """
    print("开始测试SwanLab集成...")
    
    # 检查SwanLab是否可用
    try:
        import swanlab
        print("✓ SwanLab已安装")
    except ImportError:
        print("✗ SwanLab未安装，请运行: pip install swanlab")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建简单的数据集（使用测试数据）
    test_data_dir = "tests/test_data/spotgeov2"
    if not os.path.exists(test_data_dir):
        print(f"测试数据目录不存在: {test_data_dir}")
        return
    
    # 创建数据变换
    transform = create_transform(
        input_size=(640, 480),
        augmentations=['flip', 'rotate', 'color_jitter'],
        normalize=True
    )
    
    # 创建数据集
    try:
        train_dataset = SpotGeoV2SingleFrame(
            data_dir=test_data_dir,
            split='train',
            transform=transform
        )
        val_dataset = SpotGeoV2SingleFrame(
            data_dir=test_data_dir,
            split='val',
            transform=transform
        )
        print(f"✓ 数据集创建成功 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn
    )
    
    # 创建模型
    try:
        model = SpotGeoRes(
            backbone='resnet18',
            pretrained=True,
            num_classes=1
        )
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    # 创建优化器和调度器
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 创建训练器（启用SwanLab）
    trainer = TrainerSingleFrame(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir='outputs/swanlab_test',
        experiment_name='swanlab_integration_test',
        use_swanlab=True,  # 启用SwanLab
        swanlab_project='spotgeo-singleframe-test',
        swanlab_mode='offline',  # 使用离线模式进行测试
        max_epochs=3,  # 只训练3个epoch进行测试
        eval_interval=1,
        checkpoint_interval=1,
        early_stopping_patience=5,
        log_batch_metrics=True,  # 记录每个batch的指标
        log_gradients=True,      # 记录梯度信息
        conf_thresh=0.5,
        topk=100
    )
    
    print("✓ 训练器创建成功")
    print("开始训练...")
    
    # 开始训练
    try:
        results = trainer.train()
        print("✓ 训练完成")
        print(f"最佳epoch: {results['best_epoch']}")
        print(f"最佳分数: {results['best_score']:.4f}")
        print(f"总训练轮数: {results['total_epochs']}")
        
        # 检查SwanLab日志文件
        swanlab_dir = os.path.join('outputs/swanlab_test', trainer.experiment_name)
        if os.path.exists(swanlab_dir):
            print(f"✓ SwanLab日志保存在: {swanlab_dir}")
            
            # 列出生成的文件
            for root, dirs, files in os.walk(swanlab_dir):
                for file in files:
                    if file.endswith('.json') or file.endswith('.csv'):
                        print(f"  - {os.path.join(root, file)}")
        else:
            print("✗ SwanLab日志目录未找到")
            
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()

def show_swanlab_usage():
    """
    显示SwanLab使用说明
    """
    print("\n" + "="*60)
    print("SwanLab使用说明")
    print("="*60)
    
    print("\n1. 安装SwanLab:")
    print("   pip install swanlab")
    
    print("\n2. 基本使用:")
    print("   # 在训练器中启用SwanLab")
    print("   trainer = TrainerSingleFrame(")
    print("       use_swanlab=True,")
    print("       swanlab_project='your-project-name',")
    print("       swanlab_mode='cloud',  # 或 'offline'")
    print("       log_batch_metrics=True,")
    print("       log_gradients=True")
    print("   )")
    
    print("\n3. 记录的指标包括:")
    print("   - 训练损失 (每个batch和epoch)")
    print("   - 验证损失 (每个batch和epoch)")
    print("   - 学习率变化")
    print("   - 梯度范数")
    print("   - 评估指标 (F1, MSE等)")
    print("   - 训练时间")
    print("   - 最佳模型信息")
    
    print("\n4. 查看结果:")
    print("   # 在线模式: 访问 https://swanlab.ai")
    print("   # 离线模式: 查看outputs目录下的日志文件")
    
    print("\n5. 高级功能:")
    print("   - 自定义指标记录")
    print("   - 实验对比")
    print("   - 模型版本管理")
    print("   - 团队协作")

if __name__ == "__main__":
    print("SwanLab集成测试")
    print("="*40)
    
    # 显示使用说明
    show_swanlab_usage()
    
    # 运行测试
    test_swanlab_integration()
    
    print("\n测试完成！") 