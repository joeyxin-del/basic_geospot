#!/usr/bin/env python3
"""
测试visualizer的plot_single_metric方法。
"""

import os
import sys

# 添加src目录到路径
sys.path.append('src')

from src.utils.visualizer import Visualizer

def test_single_metric_plot():
    """测试单个指标绘图功能。"""
    
    # 创建保存目录
    save_dir = os.path.join('tests', 'visualize_single_metric')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"图表保存目录: {save_dir}")
    
    # 初始化可视化器
    visualizer = Visualizer(save_dir=save_dir)
    
    # 模拟训练指标数据
    test_data = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.35, 0.3, 0.28],
        'val_f1': [0.3, 0.5, 0.7, 0.75, 0.8, 0.82],
        'val_mse': [15.2, 12.8, 10.5, 8.9, 7.3, 6.8],
        'lr': [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
    }
    epochs = [1, 2, 3, 4, 5, 6]
    
    print("正在生成训练指标图表...")
    
    # 为每个指标生成图表
    for metric_name, values in test_data.items():
        save_path = os.path.join(save_dir, f'{metric_name}_test.png')
        visualizer.plot_single_metric(metric_name, values, epochs, save_path)
    
    # 检查生成的文件
    generated_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
    
    print(f"\n✅ 成功生成 {len(generated_files)} 张图表:")
    for file in generated_files:
        print(f"  📊 {file}")
        
    print(f"\n💡 图表特征:")
    print("  - 横轴: Epoch")
    print("  - 纵轴: 对应指标值")
    print("  - 标注: 显示最大值和最小值")
    print("  - 格式: 清晰的曲线图")
    print(f"\n📁 保存位置: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    test_single_metric_plot() 