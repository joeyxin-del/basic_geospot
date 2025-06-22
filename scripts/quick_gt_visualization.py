#!/usr/bin/env python3
"""
快速可视化工具：将5帧序列中的目标标注在第一帧原图上
每一帧的目标用不同的颜色表示
输出效果类似于 outputs/gt_visualization/sequence_3_5frame_gt.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='快速可视化5帧序列中的目标标注')
    parser.add_argument('--sequence_dir', type=str, required=True,
                       help='序列图像目录路径')
    parser.add_argument('--anno_path', type=str, required=True,
                       help='标注文件路径')
    parser.add_argument('--sequence_name', type=str, required=True,
                       help='序列名称')
    parser.add_argument('--output_dir', type=str, default='outputs/gt_visualization',
                       help='输出目录路径')
    return parser.parse_args()

def draw_point(draw, x, y, color, radius=3):
    """在图像上画点"""
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取标注文件
    with open(args.anno_path, 'r') as f:
        annotations = json.load(f)
    
    # 获取指定序列的标注
    sequence_id = int(args.sequence_name)
    sequence_annos = []
    for anno in annotations:
        if anno['sequence_id'] == sequence_id:
            sequence_annos.append(anno)
    
    if not sequence_annos:
        raise ValueError(f"找不到序列 {sequence_id} 的标注")
    
    # 按帧号排序
    sequence_annos.sort(key=lambda x: x['frame'])
    
    # 获取第一帧图像
    frame_0_path = os.path.join(args.sequence_dir, str(sequence_id), "1.png")
    if not os.path.exists(frame_0_path):
        raise FileNotFoundError(f"找不到第一帧图像: {frame_0_path}")
    
    # 打开第一帧图像
    image = Image.open(frame_0_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 定义每一帧的颜色
    colors = [
        'red',      # 第1帧
        'green',    # 第2帧
        'blue',     # 第3帧
        'yellow',   # 第4帧
        'magenta'   # 第5帧
    ]
    
    # 在图像上标注每一帧的目标点
    for frame_anno, color in zip(sequence_annos, colors):
        for coord in frame_anno['object_coords']:
            x, y = coord
            draw_point(draw, x, y, color)
    
    # 添加图例
    legend_text = [
        "Frame 1: Red",
        "Frame 2: Green", 
        "Frame 3: Blue",
        "Frame 4: Yellow",
        "Frame 5: Magenta"
    ]
    
    # 使用等宽字体
    try:
        font = ImageFont.truetype("consolas", 20)
    except:
        font = ImageFont.load_default()
    
    # 在左上角添加图例
    x, y = 10, 10
    for text, color in zip(legend_text, colors):
        # 画点
        draw_point(draw, x + 5, y + 10, color)
        # 写文字
        draw.text((x + 15, y), text, fill=color, font=font)
        y += 25
    
    # 在底部添加序列信息
    total_objects = sum(len(anno['object_coords']) for anno in sequence_annos)
    info_text = f"Sequence {sequence_id} | Total Objects: {total_objects}"
    draw.text((10, image.height - 30), info_text, fill='white', font=font)
    
    # 保存图像
    output_path = os.path.join(args.output_dir, f"sequence_{sequence_id}_5frame_gt.png")
    image.save(output_path)
    print(f"已保存可视化结果到: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"处理序列时出错: {str(e)}")
        sys.exit(1) 