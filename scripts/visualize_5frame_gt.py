#!/usr/bin/env python3
"""
将5帧的GT目标都画在第一帧的原图上
使用不同颜色区分不同帧的目标，便于观察目标在序列中的运动轨迹
"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb


def load_annotations(annotation_path: str) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    加载标注文件，按序列ID和帧号组织
    
    Args:
        annotation_path: 标注文件路径
        
    Returns:
        按序列ID和帧号组织的标注字典: {sequence_id: {frame: annotation}}
    """
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 按序列ID和帧号组织
    organized_annotations = {}
    for anno in annotations:
        seq_id = anno['sequence_id']
        frame = anno['frame']
        
        if seq_id not in organized_annotations:
            organized_annotations[seq_id] = {}
        
        organized_annotations[seq_id][frame] = anno
    
    return organized_annotations


def get_frame_colors(num_frames: int = 5) -> List[Tuple[int, int, int]]:
    """
    为不同帧生成不同的颜色
    
    Args:
        num_frames: 帧数
        
    Returns:
        颜色列表，每个元素为BGR格式的元组
    """
    colors = []
    for i in range(num_frames):
        # 使用HSV色彩空间生成均匀分布的颜色
        hue = i / num_frames
        saturation = 0.8
        value = 0.9
        
        # 转换为RGB
        rgb = hsv_to_rgb([hue, saturation, value])[0]
        
        # 转换为BGR (OpenCV格式)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    return colors


def draw_gt_on_first_frame(
    sequence_id: int,
    first_frame_path: str,
    annotations: Dict[int, Dict[str, Any]],
    output_path: str,
    num_frames: int = 5,
    point_radius: int = 8,
    line_thickness: int = 3,
    show_frame_numbers: bool = True,
    show_trajectory: bool = True
) -> bool:
    """
    将5帧的GT目标都画在第一帧的原图上
    
    Args:
        sequence_id: 序列ID
        first_frame_path: 第一帧图像路径
        annotations: 该序列的标注数据
        output_path: 输出图像路径
        num_frames: 要绘制的帧数
        point_radius: 点半径
        line_thickness: 线条粗细
        show_frame_numbers: 是否显示帧号
        show_trajectory: 是否显示轨迹线
        
    Returns:
        是否成功生成图像
    """
    # 读取第一帧图像
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"无法读取第一帧图像: {first_frame_path}")
        return False
    
    # 创建输出图像（复制第一帧）
    output_image = first_frame.copy()
    height, width = output_image.shape[:2]
    
    # 获取颜色列表
    colors = get_frame_colors(num_frames)
    
    # 存储所有帧的目标坐标，用于绘制轨迹
    all_coords = []
    
    # 绘制每一帧的目标
    for frame_idx in range(1, num_frames + 1):  # 帧号从1开始
        if frame_idx not in annotations:
            print(f"警告: 序列{sequence_id}的帧{frame_idx}没有标注数据")
            continue
        
        annotation = annotations[frame_idx]
        coords = annotation.get('object_coords', [])
        color = colors[frame_idx - 1]  # 颜色索引从0开始
        
        # 存储当前帧的坐标
        frame_coords = []
        
        for obj_idx, coord in enumerate(coords):
            if len(coord) == 4:  # 边界框格式 [x1, y1, x2, y2]
                # 转换为中心点
                center_x = int((coord[0] + coord[2]) / 2)
                center_y = int((coord[1] + coord[3]) / 2)
            elif len(coord) == 2:  # 点格式 [x, y]
                center_x = int(coord[0])
                center_y = int(coord[1])
            else:
                print(f"警告: 未知的坐标格式: {coord}")
                continue
            
            # 确保坐标在图像范围内
            if 0 <= center_x < width and 0 <= center_y < height:
                # 绘制目标点
                cv2.circle(output_image, (center_x, center_y), point_radius, color, -1)
                cv2.circle(output_image, (center_x, center_y), point_radius + 2, color, line_thickness)
                
                # 存储坐标
                frame_coords.append((center_x, center_y))
                
                # 显示帧号
                if show_frame_numbers:
                    text = f"F{frame_idx}"
                    if obj_idx > 0:
                        text += f"-{obj_idx + 1}"
                    
                    # 计算文本位置
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = center_x + point_radius + 5
                    text_y = center_y + text_size[1] // 2
                    
                    # 确保文本不超出图像边界
                    if text_x + text_size[0] > width:
                        text_x = center_x - point_radius - text_size[0] - 5
                    if text_y < text_size[1]:
                        text_y = center_y + point_radius + text_size[1] + 5
                    
                    # 绘制文本背景
                    cv2.rectangle(output_image, 
                                (text_x - 2, text_y - text_size[1] - 2),
                                (text_x + text_size[0] + 2, text_y + 2),
                                color, -1)
                    
                    # 绘制文本
                    cv2.putText(output_image, text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        all_coords.append(frame_coords)
    
    # 绘制轨迹线
    if show_trajectory and len(all_coords) > 1:
        for obj_idx in range(max(len(coords) for coords in all_coords)):
            trajectory_points = []
            
            # 收集每个目标在所有帧中的坐标
            for frame_coords in all_coords:
                if obj_idx < len(frame_coords):
                    trajectory_points.append(frame_coords[obj_idx])
            
            # 绘制轨迹线
            if len(trajectory_points) > 1:
                for i in range(len(trajectory_points) - 1):
                    pt1 = trajectory_points[i]
                    pt2 = trajectory_points[i + 1]
                    
                    # 使用渐变色绘制轨迹线
                    color1 = colors[i]
                    color2 = colors[i + 1]
                    
                    # 绘制线段
                    cv2.line(output_image, pt1, pt2, color1, line_thickness - 1)
    
    # 添加图例
    legend_y = 30
    for frame_idx in range(1, num_frames + 1):
        if frame_idx in annotations:
            color = colors[frame_idx - 1]
            # 绘制图例点
            cv2.circle(output_image, (20, legend_y), point_radius, color, -1)
            cv2.circle(output_image, (20, legend_y), point_radius + 2, color, line_thickness)
            
            # 绘制图例文本
            legend_text = f"Frame {frame_idx}"
            cv2.putText(output_image, legend_text, (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            legend_y += 30
    
    # 添加统计信息
    total_objects = sum(len(annotations.get(frame, {}).get('object_coords', [])) 
                       for frame in range(1, num_frames + 1) if frame in annotations)
    
    info_text = f"Sequence {sequence_id}: {total_objects} objects across {num_frames} frames"
    cv2.putText(output_image, info_text, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存图像
    cv2.imwrite(output_path, output_image)
    print(f"已生成可视化图像: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="将5帧的GT目标都画在第一帧的原图上")
    parser.add_argument("--annotation_path", type=str, 
                       default="datasets/SpotGEOv2/test_anno.json",
                       help="标注文件路径")
    parser.add_argument("--data_root", type=str,
                       default="datasets/SpotGEOv2/test",
                       help="数据根目录")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/gt_visualization",
                       help="输出目录")
    parser.add_argument("--sequence_ids", type=str, nargs="+",
                       help="要处理的序列ID列表，不指定则处理所有序列")
    parser.add_argument("--num_frames", type=int, default=5,
                       help="要绘制的帧数")
    parser.add_argument("--point_radius", type=int, default=8,
                       help="目标点半径")
    parser.add_argument("--line_thickness", type=int, default=3,
                       help="线条粗细")
    parser.add_argument("--no_frame_numbers", action="store_true",
                       help="不显示帧号")
    parser.add_argument("--no_trajectory", action="store_true",
                       help="不显示轨迹线")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载标注数据
    print(f"加载标注文件: {args.annotation_path}")
    annotations = load_annotations(args.annotation_path)
    
    # 确定要处理的序列
    if args.sequence_ids:
        sequence_ids = [int(sid) for sid in args.sequence_ids]
    else:
        sequence_ids = list(annotations.keys())
    
    print(f"将处理 {len(sequence_ids)} 个序列")
    
    # 处理每个序列
    success_count = 0
    for sequence_id in sequence_ids:
        if sequence_id not in annotations:
            print(f"警告: 序列{sequence_id}在标注文件中不存在")
            continue
        
        # 构建第一帧图像路径
        first_frame_path = os.path.join(args.data_root, str(sequence_id), "1.png")
        
        if not os.path.exists(first_frame_path):
            print(f"警告: 第一帧图像不存在: {first_frame_path}")
            continue
        
        # 构建输出路径
        output_path = output_dir / f"sequence_{sequence_id}_5frame_gt.png"
        
        # 绘制GT目标
        success = draw_gt_on_first_frame(
            sequence_id=sequence_id,
            first_frame_path=first_frame_path,
            annotations=annotations[sequence_id],
            output_path=str(output_path),
            num_frames=args.num_frames,
            point_radius=args.point_radius,
            line_thickness=args.line_thickness,
            show_frame_numbers=not args.no_frame_numbers,
            show_trajectory=not args.no_trajectory
        )
        
        if success:
            success_count += 1
    
    print(f"\n处理完成! 成功生成 {success_count}/{len(sequence_ids)} 个可视化图像")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main() 