#!/usr/bin/env python3
"""
快速测试脚本：将5帧的GT目标都画在第一帧的原图上
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path


def visualize_5frame_gt(sequence_id=5016, num_frames=5):
    """
    快速可视化指定序列的5帧GT目标
    
    Args:
        sequence_id: 序列ID
        num_frames: 帧数
    """
    # 文件路径
    annotation_path = "datasets/SpotGEOv2/test_anno.json"
    data_root = "datasets/SpotGEOv2/test"
    output_dir = "outputs/gt_visualization"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载标注数据
    print(f"加载标注文件: {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)
    
    # 筛选指定序列的标注
    sequence_annotations = {}
    for anno in all_annotations:
        if anno['sequence_id'] == sequence_id:
            frame = anno['frame']
            sequence_annotations[frame] = anno
    
    print(f"找到序列{sequence_id}的标注: {len(sequence_annotations)}帧")
    
    # 读取第一帧图像
    first_frame_path = os.path.join(data_root, str(sequence_id), "1.png")
    if not os.path.exists(first_frame_path):
        print(f"第一帧图像不存在: {first_frame_path}")
        return
    
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"无法读取第一帧图像: {first_frame_path}")
        return
    
    # 创建输出图像
    output_image = first_frame.copy()
    height, width = output_image.shape[:2]
    
    # 定义颜色（BGR格式）
    colors = [
        (0, 0, 255),    # 红色 - 第1帧
        (0, 255, 0),    # 绿色 - 第2帧
        (255, 0, 0),    # 蓝色 - 第3帧
        (0, 255, 255),  # 黄色 - 第4帧
        (255, 0, 255),  # 紫色 - 第5帧
    ]
    
    # 存储所有帧的坐标用于绘制轨迹
    all_coords = []
    
    # 绘制每一帧的目标
    for frame_idx in range(1, num_frames + 1):
        if frame_idx not in sequence_annotations:
            print(f"警告: 帧{frame_idx}没有标注数据")
            continue
        
        annotation = sequence_annotations[frame_idx]
        coords = annotation.get('object_coords', [])
        color = colors[frame_idx - 1]
        
        print(f"帧{frame_idx}: {len(coords)}个目标")
        
        # 存储当前帧的坐标
        frame_coords = []
        
        for obj_idx, coord in enumerate(coords):
            # 处理坐标格式
            if len(coord) == 4:  # 边界框格式 [x1, y1, x2, y2]
                center_x = int((coord[0] + coord[2]) / 2)
                center_y = int((coord[1] + coord[3]) / 2)
            elif len(coord) == 2:  # 点格式 [x, y]
                center_x = int(coord[0])
                center_y = int(coord[1])
            else:
                print(f"警告: 未知坐标格式: {coord}")
                continue
            
            # 确保坐标在图像范围内
            if 0 <= center_x < width and 0 <= center_y < height:
                # 绘制目标点
                cv2.circle(output_image, (center_x, center_y), 8, color, -1)
                # cv2.circle(output_image, (center_x, center_y), 10, color, 2)
                
                # 添加帧号标签
                text = f"F{frame_idx}"
                if obj_idx > 0:
                    text += f"-{obj_idx + 1}"
                
                # 计算文本位置
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = center_x + 15
                text_y = center_y + 5
                
                # # 绘制文本背景
                # cv2.rectangle(output_image, 
                #             (text_x - 2, text_y - text_size[1] - 2),
                #             (text_x + text_size[0] + 2, text_y + 2),
                #             color, -1)
                
                # # 绘制文本
                # cv2.putText(output_image, text, (text_x, text_y),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                frame_coords.append((center_x, center_y))
        
        all_coords.append(frame_coords)
    
    # 绘制轨迹线
    if len(all_coords) > 1:
        for obj_idx in range(max(len(coords) for coords in all_coords)):
            trajectory_points = []
            
            # 收集每个目标在所有帧中的坐标
            for frame_coords in all_coords:
                if obj_idx < len(frame_coords):
                    trajectory_points.append(frame_coords[obj_idx])
            
            # 绘制轨迹线
            # if len(trajectory_points) > 1:
            #     for i in range(len(trajectory_points) - 1):
            #         pt1 = trajectory_points[i]
            #         pt2 = trajectory_points[i + 1]
            #         cv2.line(output_image, pt1, pt2, colors[i], 2)
    
    # 添加图例
    legend_y = 30
    for frame_idx in range(1, num_frames + 1):
        if frame_idx in sequence_annotations:
            color = colors[frame_idx - 1]
            # 绘制图例点
            cv2.circle(output_image, (20, legend_y), 8, color, -1)
            cv2.circle(output_image, (20, legend_y), 10, color, 2)
            
            # 绘制图例文本
            legend_text = f"Frame {frame_idx}"
            cv2.putText(output_image, legend_text, (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            legend_y += 30
    
    # 添加统计信息
    total_objects = sum(len(sequence_annotations.get(frame, {}).get('object_coords', [])) 
                       for frame in range(1, num_frames + 1) if frame in sequence_annotations)
    
    info_text = f"Sequence {sequence_id}: {total_objects} objects across {num_frames} frames"
    cv2.putText(output_image, info_text, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存图像
    output_path = os.path.join(output_dir, f"sequence_{sequence_id}_5frame_gt.png")
    cv2.imwrite(output_path, output_image)
    print(f"可视化图像已保存: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # 测试序列5016
    visualize_5frame_gt(sequence_id=3, num_frames=5) 