import cv2
import numpy as np
import json
import os
from utils.visualizer import Visualizer


# 将标注文件转换为图片


def anno_to_pic(anno_dic, pic_path, img_width=800, img_height=600, object_color=(0, 0, 255)):
    """
    将标注数据转换为图片
    Args:
        anno_dic: 包含object_coords的字典
        pic_path: 输出图片路径
        img_width: 图片宽度，默认800
        img_height: 图片高度，默认600
        object_color: 对象颜色，默认红色(0,0,255)
    """
    object_coords = anno_dic["object_coords"]
    if len(object_coords) == 0:
        print(f"{anno_dic['sequence_id']}, {anno_dic['frame']},没有标注对象")
        return
    
    # 使用visualizer中的coords_to_img方法
    visualizer = Visualizer()
    img = visualizer.coords_to_img(object_coords, img_width, img_height, object_color)
    
    # 保存图片
    cv2.imwrite(pic_path, img)
    print(f"图片已保存到: {pic_path}")
    
    return img

def anno_to_contrast(gt_anno, pred_anno, original_img_path, output_path, 
                    gt_color=(0, 0, 255), pred_color=(255, 0, 0), 
                    point_radius=5, line_thickness=2):
    """
    将预测标注和GT标注都显示在原图上进行对比
    Args:
        gt_anno: GT标注数据字典，包含object_coords
        pred_anno: 预测标注数据字典，包含object_coords
        original_img_path: 原图路径
        output_path: 输出图片路径
        gt_color: GT标注颜色，默认红色(0,0,255)
        pred_color: 预测标注颜色，默认蓝色(255,0,0)
        point_radius: 点半径，默认5
        line_thickness: 线条粗细，默认2
    """
    # 读取原图
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"无法读取原图: {original_img_path}")
        return None
    
    # 创建对比图像（复制原图）
    contrast_img = original_img.copy()
    
    # 获取图像尺寸
    height, width = contrast_img.shape[:2]
    
    # 绘制GT标注点
    if gt_anno and "object_coords" in gt_anno:
        gt_coords = gt_anno["object_coords"]
        for coord in gt_coords:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < width and 0 <= y < height:
                # 绘制实心圆点
                cv2.circle(contrast_img, (x, y), point_radius, gt_color, -1)
                # 绘制外圈
                cv2.circle(contrast_img, (x, y), point_radius + 2, gt_color, line_thickness)
    
    # 绘制预测标注点
    if pred_anno and "object_coords" in pred_anno:
        pred_coords = pred_anno["object_coords"]
        for coord in pred_coords:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < width and 0 <= y < height:
                # 绘制实心圆点
                cv2.circle(contrast_img, (x, y), point_radius, pred_color, -1)
                # 绘制外圈
                cv2.circle(contrast_img, (x, y), point_radius + 2, pred_color, line_thickness)
    
    # 添加图例
    legend_y = 30
    # GT图例
    cv2.circle(contrast_img, (20, legend_y), point_radius, gt_color, -1)
    cv2.circle(contrast_img, (20, legend_y), point_radius + 2, gt_color, line_thickness)
    cv2.putText(contrast_img, "GT", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)
    
    # 预测图例
    legend_y += 30
    cv2.circle(contrast_img, (20, legend_y), point_radius, pred_color, -1)
    cv2.circle(contrast_img, (20, legend_y), point_radius + 2, pred_color, line_thickness)
    cv2.putText(contrast_img, "Pred", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
    
    # 添加统计信息
    gt_count = len(gt_anno.get("object_coords", [])) if gt_anno else 0
    pred_count = len(pred_anno.get("object_coords", [])) if pred_anno else 0
    
    info_text = f"GT: {gt_count}, Pred: {pred_count}"
    cv2.putText(contrast_img, info_text, (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存图片
    cv2.imwrite(output_path, contrast_img)
    print(f"对比图已保存到: {output_path}")
    
    return contrast_img

# 示例使用代码
if __name__ == "__main__":
    gt_path = "datasets/SpotGEOv2/test_anno.json"
    pred_path = "outputs/evaluation/my_evalv2/predictions.json"
    
    # 加载GT标注数据
    gt_anno_list = []
    with open(gt_path, 'r') as f:
        gt_anno_list = json.load(f)
    
    # 加载预测标注数据
    pred_anno_list = []
    with open(pred_path, 'r') as f:
        pred_anno_list = json.load(f)
    
    # 将预测数据转换为字典格式，便于查找
    pred_dict = {}
    for pred_anno in pred_anno_list:
        key = f"{pred_anno['sequence_id']}_{pred_anno['frame']}"
        pred_dict[key] = pred_anno
    
    # 确保输出目录存在
    output_dir = "outputs/test_anno_contrast"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个GT标注
    for gt_anno in gt_anno_list:
        sequence_id = gt_anno['sequence_id']
        frame = gt_anno['frame']
        
        # 查找对应的预测数据
        pred_key = f"{sequence_id}_{frame}"
        pred_anno = pred_dict.get(pred_key, None)
        
        # 构建原图路径
        original_img_path = f"datasets/SpotGEOv2/test/{sequence_id}/{frame}.png"
        
        # 检查原图是否存在
        if not os.path.exists(original_img_path):
            print(f"原图不存在: {original_img_path}")
            continue
        
        # 为每个sequence_id创建单独的目录
        sequence_dir = f"{output_dir}/{sequence_id}"
        os.makedirs(sequence_dir, exist_ok=True)
        
        # 生成对比图
        output_path = f"{sequence_dir}/{frame}_contrast.png"
        anno_to_contrast(
            gt_anno=gt_anno,
            pred_anno=pred_anno,
            original_img_path=original_img_path,
            output_path=output_path,
            gt_color=(0, 0, 255),      # 红色表示GT
            pred_color=(255, 0, 0),    # 蓝色表示预测
            point_radius=5,
            line_thickness=2
        )
        
        print(f"处理完成: sequence_id={sequence_id}, frame={frame}")
    
    print("所有对比图生成完成！")
    