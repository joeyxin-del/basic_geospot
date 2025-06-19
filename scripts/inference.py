import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

import torch
import numpy as np
from PIL import Image
import cv2

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.models import ModelFactory
from src.utils import get_logger
from src.postprocessing.heatmap_to_coords import heatmap_to_coords
from src.utils.visualizer import Visualizer

logger = get_logger('inference_script')

def main():
    """
    推理脚本主函数。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SpotGEO模型推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图像路径或目录路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出结果保存路径')
    parser.add_argument('--conf-thresh', type=float, default=0.1,
                      help='置信度阈值')
    parser.add_argument('--topk', type=int, default=100,
                      help='每帧最多输出目标数')
    parser.add_argument('--device', type=str, default='auto',
                      help='计算设备 (auto, cpu, cuda)')
    parser.add_argument('--model-name', type=str, default='spotgeo',
                      help='模型名称')
    parser.add_argument('--model-config', type=str, default=None,
                      help='模型配置文件路径')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    if args.output is None:
        output_dir = os.path.join('outputs', 'inference')
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载检查点
    logger.info(f"加载检查点: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    
    # 优先使用权重文件中的模型配置
    if 'config' in checkpoint_data:
        logger.info("从权重文件中读取模型配置")
        model_config = checkpoint_data['config']
    elif args.model_config and os.path.exists(args.model_config):
        logger.info("从配置文件读取模型配置")
        import yaml
        with open(args.model_config, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
    else:
        logger.info("使用默认模型配置")
        model_config = {}
    
    # 创建模型
    logger.info(f"创建模型: {args.model_name}")
    model = ModelFactory.create(
        name=args.model_name,
        config=model_config
    )
    
    # 加载权重
    state_dict = checkpoint_data['model_state_dict']
    
    # 过滤掉不需要的键（如total_ops, total_params等统计信息）
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not any(skip_key in key for skip_key in ['total_ops', 'total_params']):
            filtered_state_dict[key] = value
    
    logger.info(f"加载 {len(filtered_state_dict)} 个权重参数")
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像推理
        results = inference_single_image(
            model, input_path, device, args.conf_thresh, args.topk
        )
        save_results(results, output_dir, input_path.stem)
    elif input_path.is_dir():
        # 目录批量推理
        results = inference_directory(
            model, input_path, device, args.conf_thresh, args.topk
        )
        save_results(results, output_dir, "batch_inference")
    else:
        raise ValueError(f"输入路径不存在: {args.input}")
    
    logger.info(f"推理完成，结果保存在: {output_dir}")

def inference_single_image(model: torch.nn.Module, 
                          image_path: Path, 
                          device: torch.device,
                          conf_thresh: float = 0.5,
                          topk: int = 100) -> Dict[str, Any]:
    """
    对单张图像进行推理。
    
    Args:
        model: 加载好权重的模型
        image_path: 图像文件路径
        device: 计算设备
        conf_thresh: 置信度阈值
        topk: 最多输出目标数
        
    Returns:
        推理结果字典
    """
    import json  # 在函数开头导入json模块
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    logger.info(f"处理图像: {image_path}")
    
    # 预处理图像
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 后处理获取坐标
    cls_pred = outputs['predictions']['cls']
    reg_pred = outputs['predictions']['reg']
    
    # 计算缩放因子
    out_h, out_w = cls_pred.shape[-2:]
    scale_x = 640 / out_w  # 假设原图宽度为640
    scale_y = 480 / out_h  # 假设原图高度为480
    scale = (scale_x + scale_y) / 2
    
    coords_list = heatmap_to_coords(
        cls_pred=cls_pred.detach().cpu(),
        reg_pred=reg_pred.detach().cpu(),
        conf_thresh=conf_thresh,
        topk=topk,
        scale=scale
    )
    
    # 尝试加载GT数据
    gt_coords = None
    
    # 方法1: 尝试从同名的JSON文件加载
    gt_path = image_path.with_suffix('.json')
    if gt_path.exists():
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                if 'object_coords' in gt_data:
                    gt_coords = gt_data['object_coords']
                    logger.info(f"从同名JSON文件加载GT数据: {len(gt_coords)} 个目标")
        except Exception as e:
            logger.warning(f"从同名JSON文件加载GT数据失败: {str(e)}")
    
    # 方法2: 如果同名文件不存在，尝试从全局标注文件加载
    if gt_coords is None:
        # 从图像路径推断序列ID和帧号
        # 假设路径格式: datasets/SpotGEOv2/test/sequence_id/frame.png
        path_parts = image_path.parts
        try:
            # 查找序列ID和帧号
            sequence_id = None
            frame_idx = None
            
            for i, part in enumerate(path_parts):
                if part.isdigit() and i < len(path_parts) - 1:
                    sequence_id = int(part)
                    # 下一个部分应该是帧号
                    if i + 1 < len(path_parts):
                        frame_name = path_parts[i + 1]
                        if frame_name.endswith('.png'):
                            frame_idx = int(frame_name.replace('.png', ''))
                            break
            
            if sequence_id is not None and frame_idx is not None:
                # 尝试从全局标注文件加载
                global_anno_paths = [
                    image_path.parent.parent.parent / 'test_anno.json',  # datasets/SpotGEOv2/test_anno.json
                    image_path.parent.parent.parent / 'train_anno.json',  # datasets/SpotGEOv2/train_anno.json
                    image_path.parent.parent / 'test_anno.json',          # 其他可能的路径
                    image_path.parent.parent / 'train_anno.json'
                ]
                
                for anno_path in global_anno_paths:
                    if anno_path.exists():
                        try:
                            with open(anno_path, 'r', encoding='utf-8') as f:
                                all_annotations = json.load(f)
                            
                            # 查找对应的标注
                            for anno in all_annotations:
                                if (anno.get('sequence_id') == sequence_id and 
                                    anno.get('frame') == frame_idx):
                                    gt_coords = anno.get('object_coords', [])
                                    logger.info(f"从全局标注文件 {anno_path} 加载GT数据: 序列{sequence_id}, 帧{frame_idx}, {len(gt_coords)} 个目标")
                                    break
                            
                            if gt_coords is not None:
                                break
                                
                        except Exception as e:
                            logger.warning(f"从全局标注文件 {anno_path} 加载GT数据失败: {str(e)}")
                            
        except Exception as e:
            logger.warning(f"解析图像路径获取序列信息失败: {str(e)}")
    
    # 处理GT坐标格式（如果是边界框格式，转换为中心点）
    if gt_coords is not None:
        processed_gt_coords = []
        for coord in gt_coords:
            if len(coord) == 4:  # 边界框格式 [x1, y1, x2, y2]
                # 转换为中心点
                center_x = (coord[0] + coord[2]) / 2
                center_y = (coord[1] + coord[3]) / 2
                processed_gt_coords.append([center_x, center_y])
            elif len(coord) == 2:  # 已经是点格式 [x, y]
                processed_gt_coords.append(coord)
            else:
                logger.warning(f"未知的GT坐标格式: {coord}")
        
        gt_coords = processed_gt_coords
    
    # 整理结果
    result = {
        'image_path': str(image_path),
        'predictions': coords_list[0],  # 单张图像只有一个结果
        'model_outputs': {
            'cls_shape': list(cls_pred.shape),
            'reg_shape': list(reg_pred.shape)
        },
        'conf_thresh': conf_thresh,
        'topk': topk
    }
    
    # 如果有GT数据，添加到结果中
    if gt_coords is not None:
        result['gt_coords'] = gt_coords
    
    return result

def inference_directory(model: torch.nn.Module, 
                       dir_path: Path, 
                       device: torch.device,
                       conf_thresh: float = 0.5,
                       topk: int = 100) -> Dict[str, Any]:
    """
    对目录中的所有图像进行批量推理。
    
    Args:
        model: 加载好权重的模型
        dir_path: 图像目录路径
        device: 计算设备
        conf_thresh: 置信度阈值
        topk: 最多输出目标数
        
    Returns:
        批量推理结果字典
    """
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = [
        f for f in dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"目录中没有找到图像文件: {dir_path}")
    
    logger.info(f"找到 {len(image_files)} 张图像在 {dir_path}")
    
    results = {
        'directory': str(dir_path),
        'total_images': len(image_files),
        'predictions': []
    }
    
    # 逐个处理图像
    for image_file in image_files:
        try:
            result = inference_single_image(
                model, image_file, device, conf_thresh, topk
            )
            results['predictions'].append(result)
        except Exception as e:
            logger.error(f"处理图像 {image_file} 时出错: {str(e)}")
            results['predictions'].append({
                'image_path': str(image_file),
                'error': str(e),
                'predictions': None
            })
    
    return results

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    预处理图像。
    
    Args:
        image: PIL图像
        
    Returns:
        预处理后的张量
    """
    # 转换为numpy数组
    image_array = np.array(image)
    
    # 转换为张量并归一化
    image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1) / 255.0
    
    return image_tensor

def save_results(results: Dict[str, Any], output_dir: str, filename: str):
    """
    保存推理结果。
    
    Args:
        results: 推理结果
        output_dir: 输出目录
        filename: 文件名
    """
    # 保存JSON结果
    json_path = os.path.join(output_dir, f"{filename}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"推理结果已保存到: {json_path}")
    
    # 如果是单张图像，还可以保存可视化结果
    if 'image_path' in results and 'predictions' in results:
        save_visualization(results, output_dir, filename)

def save_visualization(results: Dict[str, Any], output_dir: str, filename: str):
    """
    保存可视化结果。
    
    Args:
        results: 推理结果
        output_dir: 输出目录
        filename: 文件名
    """
    try:
        # 创建该图像的专用文件夹
        image_output_dir = os.path.join(output_dir, filename)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # 加载原图
        image_path = results['image_path']
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.warning(f"无法读取图像: {image_path}")
            return
            
        # 获取图像尺寸
        height, width = original_image.shape[:2]
        
        # 1. 保存原图
        original_path = os.path.join(image_output_dir, "01_original.jpg")
        cv2.imwrite(original_path, original_image)
        logger.info(f"原图已保存到: {original_path}")
        
        # 2. 创建预测结果图（使用coords_to_img函数）
        predictions = results['predictions']
        if predictions and 'object_coords' in predictions:
            pred_coords = predictions['object_coords']
            prediction_image = Visualizer.coords_to_img(pred_coords, width, height, color=(255, 0, 0))  # 蓝色
        else:
            prediction_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        prediction_path = os.path.join(image_output_dir, "02_prediction.jpg")
        cv2.imwrite(prediction_path, prediction_image)
        logger.info(f"预测结果图已保存到: {prediction_path}")
        
        # 3. 创建GT图（使用coords_to_img函数）
        if 'gt_coords' in results:
            gt_coords = results['gt_coords']
            gt_image = Visualizer.coords_to_img(gt_coords, width, height, color=(0, 0, 255))  # 红色
        else:
            gt_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        gt_path = os.path.join(image_output_dir, "03_ground_truth.jpg")
        cv2.imwrite(gt_path, gt_image)
        logger.info(f"GT图已保存到: {gt_path}")
        
        # 4. 创建对比图（原图上的预测和GT对比）
        comparison_image = original_image.copy()
        
        # 绘制GT点（白色）
        if 'gt_coords' in results:
            gt_coords = results['gt_coords']
            for coord in gt_coords:
                x, y = int(coord[0]), int(coord[1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(comparison_image, (x, y), 5, (0, 0, 255), -1)  # 白色圆点
                    # cv2.circle(comparison_image, (x, y), 7, (255, 255, 255), 2)   # 白色圆圈
        
        # 绘制预测点（蓝色）
        if predictions and 'object_coords' in predictions:
            coords = predictions['object_coords']
            for coord in coords:
                x, y = int(coord[0]), int(coord[1])
                if 0 <= x < width and 0 <= y < height:
                    # cv2.circle(comparison_image, (x, y), 5, (255, 0, 0), -1)      # 蓝色圆点
                    cv2.rectangle(comparison_image, (x-5, y-5), (x+5, y+5), (255, 0, 0), 2)
                    # cv2.circle(comparison_image, (x, y), 7, (255, 0, 0), 2)       # 蓝色圆圈
        
        comparison_path = os.path.join(image_output_dir, "04_comparison.jpg")
        cv2.imwrite(comparison_path, comparison_image)
        logger.info(f"对比图已保存到: {comparison_path}")
        
        # 5. 保存结果信息
        info_path = os.path.join(image_output_dir, "info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"图像路径: {image_path}\n")
            f.write(f"图像尺寸: {width} x {height}\n")
            f.write(f"预测目标数量: {len(predictions.get('object_coords', [])) if predictions else 0}\n")
            if 'gt_coords' in results:
                f.write(f"真实目标数量: {len(results['gt_coords'])}\n")
            f.write(f"置信度阈值: {results.get('conf_thresh', 'N/A')}\n")
            f.write(f"TopK: {results.get('topk', 'N/A')}\n")
        
        logger.info(f"可视化结果已保存到文件夹: {image_output_dir}")
        
    except Exception as e:
        logger.warning(f"保存可视化结果时出错: {str(e)}")

if __name__ == "__main__":
    main() 