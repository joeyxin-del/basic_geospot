#!/usr/bin/env python3
"""
单帧评估脚本
使用SpotGEOv2_SingleFrame数据集进行单帧评估
支持yaml配置文件和命令行参数
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.spotgeov2_singleframe import SpotGEOv2_SingleFrame
from src.models import get_model
from src.utils import get_logger
from src.utils.evaluator import Evaluator
from src.postprocessing.heatmap_to_coords import heatmap_to_coords

logger = get_logger('eval_singleframe')

def load_config(config_path: str) -> Dict[str, Any]:
    """加载yaml配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置，override_config会覆盖base_config中的对应项"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单帧评估脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (yaml格式)')
    
    # 模型相关参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称')
    parser.add_argument('--model_config', type=str, default=None,
                       help='模型配置文件路径')
    
    # 数据相关参数
    parser.add_argument('--test_dir', type=str, default=None,
                       help='测试数据目录路径')
    parser.add_argument('--test_anno', type=str, default=None,
                       help='测试标注文件路径')
    
    # 评估相关参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='数据加载器工作进程数')
    
    # 后处理参数
    parser.add_argument('--conf_thresh', type=float, default=None,
                       help='置信度阈值')
    parser.add_argument('--topk', type=int, default=None,
                       help='每帧最多输出目标数')
    parser.add_argument('--scale', type=float, default=None,
                       help='坐标缩放因子')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                       help='评估设备 (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    return parser.parse_args()

def args_to_config(args) -> Dict[str, Any]:
    """将命令行参数转换为配置字典格式"""
    config = {}
    
    # 数据配置
    data_config = {}
    if args.test_dir is not None:
        data_config['test_dir'] = args.test_dir
    if args.test_anno is not None:
        data_config['test_anno'] = args.test_anno
    if data_config:
        config['data'] = data_config
    
    # 模型配置
    model_config = {}
    if args.model is not None:
        model_config['name'] = args.model
        print(model_config)
    if model_config:
        config['model'] = model_config
    
    # 评估配置
    eval_config = {}
    if args.batch_size is not None:
        eval_config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        eval_config['num_workers'] = args.num_workers
    if eval_config:
        config['evaluation'] = eval_config
    
    # 后处理配置
    postprocessing_config = {}
    if args.conf_thresh is not None:
        postprocessing_config['conf_thresh'] = args.conf_thresh
    if args.topk is not None:
        postprocessing_config['topk'] = args.topk
    if args.scale is not None:
        postprocessing_config['scale'] = args.scale
    if postprocessing_config:
        config['postprocessing'] = postprocessing_config
    
    # 输出配置
    output_config = {}
    if args.output_dir is not None:
        output_config['output_dir'] = args.output_dir
    if args.experiment_name is not None:
        output_config['experiment_name'] = args.experiment_name
    if output_config:
        config['output'] = output_config
    
    # 设备配置
    if args.device is not None:
        config['device'] = args.device
    
    # 其他配置
    if args.seed is not None:
        config['seed'] = args.seed
    if args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint
    if args.model_config is not None:
        config['model_config_path'] = args.model_config
    
    return config

def get_final_config(args) -> Dict[str, Any]:
    """获取最终配置，优先级：命令行参数 > 配置文件 > 默认值"""
    # 默认配置
    default_config = {
        'data': {
            'test_dir': 'datasets/SpotGEOv2/test',
            'test_anno': 'datasets/SpotGEOv2/test_anno.json'
        },
        'model': {
            'name': 'spotgeo',
            'config': {}
        },
        'evaluation': {
            'batch_size': 8,
            'num_workers': 4
        },
        'postprocessing': {
            'conf_thresh': 0.5,
            'topk': 100,
            'scale': 1.0
        },
        'output': {
            'output_dir': 'outputs/evaluation',
            'experiment_name': None
        },
        'device': 'auto',
        'seed': 42,
        'checkpoint': None,
        'model_config_path': None
    }
    
    # 加载配置文件
    file_config = {}
    if args.config:
        file_config = load_config(args.config)
        logger.info(f"从配置文件加载配置: {args.config}")
    
    # 从命令行参数生成配置
    args_config = args_to_config(args)
    
    # 合并配置：默认配置 <- 文件配置 <- 命令行配置
    final_config = merge_configs(default_config, file_config)
    final_config = merge_configs(final_config, args_config)
    
    return final_config

def load_model_config(config_path: str) -> dict:
    """加载模型配置"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                import json
                return json.load(f)
    return {}

def validate_config(config: Dict[str, Any]) -> None:
    """验证配置的有效性"""
    # 检查必需的路径
    if not config.get('checkpoint'):
        raise ValueError("缺少必需的参数: checkpoint")
    
    if not os.path.exists(config['checkpoint']):
        raise ValueError(f"检查点文件不存在: {config['checkpoint']}")
    
    # 检查数据路径
    data_config = config.get('data', {})
    required_data_keys = ['test_dir', 'test_anno']
    
    for key in required_data_keys:
        if key not in data_config or not data_config[key]:
            raise ValueError(f"缺少必需的数据配置: {key}")
        
        path = data_config[key]
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {path}")

def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """转换配置中的数据类型，确保数值参数是正确的类型"""
    converted_config = config.copy()
    
    # 转换评估配置中的数值参数
    eval_config = converted_config.get('evaluation', {})
    
    # 需要转换为int的参数
    int_params = ['batch_size', 'num_workers']
    for param in int_params:
        if param in eval_config:
            try:
                eval_config[param] = int(eval_config[param])
            except (ValueError, TypeError):
                raise ValueError(f"无法将评估参数 {param} 转换为整数: {eval_config[param]}")
    
    # 转换后处理配置中的数值参数
    postprocessing_config = converted_config.get('postprocessing', {})
    
    # 需要转换为float的参数
    float_params = ['conf_thresh', 'scale']
    for param in float_params:
        if param in postprocessing_config:
            try:
                postprocessing_config[param] = float(postprocessing_config[param])
            except (ValueError, TypeError):
                raise ValueError(f"无法将后处理参数 {param} 转换为浮点数: {postprocessing_config[param]}")
    
    # 需要转换为int的参数
    postprocessing_int_params = ['topk']
    for param in postprocessing_int_params:
        if param in postprocessing_config:
            try:
                postprocessing_config[param] = int(postprocessing_config[param])
            except (ValueError, TypeError):
                raise ValueError(f"无法将后处理参数 {param} 转换为整数: {postprocessing_config[param]}")
    
    # 转换其他数值参数
    if 'seed' in converted_config:
        try:
            converted_config['seed'] = int(converted_config['seed'])
        except (ValueError, TypeError):
            raise ValueError(f"无法将参数 seed 转换为整数: {converted_config['seed']}")
    
    return converted_config

def custom_collate_fn(batch):
    """
    自定义的批处理函数，处理PIL图像转tensor
    """
    # 准备转换器
    transform = transforms.ToTensor()
    
    # 分别处理每个字段
    images = []
    labels = []
    sequence_names = []
    frame_indices = []
    
    for sample in batch:
        # 处理图像
        image = sample['image']
        if isinstance(image, Image.Image):
            image = transform(image)  # PIL -> tensor
        images.append(image)
        
        # 其他字段直接添加
        labels.append(sample['label'])
        sequence_names.append(sample['sequence_name'])
        frame_indices.append(sample['frame_idx'])
    
    # 堆叠图像tensor
    images = torch.stack(images)
    
    return {
        'image': images,
        'label': labels,
        'sequence_name': sequence_names,
        'frame_idx': frame_indices
    }

def main():
    """主函数"""
    args = parse_args()
    
    # 获取最终配置
    config = get_final_config(args)
    
    # 验证配置
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"配置验证失败: {e}")
        return
    
    # 转换配置中的数据类型
    try:
        config = convert_config_types(config)
    except ValueError as e:
        logger.error(f"配置类型转换失败: {e}")
        return
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # 设置设备
    device_config = config['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    logger.info(f"使用设备: {device}")
    logger.info(f"最终配置:")
    logger.info(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    
    # 创建输出目录
    output_config = config['output']
    output_dir = output_config['output_dir']
    if output_config['experiment_name']:
        output_dir = os.path.join(output_dir, output_config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 加载模型配置
    model_config_path = config.get('model_config_path')
    model_config = load_model_config(model_config_path) if model_config_path else config['model'].get('config', {})
    
    # 如果从配置文件加载了模型配置，使用配置文件中的配置
    if model_config_path and model_config:
        logger.info(f"从配置文件加载模型配置: {model_config_path}")
        logger.info(f"完整配置: {model_config}")
        # 从配置文件中提取model.config字段
        if 'model' in model_config and 'config' in model_config['model']:
            final_model_config = model_config['model']['config']
            logger.info(f"提取的模型配置: {final_model_config}")
        else:
            # 如果没有model.config字段，使用整个配置
            final_model_config = model_config
            logger.info(f"使用完整配置作为模型配置: {final_model_config}")
    else:
        # 使用默认配置或命令行指定的配置
        final_model_config = config['model'].get('config', {})
        logger.info(f"使用默认模型配置: {final_model_config}")
    
    # 创建数据集
    logger.info("创建测试数据集...")
    data_config = config['data']
    test_dataset = SpotGEOv2_SingleFrame(
        root_dir=data_config['test_dir'],
        annotation_path=data_config['test_anno'],
        transform=None,
        config={}
    )
    
    # 创建数据加载器
    eval_config = config['evaluation']
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_config['batch_size'],
        shuffle=False,
        num_workers=eval_config['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if eval_config['num_workers'] > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"测试集大小: {len(test_dataset)} 帧")
    logger.info(f"测试批次数: {len(test_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = get_model(config['model']['name'], config=final_model_config)
    
    # 加载检查点
    logger.info(f"加载检查点: {config['checkpoint']}")
    checkpoint = torch.load(config['checkpoint'], map_location=device, weights_only=True)
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 创建评估器
    evaluator = Evaluator(save_dir=output_dir)
    
    # 获取后处理配置
    postprocessing_config = config['postprocessing']
    conf_thresh = postprocessing_config['conf_thresh']
    topk = postprocessing_config['topk']
    # scale = postprocessing_config['scale']
    
    # 收集预测结果
    predictions = []
    ground_truth = []
    
    logger.info("开始评估...")
    with torch.no_grad():
        # 使用tqdm创建进度条
        progress_bar = tqdm(
            test_loader, 
            desc="评估进度", 
            total=len(test_loader),
            unit="batch",
            ncols=100
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            images = batch['image'].to(device)  # [B, 3, H, W]
            labels = batch['label']  # List[Dict]
            sequence_names = batch['sequence_name']  # List[str]
            frame_indices = batch['frame_idx']  # List[int]
            
            # 前向传播
            outputs = model(images)
            
            # 获取预测结果 - 根据SpotGEOModel的输出格式
            if 'predictions' in outputs:
                # SpotGEOModel的输出格式
                predictions_dict = outputs['predictions']
                cls_pred = predictions_dict.get('cls', None)
                reg_pred = predictions_dict.get('reg', None)
            else:
                # 兼容其他格式
                cls_pred = outputs.get('cls_pred', outputs.get('classification', None))
                reg_pred = outputs.get('reg_pred', outputs.get('regression', None))
            
            if cls_pred is None or reg_pred is None:
                logger.error(f"模型输出格式不正确，实际输出键: {list(outputs.keys())}")
                if 'predictions' in outputs:
                    logger.error(f"predictions键包含: {list(outputs['predictions'].keys())}")
                continue
            
                # 计算缩放因子
            out_h, out_w = cls_pred.shape[-2:]
            scale_x = 640 / out_w  # 假设原图宽度为640
            scale_y = 480 / out_h  # 假设原图高度为480
            scale = (scale_x + scale_y) / 2

            # 使用heatmap_to_coords转换坐标
            coords_results = heatmap_to_coords(
                cls_pred,
                reg_pred,
                conf_thresh=conf_thresh,
                topk=topk,
                scale=scale
            )
            
            # 转换为评估器需要的格式
            for i, (label, seq_name, frame_idx) in enumerate(zip(labels, sequence_names, frame_indices)):
                # 预测结果
                pred_result = coords_results[i]
                pred_dict = {
                    'sequence_id': int(seq_name),
                    'frame': int(frame_idx),
                    'object_coords': pred_result['object_coords']
                }
                predictions.append(pred_dict)
                
                # 真实标签
                gt_dict = {
                    'sequence_id': int(seq_name),
                    'frame': int(frame_idx),
                    'object_coords': label['object_coords']
                }
                ground_truth.append(gt_dict)
            
            # 更新进度条描述
            progress_bar.set_postfix({
                'processed': f"{batch_idx + 1}/{len(test_loader)}",
                'predictions': len(predictions),
                'ground_truth': len(ground_truth)
            })
    
    # 评估预测结果
    logger.info("开始计算评估指标...")
    try:
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            save=True,
            plot=True
        )
        logger.info("评估完成！")
        logger.info(evaluator.get_summary(results))
        
        # 保存详细结果
        results_save_path = os.path.join(output_dir, 'evaluation_results.json')
        import json
        with open(results_save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"详细结果已保存到: {results_save_path}")
        
        # 保存预测结果
        predictions_save_path = os.path.join(output_dir, 'predictions.json')
        with open(predictions_save_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        logger.info(f"预测结果已保存到: {predictions_save_path}")
        
        # 保存真实标签
        ground_truth_save_path = os.path.join(output_dir, 'ground_truth.json')
        with open(ground_truth_save_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=4, ensure_ascii=False)
        logger.info(f"真实标签已保存到: {ground_truth_save_path}")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 