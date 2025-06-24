#!/usr/bin/env python3
"""
单帧训练脚本
使用SpotGEOv2_SingleFrame数据集和TrainerSingleFrame进行单帧训练
支持yaml配置文件和命令行参数
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.spotgeov2_singleframe import SpotGEOv2_SingleFrame
from src.training.trainer_singleframe import TrainerSingleFrame
from src.models import get_model
from src.utils import get_logger
from src.transforms.factory import TransformFactory
from src.transforms.image import AdvancedAugmentation

logger = get_logger('train_singleframe')

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
    parser = argparse.ArgumentParser(description='单帧训练脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (yaml格式)')
    
    # 数据相关参数
    parser.add_argument('--train_dir', type=str, default=None,
                       help='训练数据目录路径')
    parser.add_argument('--train_anno', type=str, default=None,
                       help='训练标注文件路径')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='验证数据目录路径')
    parser.add_argument('--val_anno', type=str, default=None,
                       help='验证标注文件路径')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称')
    parser.add_argument('--model_config', type=str, default=None,
                       help='模型配置文件路径')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['step', 'cosine', 'none'], help='学习率调度器类型')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    # 训练控制参数
    parser.add_argument('--checkpoint_interval', type=int, default=None,
                       help='检查点保存间隔')
    parser.add_argument('--eval_interval', type=int, default=None,
                       help='验证间隔')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='早停耐心值')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备 (cuda/cpu/auto)')
    parser.add_argument('--use_swanlab', action='store_true', default=None,
                        help='是否使用swanlab记录')
    parser.add_argument('--swanlab_project', type=str, default=None,
                        help='swanlab项目名称')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    return parser.parse_args()

def args_to_config(args) -> Dict[str, Any]:
    """将命令行参数转换为配置字典格式"""
    config = {}
    
    # 数据配置
    data_config = {}
    if args.train_dir is not None:
        data_config['train_dir'] = args.train_dir
    if args.train_anno is not None:
        data_config['train_anno'] = args.train_anno
    if args.val_dir is not None:
        data_config['val_dir'] = args.val_dir
    if args.val_anno is not None:
        data_config['val_anno'] = args.val_anno
    if data_config:
        config['data'] = data_config
    
    # 模型配置
    model_config = {}
    if args.model is not None:
        model_config['name'] = args.model
    if model_config:
        config['model'] = model_config
    
    # 训练配置
    training_config = {}
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        training_config['num_workers'] = args.num_workers
    if args.epochs is not None:
        training_config['epochs'] = args.epochs
    if args.lr is not None:
        training_config['lr'] = args.lr
    if args.optimizer is not None:
        training_config['optimizer'] = args.optimizer
    if args.scheduler is not None:
        training_config['scheduler'] = args.scheduler
    if args.checkpoint_interval is not None:
        training_config['checkpoint_interval'] = args.checkpoint_interval
    if args.eval_interval is not None:
        training_config['eval_interval'] = args.eval_interval
    if args.early_stopping_patience is not None:
        training_config['early_stopping_patience'] = args.early_stopping_patience
    if training_config:
        config['training'] = training_config
    
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
    
    # 日志配置
    logging_config = {}
    if args.use_swanlab is not None:
        logging_config['use_swanlab'] = args.use_swanlab
    if args.swanlab_project is not None:
        logging_config['swanlab_project'] = args.swanlab_project
    if logging_config:
        config['logging'] = logging_config
    
    # 其他配置
    if args.seed is not None:
        config['seed'] = args.seed
    if args.resume is not None:
        config['resume'] = args.resume
    if args.model_config is not None:
        config['model_config_path'] = args.model_config
    
    return config

def get_final_config(args) -> Dict[str, Any]:
    """获取最终配置，优先级：命令行参数 > 配置文件 > 默认值"""
    # 默认配置
    default_config = {
        'data': {
            'train_dir': 'datasets/SpotGEOv2/train',
            'train_anno': 'datasets/SpotGEOv2/train_anno.json',
            'val_dir': 'datasets/SpotGEOv2/test',
            'val_anno': 'datasets/SpotGEOv2/test_anno.json'
        },
        'model': {
            'name': 'spotgeo',
            'config': {}
        },
        'training': {
            'batch_size': 8,
            'num_workers': 4,
            'epochs': 100,
            'lr': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'step',
            'checkpoint_interval': 10,
            'eval_interval': 1,
            'early_stopping_patience': 10
        },
        'output': {
            'output_dir': 'outputs',
            'experiment_name': None
        },
        'device': 'auto',
        'logging': {
            'use_swanlab': False,
            'swanlab_project': 'spotgeo-singleframe',
            'swanlab_mode': 'cloud'
        },
        'seed': 42,
        'resume': None,
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

def create_optimizer(model: nn.Module, optimizer_type: str, lr: float) -> torch.optim.Optimizer:
    """创建优化器"""
    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str, scheduler_config: dict = None):
    """创建学习率调度器"""
    from src.schedulers import SchedulerFactory
    
    if scheduler_type == 'none':
        return None
    
    return SchedulerFactory.create(
        name=scheduler_type,
        optimizer=optimizer,
        config=scheduler_config
    )

def validate_config(config: Dict[str, Any]) -> None:
    """验证配置的有效性"""
    # 检查必需的数据路径
    data_config = config.get('data', {})
    required_data_keys = ['train_dir', 'train_anno', 'val_dir', 'val_anno']
    
    for key in required_data_keys:
        if key not in data_config or not data_config[key]:
            raise ValueError(f"缺少必需的数据配置: {key}")
        
        path = data_config[key]
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {path}")

def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """转换配置中的数据类型，确保数值参数是正确的类型"""
    converted_config = config.copy()
    
    # 转换训练配置中的数值参数
    training_config = converted_config.get('training', {})
    
    # 需要转换为int的参数
    int_params = ['batch_size', 'num_workers', 'epochs', 'checkpoint_interval', 
                  'eval_interval', 'early_stopping_patience']
    for param in int_params:
        if param in training_config:
            try:
                training_config[param] = int(training_config[param])
            except (ValueError, TypeError):
                raise ValueError(f"无法将训练参数 {param} 转换为整数: {training_config[param]}")
    
    # 需要转换为float的参数
    float_params = ['lr']
    for param in float_params:
        if param in training_config:
            try:
                training_config[param] = float(training_config[param])
            except (ValueError, TypeError):
                raise ValueError(f"无法将训练参数 {param} 转换为浮点数: {training_config[param]}")
    
    # 转换模型配置中的数值参数
    model_config = converted_config.get('model', {}).get('config', {})
    
    # 转换模型参数
    if 'num_classes' in model_config:
        try:
            model_config['num_classes'] = int(model_config['num_classes'])
        except (ValueError, TypeError):
            raise ValueError(f"无法将模型参数 num_classes 转换为整数: {model_config['num_classes']}")
    
    if 'dropout' in model_config:
        try:
            model_config['dropout'] = float(model_config['dropout'])
        except (ValueError, TypeError):
            raise ValueError(f"无法将模型参数 dropout 转换为浮点数: {model_config['dropout']}")
    
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

def create_transforms(config: Dict[str, Any]):
    """创建数据增强转换"""
    aug_config = config.get('augmentation', {})
    
    # 如果使用高级数据增强
    if aug_config.get('use_advanced', True):
        logger.info("使用高级数据增强...")
        return [AdvancedAugmentation(config=aug_config.get('advanced', {}))]
    
    # 如果使用组合流水线
    if aug_config.get('use_pipeline', False):
        logger.info("使用数据增强流水线...")
        return [AugmentationPipeline(config=aug_config.get('pipeline', {}))]
    
    # 否则使用单独的数据增强
    logger.info("使用单独的数据增强...")
    train_transforms = []

    # 随机翻转
    flip_config = aug_config.get('random_flip', {})
    if flip_config.get('enabled', False):
        train_transforms.append(
            RandomFlip(config=flip_config)
        )

    # 随机旋转
    rotation_config = aug_config.get('random_rotation', {})
    if rotation_config.get('enabled', False):
        train_transforms.append(
            RandomRotation(
                degrees=rotation_config.get('degrees', 10),
                p=rotation_config.get('p', 0.5),
                config=rotation_config
            )
        )

    # 颜色增强
    color_config = aug_config.get('color_augmentation', {})
    if color_config.get('enabled', False):
        train_transforms.append(
            ColorAugmentation(config=color_config)
        )

    if not train_transforms:
        logger.warning("没有启用任何数据增强转换！")
    else:
        logger.info(f"已启用的数据增强: {[type(t).__name__ for t in train_transforms]}")

    return train_transforms

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
    
    # 创建数据增强
    train_transforms = create_transforms(config)
    logger.info("创建数据增强...")
    logger.info(f"使用的数据增强: {[type(t).__name__ for t in train_transforms]}")
    
    # 创建数据集
    logger.info("创建数据集...")
    data_config = config['data']
    train_dataset = SpotGEOv2_SingleFrame(
        root_dir=data_config['train_dir'],
        annotation_path=data_config['train_anno'],
        transform=train_transforms,  # 应用数据增强到训练集
        config={}
    )
    
    val_dataset = SpotGEOv2_SingleFrame(
        root_dir=data_config['val_dir'],
        annotation_path=data_config['val_anno'],
        transform=None,  # 验证集不使用数据增强
        config={}
    )
    
    # 创建数据加载器
    training_config = config['training']
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if training_config['num_workers'] > 0 else False,  # 保持工作进程存活
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if training_config['num_workers'] > 0 else False,  # 保持工作进程存活
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"训练集大小: {len(train_dataset)} 帧")
    logger.info(f"验证集大小: {len(val_dataset)} 帧")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = get_model(config['model']['name'], config=config['model'].get('config', {}))
    # 使用torchinfo打印模型信息
    from torchinfo import summary
    
    # 获取一个批次的输入尺寸
    batch_size = training_config['batch_size']
    input_size = (batch_size, 3, 480, 640)  # 假设输入图像尺寸为480x640
    
    # 打印模型结构、参数量和计算量
    model_stats = summary(
        model, 
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=4,
        device=device
    )
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    # 计算模型的计算量(FLOPs)和参数量
    from thop import profile
    input_tensor = torch.randn(32, 3, 480, 640).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    
    # 转换为更易读的格式
    def format_size(size):
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if size < 1000:
                return f"{size:.2f}{unit}"
            size /= 1000
    
    logger.info(f"模型计算量(FLOPs): {format_size(flops)}FLOPs")
    logger.info(f"模型参数量(Params): {format_size(params)}") 
    
    # 创建优化器和学习率调度器
    optimizer = create_optimizer(model, training_config['optimizer'], training_config['lr'])
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=training_config['scheduler'],
        scheduler_config=training_config.get('scheduler_config')
    )
    
    # 创建训练器
    logger.info("创建训练器...")
    output_config = config['output']
    logging_config = config['logging']
    
    # 获取后处理配置
    postprocessing_config = config.get('postprocessing', {})
    conf_thresh = postprocessing_config.get('conf_thresh', 0.5)
    topk = postprocessing_config.get('topk', 100)
    
    trainer = TrainerSingleFrame(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_config['output_dir'],
        experiment_name=output_config['experiment_name'],
        use_swanlab=logging_config['use_swanlab'],
        swanlab_project=logging_config['swanlab_project'],
        swanlab_mode=logging_config['swanlab_mode'],
        checkpoint_interval=training_config['checkpoint_interval'],
        eval_interval=training_config['eval_interval'],
        max_epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping_patience'],
        resume=config['resume'],
        conf_thresh=conf_thresh,  # 传递置信度阈值
        topk=topk                 # 传递topk参数
    )
    
    # 开始训练
    logger.info("开始单帧训练...")
    try:
        results = trainer.train()
        logger.info("训练完成！")
        logger.info(f"最佳epoch: {results['best_epoch']}")
        logger.info(f"最佳分数: {results['best_score']:.4f}")
        logger.info(f"总训练epoch: {results['total_epochs']}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 