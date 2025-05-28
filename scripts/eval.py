import os
import sys
import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.datasets import SpotGEOv2Dataset
from src.datasets.collate import spotgeo_collate_fn
from src.models import ModelFactory
from src.utils import get_logger
from src.utils.evaluator import Evaluator

logger = get_logger('eval_script')

@hydra.main(config_path="../configs", config_name="default")
def evaluate(cfg: DictConfig):
    """
    评估脚本入口函数。
    
    Args:
        cfg: Hydra配置对象
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='评估结果输出目录')
    args = parser.parse_args()
    
    # 打印配置
    logger.info("Evaluation configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # 设置设备
    device = torch.device(cfg.training.device)
    
    # 创建输出目录
    output_dir = args.output_dir or os.path.join(cfg.training.output_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Configuration saved to {config_path}")
    
    # 创建数据集
    test_dataset = SpotGEOv2Dataset(
        root_dir=cfg.data.test_dir,
        annotation_path=cfg.data.test_anno,
        transform=None
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=spotgeo_collate_fn  # 使用自定义的collate函数
    )
    
    # 创建模型
    model = ModelFactory.create(
        name=cfg.model.name,
        **cfg.model
    )
    
    # 加载检查点
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 创建评估器
    evaluator = Evaluator(save_dir=output_dir)
    
    # 收集预测结果
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 准备数据
            images = batch['images']  # List[List[PIL.Image]]
            labels = batch['labels']  # List[List[Dict]]
            sequence_names = batch['sequence_name']  # List[str]
            
            # 处理每个序列
            for seq_idx, (seq_images, seq_labels) in enumerate(zip(images, labels)):
                # 将图像转换为张量
                seq_images = torch.stack([
                    torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                    for img in seq_images
                ]).to(device)
                
                # 前向传播
                outputs = model(seq_images)
                
                # 收集预测结果
                pred_coords = outputs['predictions']['reg'].cpu().numpy()
                
                # 转换为评估器需要的格式
                for frame_idx, pred in enumerate(pred_coords):
                    pred_dict = {
                        'sequence_id': int(sequence_names[seq_idx]),
                        'frame': frame_idx,
                        'object_coords': pred.tolist()
                    }
                    gt_dict = {
                        'sequence_id': int(sequence_names[seq_idx]),
                        'frame': frame_idx,
                        'object_coords': seq_labels[frame_idx]['object_coords']
                    }
                    predictions.append(pred_dict)
                    ground_truth.append(gt_dict)
                    
    # 评估预测结果
    try:
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            save=True,
            plot=True
        )
        logger.info("Evaluation completed successfully")
        logger.info(evaluator.get_summary(results))
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
        
if __name__ == "__main__":
    evaluate() 