import os
import sys
import argparse
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import get_logger
from src.datasets import SpotGEOv2Dataset
from src.datasets.transforms import SpotGEOTransform
from src.models import ModelFactory
from src.training import Trainer
from torch.utils.data import DataLoader, random_split
from src.datasets.collate import spotgeo_collate_fn

logger = get_logger('train')

# 根据操作系统设置数据加载器参数
def get_dataloader_config():
    """根据操作系统返回数据加载器配置"""
    if os.name == 'nt':  # Windows系统
        return {
            'num_workers': 0,
            'pin_memory': True,
            'persistent_workers': False
        }
    else:  # Linux/Unix系统
        return {
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        }

@hydra.main(config_path="../configs", config_name="default", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """
    训练模型的主函数。
    
    Args:
        cfg: Hydra配置对象
    """
    # 打印训练设置
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # 创建输出目录
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(cfg.training.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
        
    # 获取数据加载器配置
    dataloader_config = get_dataloader_config()
    logger.info(f"Using dataloader config: {dataloader_config}")
    
    # 数据转换
    train_transform = SpotGEOTransform(
        image_size=(640, 480),
        normalize=True,
        augment=cfg.data.augmentation.train.enabled,
        augment_config=cfg.data.augmentation.train
    )
    
    val_transform = SpotGEOTransform(
        image_size=(640, 480),
        normalize=True,
        augment=False
    )
    
    # 创建数据集
    dataset = SpotGEOv2Dataset(
        root_dir=cfg.data.train_dir,
        annotation_path=cfg.data.train_anno,
        transform=train_transform,
        config=cfg
    )
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=spotgeo_collate_fn,
        **dataloader_config
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=spotgeo_collate_fn,
        **dataloader_config
    )
    
    # 创建模型
    model = ModelFactory.create(
        name=cfg.model.name,  # 只传递模型名称
        config=cfg.model  # 传递完整的模型配置
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas)
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.T_max,
        eta_min=cfg.scheduler.eta_min
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.training.device,
        output_dir=cfg.training.output_dir,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
        checkpoint_interval=cfg.training.checkpoint_interval,
        eval_interval=cfg.training.eval_interval,
        max_epochs=cfg.training.max_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        resume=cfg.training.resume
    )
    
    # 开始训练
    try:
        results = trainer.train()
        logger.info("Training completed successfully")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
if __name__ == "__main__":
    train() 