import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import swanlab
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
from tqdm import tqdm

from src.utils import get_logger
from src.utils.evaluator import Evaluator
from src.utils.visualizer import Visualizer
from src.postprocessing.heatmap_to_coords import heatmap_to_coords

logger = get_logger('trainer')

class Trainer:
    """
    训练器类，负责模型训练和验证的核心功能。
    支持断点续训、swanlab集成和实验管理。
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'outputs',
        experiment_name: Optional[str] = None,
        use_swanlab: bool = True,
        swanlab_project: str = 'spotgeo',
        checkpoint_interval: int = 10,
        eval_interval: int = 1,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        resume: Optional[str] = None
    ):
        """
        初始化训练器。
        
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器实例
            scheduler: 学习率调度器实例（可选）
            criterion: 损失函数实例（可选）
            device: 训练设备
            output_dir: 输出目录
            experiment_name: 实验名称（可选，默认使用时间戳）
            use_swanlab: 是否使用swanlab
            swanlab_project: swanlab项目名称
            checkpoint_interval: 检查点保存间隔（epoch）
            eval_interval: 验证间隔（epoch）
            max_epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            resume: 恢复训练的检查点路径（可选）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        
        # 实验管理
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        # 训练配置
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # 评估器
        self.evaluator = Evaluator(save_dir=os.path.join(self.output_dir, 'evaluations'))
        
        # 可视化器
        self.visualizer = Visualizer(save_dir=os.path.join(self.output_dir, 'plots'))
        
        # 训练状态
        self.current_epoch = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # swanlab集成
        self.use_swanlab = use_swanlab
        if use_swanlab:
            swanlab.init(
                project=swanlab_project,
                name=self.experiment_name,
                config={
                    'model': model.__class__.__name__,
                    'optimizer': optimizer.__class__.__name__,
                    'scheduler': scheduler.__class__.__name__ if scheduler else None,
                    'criterion': criterion.__class__.__name__ if criterion else None,
                    'device': device,
                    'max_epochs': max_epochs,
                    'checkpoint_interval': checkpoint_interval,
                    'eval_interval': eval_interval,
                    'early_stopping_patience': early_stopping_patience
                }
            )
            swanlab.watch(model)
        
        # 恢复训练
        if resume:
            self._load_checkpoint(resume)
            
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点。
        
        Args:
            epoch: 当前轮数
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
            
        # 保存最新模型
        last_path = os.path.join(self.output_dir, 'last.pth')
        torch.save(checkpoint, last_path)
        
    def _load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点。
        
        Args:
            checkpoint_path: 检查点路径
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.best_epoch = checkpoint['best_epoch']
        self.patience_counter = checkpoint['patience_counter']
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
        
    def _train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch。
        
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        # 创建训练进度条
        train_pbar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1} Training",
            unit="batch",
            leave=False
        )
        
        for batch_idx, batch in train_pbar:
            # 准备数据
            images = batch['images']  # List[List[PIL.Image]]
            labels = batch['labels']  # List[List[Dict]]
            sequence_names = batch['sequence_name']  # List[str]
            
            batch_loss = 0
            batch_count = 0
            
            # 处理每个序列
            for seq_idx, (seq_images, seq_labels) in enumerate(zip(images, labels)):
                # 将图像转换为张量，确保shape为[batch, 3, H, W]
                seq_images_tensor = torch.stack(seq_images).to(self.device)
                
                # 前向传播获取输出
                self.optimizer.zero_grad()
                outputs = self.model(seq_images_tensor)
                
                # 获取模型输出的空间分辨率
                cls_pred = outputs['predictions']['cls']
                out_h, out_w = cls_pred.shape[-2:]
                
                # 创建目标张量（使用模型输出的分辨率）
                targets = self._create_target_tensors(seq_labels, (out_h, out_w))
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 计算损失
                loss_dict = self.model.compute_loss(outputs, targets)
                loss = loss_dict['total_loss']
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新统计信息
                epoch_loss += loss.item()
                batch_loss += loss.item()
                batch_count += 1
            
            # 更新进度条显示
            avg_batch_loss = batch_loss / batch_count if batch_count > 0 else 0
            train_pbar.set_postfix({
                'Loss': f"{avg_batch_loss:.4f}",
                'Avg Loss': f"{epoch_loss / (batch_idx + 1):.4f}"
            })
                
        # 计算平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
            
        return {
            'loss': avg_loss,
            'time': epoch_time,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
    def _create_target_tensors(self, labels: List[Dict[str, Any]], 
                             image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        创建目标张量。
        
        Args:
            labels: 标注列表，每个元素包含frame、num_objects和object_coords
            image_size: 热图大小 (height, width)
            
        Returns:
            包含分类和回归目标的字典
        """
        # 创建分类和回归目标张量
        cls_target = torch.zeros((len(labels), 1, image_size[0], image_size[1]))
        reg_target = torch.zeros((len(labels), 2, image_size[0], image_size[1]))
        
        # 原图尺寸（假设为640x480）
        orig_h, orig_w = 480, 640
        # 热图尺寸
        heat_h, heat_w = image_size
        
        # 计算缩放比例
        scale_x = heat_w / orig_w
        scale_y = heat_h / orig_h
        
        # 填充目标张量
        for i, label in enumerate(labels):
            coords = label['object_coords']
            for x, y in coords:
                # 将原图坐标映射到热图坐标
                x_heat = x * scale_x
                y_heat = y * scale_y
                
                # 获取网格索引
                x_idx = int(x_heat)
                y_idx = int(y_heat)
                
                # 确保索引在有效范围内
                if 0 <= x_idx < heat_w and 0 <= y_idx < heat_h:
                    # 设置分类目标
                    cls_target[i, 0, y_idx, x_idx] = 1.0
                    
                    # 设置回归目标（相对于网格中心的偏移）
                    reg_target[i, 0, y_idx, x_idx] = x_heat - x_idx  # x偏移
                    reg_target[i, 1, y_idx, x_idx] = y_heat - y_idx  # y偏移
                    
        return {
            'cls': cls_target,
            'reg': reg_target
        }
        
    def _validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch。
        
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        val_loss = 0
        predictions = []
        ground_truth = []
        
        # 创建验证进度条
        val_pbar = tqdm(
            self.val_loader, 
            desc=f"Epoch {self.current_epoch + 1} Validation",
            unit="batch",
            leave=False
        )
        
        with torch.no_grad():
            for batch in val_pbar:
                # 准备数据
                images = batch['images']  # List[List[PIL.Image]]
                labels = batch['labels']  # List[List[Dict]]
                sequence_names = batch['sequence_name']  # List[str]
                
                batch_loss = 0
                batch_count = 0
                
                # 处理每个序列
                for seq_idx, (seq_images, seq_labels) in enumerate(zip(images, labels)):
                    # 将图像转换为张量，确保shape为[batch, 3, H, W]
                    seq_images_tensor = torch.stack(seq_images).to(self.device)
                    
                    # 前向传播
                    outputs = self.model(seq_images_tensor)
                    
                    # 获取模型输出的空间分辨率
                    cls_pred = outputs['predictions']['cls']
                    out_h, out_w = cls_pred.shape[-2:]
                    
                    # 创建目标张量（使用模型输出的分辨率）
                    targets = self._create_target_tensors(seq_labels, (out_h, out_w))
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # 计算损失
                    loss_dict = self.model.compute_loss(outputs, targets)
                    val_loss += loss_dict['total_loss'].item()
                    batch_loss += loss_dict['total_loss'].item()
                    batch_count += 1
                    
                    # 使用后处理将热图转换为坐标列表
                    scale_x = 640 / out_w  # 原图宽度 / 热图宽度
                    scale_y = 480 / out_h  # 原图高度 / 热图高度
                    scale = (scale_x + scale_y) / 2  # 平均缩放因子
                    
                    coords_list = heatmap_to_coords(
                        cls_pred=outputs['predictions']['cls'].detach().cpu(),
                        reg_pred=outputs['predictions']['reg'].detach().cpu(),
                        conf_thresh=0.5,
                        topk=100,
                        scale=scale
                    )
                    
                    # 转换为评估器需要的格式
                    for frame_idx, coord_data in enumerate(coords_list):
                        pred_dict = {
                            'sequence_id': int(sequence_names[seq_idx]),
                            'frame': int(frame_idx),  # 确保是Python int
                            'object_coords': coord_data['object_coords']
                        }
                        gt_dict = {
                            'sequence_id': int(sequence_names[seq_idx]),
                            'frame': int(frame_idx),  # 确保是Python int
                            'object_coords': seq_labels[frame_idx]['object_coords']
                        }
                        predictions.append(pred_dict)
                        ground_truth.append(gt_dict)
                
                # 更新验证进度条
                avg_batch_loss = batch_loss / batch_count if batch_count > 0 else 0
                val_pbar.set_postfix({
                    'Loss': f"{avg_batch_loss:.4f}"
                })
                    
        # 计算平均损失
        avg_loss = val_loss / len(self.val_loader)
        
        # 评估预测结果
        eval_results = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            save=False,
            plot=False
        )
        
        return {
            'loss': avg_loss,
            **eval_results
        }
        
    def train(self):
        """
        训练模型。
        实现完整的训练循环，包括：
        1. 训练和验证
        2. 检查点保存
        3. 早停
        4. swanlab日志记录
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Training device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # 创建epoch级别的进度条
        epoch_pbar = tqdm(
            range(self.current_epoch, self.max_epochs),
            desc="Training Progress",
            unit="epoch",
            initial=self.current_epoch,
            total=self.max_epochs
        )
        
        metrics_trainning = []
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # 更新epoch进度条描述
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.max_epochs}")
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 验证
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._validate_epoch()
                
                # 更新最佳模型
                current_score = val_metrics['score']
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    
                # 记录指标
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_score': current_score,
                    'val_f1': val_metrics['f1_score'],
                    'val_mse': val_metrics['mse'],
                    'lr': train_metrics['lr']
                }
                metrics_trainning.append(metrics)
                
                # 记录到swanlab
                if self.use_swanlab:
                    swanlab.log(metrics)
                
                # 更新进度条后缀信息
                epoch_pbar.set_postfix({
                    'Train Loss': f"{train_metrics['loss']:.4f}",
                    'Val Loss': f"{val_metrics['loss']:.4f}",
                    'Val F1': f"{val_metrics['f1_score']:.4f}",
                    'Best F1': f"{1 - self.best_score:.4f}"
                })
                    
                # 打印详细验证结果
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"Val Score: {current_score:.4f} - "
                    f"Val F1: {val_metrics['f1_score']:.4f} - "
                    f"Val MSE: {val_metrics['mse']:.4f}"
                )
                
                # 早停检查
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs. "
                        f"Best score: {self.best_score:.4f} at epoch {self.best_epoch + 1}"
                    )
                    break
            else:
                # 只有训练，更新简单的进度条信息
                epoch_pbar.set_postfix({
                    'Train Loss': f"{train_metrics['loss']:.4f}",
                    'LR': f"{train_metrics['lr']:.6f}"
                })
                    
            # 保存检查点
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
        
        # 关闭进度条
        epoch_pbar.close()

        # 绘制训练指标图表
        if metrics_trainning:
            # 转换数据格式：从字典列表转为每个指标的值列表
            plot_data = {}
            for metric_name in metrics_trainning[0].keys():
                plot_data[metric_name] = [m[metric_name] for m in metrics_trainning]
            
            # 准备epoch列表
            epochs = [m['epoch'] for m in metrics_trainning]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 为每个指标生成单独的图表
            for metric_name, values in plot_data.items():
                if metric_name != 'epoch':  # 排除epoch键
                    save_path = os.path.join(self.output_dir, 'plots', f'{metric_name}_{timestamp}.png')
                    self.visualizer.plot_single_metric(metric_name, values, epochs, save_path)
        else:
            logger.warning("No metrics to plot - training was interrupted early")
                
        # 训练结束
        logger.info("Training completed")
        if self.use_swanlab:
            swanlab.finish()
            
        # 保存最终结果
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_score': self.best_score,
            'total_epochs': epoch + 1
        }
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

# 导出
__all__ = ['Trainer'] 