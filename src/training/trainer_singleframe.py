import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
from tqdm import tqdm

# 可选导入swanlab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not available, logging will be disabled")

from src.utils import get_logger
from src.utils.evaluator import Evaluator
from src.utils.visualizer import Visualizer
from src.postprocessing.heatmap_to_coords import heatmap_to_coords

logger = get_logger('trainer_singleframe')

class TrainerSingleFrame:
    """
    单帧训练器类，负责单帧模型训练和验证的核心功能。
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
        use_swanlab: bool = True,  # 默认设置为False
        swanlab_project: str = 'spotgeo-singleframe',
        swanlab_mode: str = 'cloud',  # 默认设置为offline模式
        checkpoint_interval: int = 10,
        eval_interval: int = 1,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        resume: Optional[str] = None,
        conf_thresh: float = 0.5,  # 添加置信度阈值参数
        topk: int = 100,           # 添加topk参数
        log_epoch_metrics: bool = True,  # 是否记录每个epoch的指标  
        log_batch_metrics: bool = True,  # 是否记录每个batch的指标
        log_gradients: bool = True,      # 是否记录梯度信息
        config: Optional[Dict[str, Any]] = None  # 添加配置参数
    ):
        """
        初始化单帧训练器。
        
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
            swanlab_mode: swanlab模式 ('online', 'offline', 'disabled')
            checkpoint_interval: 检查点保存间隔（epoch）
            eval_interval: 验证间隔（epoch）
            max_epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            resume: 恢复训练的检查点路径（可选）
            conf_thresh: 置信度阈值
            topk: topk参数
            log_batch_metrics: 是否记录每个batch的指标
            log_gradients: 是否记录梯度信息
            config: 配置参数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        
        # 实验管理
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S_singleframe")
        self.output_dir = os.path.join(output_dir, self.experiment_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'configs'), exist_ok=True)  # 添加configs目录
        
        # 保存训练配置
        config = {
            'experiment_name': self.experiment_name,
            'model': {
                'name': model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in model.parameters())
            },
            'optimizer': {
                'name': optimizer.__class__.__name__,
                'lr': optimizer.param_groups[0]['lr'],
                'params': {k: v for k, v in optimizer.defaults.items()}
            },
            'scheduler': {
                'name': scheduler.__class__.__name__ if scheduler else None,
                'params': scheduler.state_dict() if scheduler else None
            },
            'criterion': criterion.__class__.__name__ if criterion else None,
            'device': device,
            'training': {
                'max_epochs': max_epochs,
                'checkpoint_interval': checkpoint_interval,
                'eval_interval': eval_interval,
                'early_stopping_patience': early_stopping_patience,
                'conf_thresh': conf_thresh,
                'topk': topk,
            },
            'logging': {
                'use_swanlab': use_swanlab,
                'swanlab_project': swanlab_project,
                'swanlab_mode': swanlab_mode,
                'log_epoch_metrics': log_epoch_metrics,
                'log_batch_metrics': log_batch_metrics,
                'log_gradients': log_gradients
            },
            'dataloaders': {
                'train_batch_size': train_loader.batch_size,
                'val_batch_size': val_loader.batch_size,
                'train_num_workers': train_loader.num_workers,
                'val_num_workers': val_loader.num_workers,
            }
        }
        
        # 保存配置到YAML文件
        import yaml
        config_path = os.path.join(self.output_dir, 'configs', 'train_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"训练配置已保存到: {config_path}")
        
        # 训练配置
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # SwanLab配置
        self.log_batch_metrics = log_batch_metrics
        self.log_gradients = log_gradients
        self.log_epoch_metrics = log_epoch_metrics
        
        # 评估器
        self.evaluator = Evaluator(save_dir=os.path.join(self.output_dir, 'evaluations'))
        
        # 可视化器
        self.visualizer = Visualizer(save_dir=os.path.join(self.output_dir, 'plots'))
        
        # 训练状态
        self.current_epoch = 0
        self.best_score = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0  # 添加全局步数计数器
        
        # 后处理参数
        self.conf_thresh = conf_thresh
        self.topk = topk
        
        # swanlab集成
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE
        if self.use_swanlab:
            # 设置swanlab模式
            os.environ["SWANLAB_MODE"] = swanlab_mode
            
            try:
                swanlab.init(
                    project=swanlab_project,
                    name=self.experiment_name,
                    mode=swanlab_mode,
                    config={
                        'model': model.__class__.__name__,
                        'optimizer': optimizer.__class__.__name__,
                        'scheduler': scheduler.__class__.__name__ if scheduler else None,
                        'criterion': criterion.__class__.__name__ if criterion else None,
                        'device': device,
                        'max_epochs': max_epochs,
                        'checkpoint_interval': checkpoint_interval,
                        'eval_interval': eval_interval,
                        'early_stopping_patience': early_stopping_patience,
                        'training_mode': 'single_frame',
                        'log_batch_metrics': log_batch_metrics,
                        'log_gradients': log_gradients,
                        'conf_thresh': conf_thresh,
                        'topk': topk
                    }
                )
                logger.info(f"SwanLab初始化成功，模式: {swanlab_mode}")
            except Exception as e:
                logger.warning(f"SwanLab初始化失败: {e}, 将继续训练但不记录到swanlab")
                self.use_swanlab = False
        elif use_swanlab and not SWANLAB_AVAILABLE:
            logger.warning("swanlab不可用，已禁用swanlab日志记录")
        
        # 恢复训练
        if resume:
            self._load_checkpoint(resume)
        
        self.config = config or {}
        
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
        
    def _generate_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """
        生成高斯核权重。
        
        Args:
            size: 核大小（奇数）
            sigma: 高斯分布的标准差
            
        Returns:
            torch.Tensor: 高斯核权重
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd")
            
        center = size // 2
        x = torch.arange(size, dtype=torch.float32) - center
        y = torch.arange(size, dtype=torch.float32) - center
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma ** 2))
        
        # 归一化使中心点为1
        kernel = kernel / kernel[center, center]
        
        # 获取配置的最小值阈值，默认为0.1
        min_value = self.config.get('model', {}).get('soft_target', {}).get('min_value', 0.1)
        kernel[kernel < min_value] = 0
        
        return kernel
        
    def _create_soft_target_tensors(
        self, 
        labels: List[Dict[str, Any]], 
        image_sizes: Union[Tuple[int, int], List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        创建软标签目标张量，仅支持单尺度模式。
        
        Args:
            labels: 标注列表，每个元素包含frame、num_objects和object_coords
            image_sizes: 热图大小 (height, width)
                
        Returns:
            包含cls、reg和mask的字典
        """
        # 原图尺寸（假设为640x480）
        orig_h, orig_w = 480, 640
        
        # 获取软标签配置
        soft_target_config = self.config.get('model', {}).get('soft_target', {})
        enabled = soft_target_config.get('enabled', False)
        alpha = soft_target_config.get('alpha', 3.0)
        sigma = soft_target_config.get('sigma', 0.8)
        
        # 如果启用了动态sigma
        if soft_target_config.get('dynamic_sigma', {}).get('enabled', False):
            progress = self.current_epoch / self.max_epochs
            init_sigma = soft_target_config['dynamic_sigma']['init_sigma']
            final_sigma = soft_target_config['dynamic_sigma']['final_sigma']
            schedule = soft_target_config['dynamic_sigma']['schedule']
            
            if schedule == 'linear':
                sigma = init_sigma + (final_sigma - init_sigma) * progress
            elif schedule == 'cosine':
                sigma = final_sigma + (init_sigma - final_sigma) * (1 + np.cos(progress * np.pi)) / 2
            elif schedule == 'step':
                sigma = init_sigma if progress < 0.3 else (
                    (init_sigma + final_sigma) / 2 if progress < 0.7 else final_sigma
                )
        
        # 生成高斯核（如果启用软标签）
        if enabled:
            kernel_size = int(alpha * 2 + 1)  # 确保是奇数
            gaussian_kernel = self._generate_gaussian_kernel(kernel_size, sigma)
            gaussian_kernel = gaussian_kernel.to(self.device)
            
        # 获取特征图尺寸
        if isinstance(image_sizes, (list, tuple)) and len(image_sizes) == 2:
            heat_h, heat_w = image_sizes
        else:
            raise ValueError("image_sizes should be a tuple of (height, width)")
            
        # 计算缩放比例
        scale_x = heat_w / orig_w
        scale_y = heat_h / orig_h
        
        # 创建目标张量
        cls_target = torch.zeros((len(labels), 1, heat_h, heat_w), device=self.device)
        reg_target = torch.zeros((len(labels), 2, heat_h, heat_w), device=self.device)
        mask_target = torch.zeros((len(labels), 1, heat_h, heat_w), device=self.device)
        
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
                    if enabled:
                        # 计算高斯核的有效范围
                        k_size = kernel_size // 2
                        for dy in range(-k_size, k_size + 1):
                            for dx in range(-k_size, k_size + 1):
                                new_y = y_idx + dy
                                new_x = x_idx + dx
                                if 0 <= new_x < heat_w and 0 <= new_y < heat_h:
                                    weight = gaussian_kernel[dy + k_size, dx + k_size]
                                    if weight > 0:  # 只设置大于0的权重
                                        cls_target[i, 0, new_y, new_x] = weight
                                        mask_target[i, 0, new_y, new_x] = weight
                    else:
                        # 使用硬标签
                        cls_target[i, 0, y_idx, x_idx] = 1.0
                        mask_target[i, 0, y_idx, x_idx] = 1.0
                    
                    # 回归目标仍然只在中心点设置
                    reg_target[i, 0, y_idx, x_idx] = x_heat - x_idx  # x偏移
                    reg_target[i, 1, y_idx, x_idx] = y_heat - y_idx  # y偏移
        
        return {
            'cls': cls_target,
            'reg': reg_target,
            'mask': mask_target
        }
        
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
            # 准备数据 - 单帧格式
            images = batch['image']  # torch.Tensor (已通过collate_fn转换)
            labels = batch['label']  # List[Dict]
            
            # 确保images是tensor格式并移动到正确设备
            if not isinstance(images, torch.Tensor):
                raise ValueError(f"Expected images to be torch.Tensor, got {type(images)}")
            
            images = images.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()

            outputs = self.model(images)

            # 获取模型输出的空间分辨率
            if 'multi_scale' in outputs['predictions']:
                # 收集所有尺度的输出尺寸
                image_sizes = []
                for scale_pred in outputs['predictions']['multi_scale']:
                    cls_pred = scale_pred['cls']
                    image_sizes.append(cls_pred.shape[-2:])
            else:
                cls_pred = outputs['predictions']['cls']
                image_sizes = cls_pred.shape[-2:]

            # 创建目标张量（使用软标签）
            targets = self._create_soft_target_tensors(labels, image_sizes)
            
            loss_dict = self.model.compute_loss(outputs, targets)
            loss = loss_dict['total_loss']

            # 反向传播
            loss.backward()
            
            # 计算梯度范数（如果启用）
            grad_norm = None
            if self.log_gradients:
                grad_norm = self._get_gradient_norm()
                
            self.optimizer.step()

            # 更新统计信息
            epoch_loss += loss.item()
            self.global_step += 1
            
            # 记录每个batch的指标到SwanLab
            if self.log_batch_metrics:
                batch_metrics = {
                    'batch_loss': loss.item(),
                    'batch_lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch + 1,
                    'batch_idx': batch_idx + 1
                }
                
                # 添加详细的损失组件
                for loss_name, loss_value in loss_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        batch_metrics[f'batch_{loss_name}'] = loss_value.item()
                    elif isinstance(loss_value, dict) and loss_name == 'scale_losses':
                        # 记录每个尺度的损失
                        for scale, scale_loss in loss_value.items():
                            for k, v in scale_loss.items():
                                batch_metrics[f'batch_{scale}_{k}'] = v
                
                # 添加梯度信息
                if grad_norm is not None:
                    batch_metrics['batch_grad_norm'] = grad_norm
                
                # 记录到SwanLab
                self._log_swanlab_metrics(batch_metrics, step=self.global_step)
            
            # 更新进度条显示
            postfix_dict = {
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{epoch_loss / (batch_idx + 1):.4f}"
            }
            
            if grad_norm is not None:
                postfix_dict['Grad Norm'] = f"{grad_norm:.4f}"
                
            train_pbar.set_postfix(postfix_dict)
            
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
        
    def _validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch。
        
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
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
            for batch_idx, batch in enumerate(val_pbar):
                # 准备数据 - 单帧格式
                images = batch['image']  # torch.Tensor (已通过collate_fn转换)
                labels = batch['label']  # List[Dict]
                sequence_names = batch['sequence_name']  # List[str]
                frame_indices = batch['frame_idx']  # List[int]
                
                # 确保images是tensor格式并移动到正确设备
                if not isinstance(images, torch.Tensor):
                    raise ValueError(f"Expected images to be torch.Tensor, got {type(images)}")
                
                images = images.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 获取模型输出的空间分辨率并进行后处理
                if 'multi_scale' in outputs['predictions']:
                    # 使用P2尺度的输出（最高分辨率）进行后处理
                    p2_pred = outputs['predictions']['multi_scale'][0]  # p2是第一个
                    cls_pred = p2_pred['cls']
                    reg_pred = p2_pred['reg']
                else:
                    
                    cls_pred = outputs['predictions']['cls']
                    reg_pred = outputs['predictions']['reg']
                
                out_h, out_w = cls_pred.shape[-2:]
                
                # 使用后处理将热图转换为坐标列表
                scale_x = 640 / out_w
                scale_y = 480 / out_h  # 原图高度 / 热图高度
                scale = (scale_x + scale_y) / 2  # 平均缩放因子
                
                coords_list = heatmap_to_coords(
                    cls_pred=cls_pred.detach().cpu(),
                    reg_pred=reg_pred.detach().cpu(),
                    conf_thresh=self.conf_thresh,  # 使用配置的置信度阈值
                    topk=self.topk,                # 使用配置的topk
                    scale=scale
                )
                
                # 转换为评估器需要的格式
                for batch_idx, (coord_data, label, seq_name, frame_idx) in enumerate(
                    zip(coords_list, labels, sequence_names, frame_indices)
                ):
                    pred_dict = {
                        'sequence_id': int(seq_name),
                        'frame': int(frame_idx),  # 确保是Python int
                        'object_coords': coord_data['object_coords']
                    }
                    gt_dict = {
                        'sequence_id': int(seq_name),
                        'frame': int(frame_idx),  # 确保是Python int
                        'object_coords': label['object_coords']
                    }
                    predictions.append(pred_dict)
                    ground_truth.append(gt_dict)
                    
        # 评估预测结果
        eval_results = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            save=False,
            plot=False
        )
        
        return eval_results
        
    def _log_swanlab_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录指标到SwanLab。
        
        Args:
            metrics: 要记录的指标字典
            step: 步数（可选）
        # """

        if not self.use_swanlab or not SWANLAB_AVAILABLE:
            return
            
        try:
            if step is not None:
                swanlab.log(metrics, step=step)
            else:
                swanlab.log(metrics)
        except Exception as e:
            logger.warning(f"SwanLab日志记录失败: {e}")
            
    def _get_gradient_norm(self) -> float:
        """
        计算模型参数的梯度范数。
        
        Returns:
            梯度范数
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
        
    def train(self):
        """
        训练模型。
        实现完整的训练循环，包括：
        1. 训练和验证
        2. 检查点保存
        3. 早停
        4. swanlab日志记录
        """
        logger.info(f"Starting single-frame training for {self.max_epochs} epochs")
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
        
        metrics_training = []
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # 更新epoch进度条描述
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.max_epochs}")
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 记录每个epoch的训练指标
            if self.log_epoch_metrics:
                epoch_train_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_lr': train_metrics['lr'],
                    'train_time': train_metrics['time']
                }
                self._log_swanlab_metrics(epoch_train_metrics, step=epoch + 1)
            
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
                    'val_score': current_score,
                    'val_f1': val_metrics['f1_score'],
                    'val_mse': val_metrics['mse'],
                    'lr': train_metrics['lr'],
                    'train_time': train_metrics['time'],
                    'best_score': self.best_score,
                    'patience_counter': self.patience_counter
                }
                metrics_training.append(metrics)
                
                # 记录到swanlab
                if self.log_epoch_metrics:
                    # 记录验证指标
                    epoch_val_metrics = {
                        'val_score': current_score,
                        'val_f1': val_metrics['f1_score'],
                        'val_mse': val_metrics['mse'],
                        'best_score': self.best_score,
                        'patience_counter': self.patience_counter
                    }
                    # 添加更多评估指标（如果存在）
                    for key, value in val_metrics.items():
                        if key not in ['score', 'f1_score', 'mse']:  # 移除'loss'检查
                            epoch_val_metrics[f'val_{key}'] = value
                            
                    self._log_swanlab_metrics(epoch_val_metrics, step=epoch + 1)
                    
                    # 如果是最佳模型，记录特殊标记
                    if current_score < self.best_score:
                        self._log_swanlab_metrics({
                            'best_model_epoch': epoch + 1,
                            'best_model_score': current_score
                        }, step=epoch + 1)
                
                # 更新进度条后缀信息
                epoch_pbar.set_postfix({
                    'Train Loss': f"{train_metrics['loss']:.4f}",
                    'Val F1': f"{val_metrics['f1_score']:.4f}",
                    'Best F1': f"{1 - self.best_score:.4f}"
                })
                    
                # 打印详细验证结果
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} - "
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
        if metrics_training:
            # 转换数据格式：从字典列表转为每个指标的值列表
            plot_data = {}
            for metric_name in metrics_training[0].keys():
                plot_data[metric_name] = [m[metric_name] for m in metrics_training]
            
            # 准备epoch列表
            epochs = [m['epoch'] for m in metrics_training]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 为每个指标生成单独的图表
            for metric_name, values in plot_data.items():
                if metric_name != 'epoch':  # 排除epoch键
                    save_path = os.path.join(self.output_dir, 'plots', f'{metric_name}_{timestamp}.png')
                    self.visualizer.plot_single_metric(metric_name, values, epochs, save_path)
        else:
            logger.warning("No metrics to plot - training was interrupted early")
                
        # 训练结束
        logger.info("Single-frame training completed")
        if self.use_swanlab and SWANLAB_AVAILABLE:
            try:
                # 记录训练总结
                final_metrics = {
                    'final_best_epoch': self.best_epoch + 1,
                    'final_best_score': self.best_score,
                    'final_total_epochs': epoch + 1,
                    'final_training_mode': 'single_frame',
                    'final_experiment_name': self.experiment_name
                }
                
                # 如果有训练指标，记录最终统计
                if metrics_training:
                    final_metrics.update({
                        'final_avg_train_loss': np.mean([m['train_loss'] for m in metrics_training]),
                        'final_avg_val_f1': np.mean([m['val_f1'] for m in metrics_training]),
                        'final_best_val_f1': max([m['val_f1'] for m in metrics_training]),
                        'final_total_training_time': sum([m['train_time'] for m in metrics_training])
                    })
                
                self._log_swanlab_metrics(final_metrics)
                
                swanlab.finish()
                logger.info("SwanLab会话已结束")
            except Exception as e:
                logger.warning(f"SwanLab结束时出现错误: {e}")
            
        # 保存最终结果
        results = {
            'best_epoch': self.best_epoch + 1,
            'best_score': self.best_score,
            'total_epochs': epoch + 1,
            'training_mode': 'single_frame'
        }
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
            
        return results

# 导出
__all__ = ['TrainerSingleFrame'] 