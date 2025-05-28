import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from src.utils import get_logger

logger = get_logger('visualizer')

class Visualizer:
    """
    可视化工具类。
    用于可视化模型训练过程和结果，包括损失曲线、准确率曲线、检测结果和特征图等。
    """
    def __init__(self, save_dir: str, style: str = 'default'):
        """
        初始化可视化器。
        
        Args:
            save_dir: 图表保存目录
            style: matplotlib样式名称
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib样式
        try:
            # 尝试使用seaborn样式
            import seaborn as sns
            sns.set_style("whitegrid")
            logger.info("Successfully set seaborn style")
        except (ImportError, Exception) as e:
            # 如果seaborn不可用，使用matplotlib默认样式
            plt.style.use('default')
            logger.warning(f"Failed to set seaborn style: {str(e)}, using default style instead")
            
        # 设置中文字体支持
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        except Exception as e:
            logger.warning(f"Failed to set Chinese font: {str(e)}")
            
        # 设置图表参数
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['font.size'] = 12
        
        # 初始化数据存储
        self.metrics_history: Dict[str, List[float]] = {}
        self.current_epoch = 0
        
    def _get_save_path(self, name: str, suffix: str = "png") -> str:
        """
        获取保存路径。
        
        Args:
            name: 图像名称
            suffix: 文件后缀
            
        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{suffix}"
        return os.path.join(self.save_dir, filename)
        
    def plot_metrics(self, metrics: Dict[str, List[float]], 
                    title: str = "Training Metrics",
                    save_name: Optional[str] = None) -> None:
        """
        绘制训练指标曲线。
        
        Args:
            metrics: 指标字典，键为指标名称，值为指标值列表
            title: 图表标题
            save_name: 保存文件名，如果不指定则使用标题
        """
        plt.figure(figsize=(12, 6))
        
        for name, values in metrics.items():
            plt.plot(values, label=name, marker='o', markersize=3)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        if save_name is None:
            save_name = title.lower().replace(' ', '_')
        save_path = self._get_save_path(f"{save_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved metrics plot to {save_path}")
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        更新训练指标。
        
        Args:
            metrics: 当前epoch的指标数据
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
            
    def plot_training_progress(
        self,
        save: bool = True,
        show: bool = False
    ) -> Optional[Figure]:
        """
        绘制当前训练进度。
        
        Args:
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            matplotlib图像对象
        """
        if not self.metrics_history:
            logger.warning("No metrics data available")
            return None
            
        return self.plot_metrics(
            self.metrics_history,
            title="Training Progress",
            save_name=None
        )
        
    def plot_detection_results(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        boxes: List[List[float]],
        scores: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        title: str = "Detection Results",
        save: bool = True,
        show: bool = False
    ) -> Optional[Figure]:
        """
        绘制目标检测结果。
        
        Args:
            image: 输入图像
            boxes: 检测框列表，每个框为 [x1, y1, x2, y2]
            scores: 检测分数列表
            labels: 检测标签列表
            title: 图像标题
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            matplotlib图像对象
        """
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(image)
        
        # 绘制检测框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # 创建矩形
            rect = plt.Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # 添加标签和分数
            if labels is not None and scores is not None:
                label = f"{labels[i]}: {scores[i]:.2f}"
            elif labels is not None:
                label = labels[i]
            elif scores is not None:
                label = f"{scores[i]:.2f}"
            else:
                label = ""
                
            if label:
                plt.text(
                    x1,
                    y1 - 5,
                    label,
                    color='red',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
        plt.title(title)
        plt.axis('off')
        
        if save:
            save_path = self._get_save_path(f"detection_{title.lower().replace(' ', '_')}")
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Saved detection plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_feature_maps(
        self,
        feature_maps: torch.Tensor,
        n_cols: int = 8,
        title: str = "Feature Maps",
        save: bool = True,
        show: bool = False
    ) -> Optional[Figure]:
        """
        绘制特征图。
        
        Args:
            feature_maps: 特征图张量，形状为 [B, C, H, W]
            n_cols: 每行显示的特征图数量
            title: 图像标题
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            matplotlib图像对象
        """
        # 确保特征图在CPU上
        feature_maps = feature_maps.detach().cpu()
        
        # 获取批次大小和通道数
        b, c, h, w = feature_maps.shape
        
        # 计算行数
        n_rows = (c + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 2, n_rows * 2), dpi=self.dpi)
        
        # 绘制每个特征图
        for i in range(c):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(feature_maps[0, i].numpy(), cmap='viridis')
            plt.title(f"Channel {i}")
            plt.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        
        if save:
            save_path = self._get_save_path(f"features_{title.lower().replace(' ', '_')}")
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Saved feature maps plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            class_names: List[str],
                            title: str = "Confusion Matrix",
                            save_name: Optional[str] = None) -> None:
        """
        绘制混淆矩阵。
        
        Args:
            confusion_matrix: 混淆矩阵数组
            class_names: 类别名称列表
            title: 图表标题
            save_name: 保存文件名
        """
        plt.figure(figsize=(10, 8))
        
        # 归一化混淆矩阵
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # 绘制热力图
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        # 设置坐标轴
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 添加数值标签
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
                
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save_name is None:
            save_name = "confusion_matrix"
        save_path = self._get_save_path(f"{save_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
        
    def plot_sample_predictions(self, images: torch.Tensor,
                              predictions: Dict[str, torch.Tensor],
                              targets: Optional[Dict[str, torch.Tensor]] = None,
                              num_samples: int = 4,
                              save_name: str = "sample_predictions") -> None:
        """
        绘制样本预测结果。
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            predictions: 预测结果字典
            targets: 目标值字典（可选）
            num_samples: 要显示的样本数量
            save_name: 保存文件名
        """
        num_samples = min(num_samples, images.size(0))
        
        # 创建子图
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(num_samples):
            # 显示原始图像
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # 显示预测结果
            pred = predictions['cls'][i, 0].cpu().detach().numpy()
            axes[i, 1].imshow(pred, cmap='hot')
            axes[i, 1].set_title('Prediction Heatmap')
            axes[i, 1].axis('off')
            
            # 如果有目标值，显示目标
            if targets is not None and 'cls' in targets:
                target = targets['cls'][i, 0].cpu().numpy()
                axes[i, 1].contour(target, colors='blue', alpha=0.5)
                
        plt.tight_layout()
        save_path = self._get_save_path(f"{save_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved sample predictions to {save_path}")
        
    def plot_feature_maps(self, features: List[torch.Tensor],
                         save_name: str = "feature_maps") -> None:
        """
        绘制特征图。
        
        Args:
            features: 特征图列表
            save_name: 保存文件名
        """
        for i, feat in enumerate(features):
            # 选择前16个通道
            feat = feat[0, :16].cpu().detach()
            
            # 创建子图
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for j in range(16):
                if j < feat.size(0):
                    axes[j].imshow(feat[j].numpy(), cmap='viridis')
                    axes[j].set_title(f'Channel {j}')
                    axes[j].axis('off')
                    
            plt.tight_layout()
            save_path = self._get_save_path(f"{save_name}_layer_{i}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved feature maps for layer {i} to {save_path}")
            
    def plot_learning_rate(self, lr_history: List[float],
                          save_name: str = "learning_rate") -> None:
        """
        绘制学习率变化曲线。
        
        Args:
            lr_history: 学习率历史记录
            save_name: 保存文件名
        """
        plt.figure(figsize=(10, 4))
        plt.plot(lr_history, marker='o', markersize=3)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.yscale('log')
        
        save_path = self._get_save_path(f"{save_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved learning rate plot to {save_path}")
        
    def plot_gradient_flow(self, named_parameters: Dict[str, torch.Tensor],
                          save_name: str = "gradient_flow") -> None:
        """
        绘制梯度流图。
        
        Args:
            named_parameters: 模型参数字典
            save_name: 保存文件名
        """
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="r")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # 根据实际情况调整
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self._get_save_path(f"{save_name}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved gradient flow plot to {save_path}")


# 导出
__all__ = ['Visualizer'] 