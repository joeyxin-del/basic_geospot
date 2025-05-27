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
    def __init__(
        self,
        save_dir: str = "visualizations",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        style: str = "seaborn"
    ):
        """
        初始化可视化工具。
        
        Args:
            save_dir: 可视化结果保存目录
            figsize: 图像大小
            dpi: 图像分辨率
            style: matplotlib样式
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use(style)
        
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
        
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        xlabel: str = "Epoch",
        ylabel: str = "Value",
        save: bool = True,
        show: bool = False
    ) -> Optional[Figure]:
        """
        绘制训练指标曲线。
        
        Args:
            metrics: 指标数据，格式为 {指标名: [值列表]}
            title: 图像标题
            xlabel: x轴标签
            ylabel: y轴标签
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            matplotlib图像对象
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        for name, values in metrics.items():
            plt.plot(values, label=name, marker='o', markersize=3)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        
        if save:
            save_path = self._get_save_path(f"metrics_{title.lower().replace(' ', '_')}")
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Saved metrics plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
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
            save=save,
            show=show
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
            
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save: bool = True,
        show: bool = False
    ) -> Optional[Figure]:
        """
        绘制混淆矩阵。
        
        Args:
            confusion_matrix: 混淆矩阵
            labels: 类别标签
            title: 图像标题
            normalize: 是否归一化
            save: 是否保存图像
            show: 是否显示图像
            
        Returns:
            matplotlib图像对象
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        
        # 添加颜色条
        plt.colorbar()
        
        # 设置刻度标签
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha='right')
        plt.yticks(tick_marks, labels)
        
        # 添加数值标签
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    format(confusion_matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black"
                )
                
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if save:
            save_path = self._get_save_path(f"confusion_{title.lower().replace(' ', '_')}")
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Saved confusion matrix plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()


# 导出
__all__ = ['Visualizer'] 