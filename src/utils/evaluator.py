import json
import os
from datetime import datetime
from typing import Dict, List, Union, Tuple

import numpy as np

from src.utils import get_logger, Visualizer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

logger = get_logger('evaluator')

class Evaluator:
    """
    SpotGEO评估工具类。
    实现与validation.py相同的评估指标和计算方法。
    """
    def __init__(
        self,
        save_dir: str = "evaluations",
        tau: float = 10.0,
        eps: float = 3.0,
        img_width: float = 639.5,
        img_height: float = 479.5,
        frames_per_sequence: int = 5,
        max_objects: int = 30
    ):
        """
        初始化评估器。
        
        Args:
            save_dir: 评估结果保存目录
            tau: 距离阈值，用于判断预测是否匹配真实目标
            eps: 最小误差阈值，小于此值的误差视为0
            img_width: 图像宽度
            img_height: 图像高度
            frames_per_sequence: 每个序列的帧数
            max_objects: 每帧最大目标数
        """
        self.save_dir = save_dir
        self.tau = tau
        self.eps = eps
        self.img_width = img_width
        self.img_height = img_height
        self.frames_per_sequence = frames_per_sequence
        self.max_objects = max_objects
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化可视化工具
        self.visualizer = Visualizer(save_dir=os.path.join(save_dir, "plots"))
        
    def _flat_to_hierarchical(self, labels: List[Dict]) -> Dict:
        """
        将扁平的标注列表转换为层次化字典。
        
        Args:
            labels: 标注列表，每个元素包含sequence_id, frame, num_objects, object_coords
            
        Returns:
            按序列ID和帧ID索引的字典
        """
        seqs = dict()
        for label in labels:
            seq_id = label['sequence_id']
            frame_id = label['frame']
            coords = label['object_coords']
            
            if seq_id not in seqs:
                seqs[seq_id] = defaultdict(dict)
            seqs[seq_id][frame_id] = np.array(coords)
        
        return seqs
        
    def _score_frame(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Tuple[int, int, int, float]:
        """
        评估单帧的预测结果。
        
        Args:
            predictions: 预测坐标数组，形状为 [N, 2]
            ground_truth: 真实坐标数组，形状为 [M, 2]
            
        Returns:
            TP: 真阳性数量
            FN: 假阴性数量
            FP: 假阳性数量
            sse: 平方误差和
        """
        if len(predictions) == 0 and len(ground_truth) == 0:
            return 0, 0, 0, 0
        elif len(predictions) == 0 and len(ground_truth) > 0:
            return 0, len(ground_truth), 0, len(ground_truth) * self.tau**2
        elif len(predictions) > 0 and len(ground_truth) == 0:
            return 0, 0, len(predictions), len(predictions) * self.tau**2
            
        # 计算预测和真实值之间的欧几里得距离
        D = cdist(predictions, ground_truth)
        
        # 截断超过阈值的距离
        D[D > self.tau] = 1000
        
        # 使用匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(D)
        matching = D[row_ind, col_ind]
        
        # 计算评估指标
        TP = sum(matching <= self.tau)
        FN = len(ground_truth) - len(row_ind) + sum(matching > self.tau)
        FP = len(predictions) - len(row_ind) + sum(matching > self.tau)
        
        # 计算截断回归误差
        tp_distances = matching[matching < self.tau]
        tp_distances[tp_distances < self.eps] = 0
        sse = sum(tp_distances) + (FN + FP) * self.tau**2
        
        return TP, FN, FP, sse
        
    def _score_sequence(
        self,
        predictions: Dict[int, np.ndarray],
        ground_truth: Dict[int, np.ndarray]
    ) -> Tuple[int, int, int, float]:
        """
        评估单个序列的预测结果。
        
        Args:
            predictions: 预测字典，键为帧ID，值为坐标数组
            ground_truth: 真实值字典，键为帧ID，值为坐标数组
            
        Returns:
            TP: 总真阳性数量
            FN: 总假阴性数量
            FP: 总假阳性数量
            mse: 均方误差
        """
        assert set(predictions.keys()) == set(ground_truth.keys())
        
        frame_scores = [
            self._score_frame(predictions[k], ground_truth[k])
            for k in predictions.keys()
        ]
        
        TP = sum(x[0] for x in frame_scores)
        FN = sum(x[1] for x in frame_scores)
        FP = sum(x[2] for x in frame_scores)
        sse = sum(x[3] for x in frame_scores)
        
        mse = 0 if (TP + FN + FP) == 0 else sse / (TP + FN + FP)
        return TP, FN, FP, mse
        
    def evaluate(
        self,
        predictions: Union[str, List[Dict]],
        ground_truth: Union[str, List[Dict]],
        save: bool = True,
        plot: bool = True
    ) -> Dict[str, float]:
        """
        评估预测结果。
        
        Args:
            predictions: 预测结果，可以是JSON文件路径或标注列表
            ground_truth: 真实值，可以是JSON文件路径或标注列表
            save: 是否保存评估结果
            plot: 是否绘制评估结果图表
            
        Returns:
            包含评估指标的字典
        """
        # 加载数据
        if isinstance(predictions, str):
            with open(predictions, 'rt') as fp:
                predictions = json.load(fp)
        if isinstance(ground_truth, str):
            with open(ground_truth, 'rt') as fp:
                ground_truth = json.load(fp)
                
        # 转换为层次化格式
        pred_h = self._flat_to_hierarchical(predictions)
        gt_h = self._flat_to_hierarchical(ground_truth)
        
        # 确保预测和真实值包含相同的序列
        assert set(pred_h.keys()) == set(gt_h.keys())
        
        # 计算每个序列的分数
        seq_scores = [
            self._score_sequence(pred_h[k], gt_h[k])
            for k in pred_h.keys()
        ]
        
        # 汇总结果
        TP = sum(x[0] for x in seq_scores)
        FN = sum(x[1] for x in seq_scores)
        FP = sum(x[2] for x in seq_scores)
        mse = sum(x[3] for x in seq_scores)
        
        # 计算评估指标
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 整理结果
        results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mse': float(mse),
            'score': float(1 - f1),  # 最终分数为1-F1
            'tp': int(TP),
            'fn': int(FN),
            'fp': int(FP)
        }
        
        # 保存结果
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"evaluation_{timestamp}.json")
            with open(save_path, 'w') as fp:
                json.dump(results, fp, indent=4)
            logger.info(f"Evaluation results saved to {save_path}")
            
        # 绘制结果
        if plot:
            self.visualizer.plot_metrics(
                metrics={
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'MSE': mse
                },
                title="Evaluation Results",
                save_name="evaluation_metrics"
            )
            
        return results
        
    def get_summary(self, results: Dict[str, float]) -> str:
        """
        生成评估结果摘要。
        
        Args:
            results: 评估结果字典
            
        Returns:
            格式化的结果摘要字符串
        """
        summary = [
            "Evaluation Results Summary:",
            f"Precision: {results['precision']:.4f}",
            f"Recall: {results['recall']:.4f}",
            f"F1 Score: {results['f1_score']:.4f}",
            f"MSE: {results['mse']:.4f}",
            f"Final Score: {results['score']:.4f}",
            f"True Positives: {results['tp']}",
            f"False Negatives: {results['fn']}",
            f"False Positives: {results['fp']}"
        ]
        return "\n".join(summary)


# 导出
__all__ = ['Evaluator'] 