import torch
import numpy as np
from typing import List, Dict, Any

def heatmap_to_coords(cls_pred: torch.Tensor, reg_pred: torch.Tensor, conf_thresh: float = 0.5, topk: int = 100, scale: float = 1.0) -> List[Dict[str, Any]]:
    """
    将模型输出的热图（cls_pred, reg_pred）转换为坐标列表。
    Args:
        cls_pred: [B, 1, H, W] 分类热图
        reg_pred: [B, 2, H, W] 回归热图
        conf_thresh: 置信度阈值
        topk: 每帧最多输出目标数
        scale: 坐标缩放因子（如果输出分辨率与原图不同）
    Returns:
        List[Dict]，每个元素格式：
        {
            'frame': int,
            'num_objects': int,
            'object_coords': List[[x, y], ...]
        }
    """
    results = []
    B, _, H, W = cls_pred.shape
    cls_pred = torch.sigmoid(cls_pred)  # 保证在0-1区间
    for b in range(B):
        # 取出当前帧的热图
        heatmap = cls_pred[b, 0]  # [H, W]
        regmap = reg_pred[b]      # [2, H, W]
        # 找到大于阈值的点
        mask = heatmap > conf_thresh
        ys, xs = torch.where(mask)
        scores = heatmap[ys, xs]
        # 如果太多点，取topk
        if scores.numel() > topk:
            topk_scores, idx = torch.topk(scores, topk)
            ys = ys[idx]
            xs = xs[idx]
            scores = topk_scores
        coords = []
        for y, x in zip(ys, xs):
            # 回归偏移
            dx = regmap[0, y, x].item()
            dy = regmap[1, y, x].item()
            # 还原到原图坐标
            x_coord = float((x + dx) * scale)  # 确保是Python float
            y_coord = float((y + dy) * scale)  # 确保是Python float
            coords.append([x_coord, y_coord])
        results.append({
            'frame': int(b),  # 确保是Python int
            'num_objects': int(len(coords)),  # 确保是Python int
            'object_coords': coords
        })
    return results 