import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .base import BaseTransform
from .factory import TransformFactory


@TransformFactory.register('sequence_crop')
class SequenceCrop(BaseTransform):
    """
    序列裁剪数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化序列裁剪。
        
        Args:
            config: 配置字典，包含：
                - length: 裁剪长度，可以是整数或元组 (min_length, max_length)
                - mode: 裁剪模式，'random'或'center'
        """
        super().__init__(config)
        self.length = self.config.get('length', 100)
        self.mode = self.config.get('mode', 'random')
        
        # 确保length是元组
        if isinstance(self.length, int):
            self.length = (self.length, self.length)
            
    def __call__(self, sequence: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        执行序列裁剪。
        
        Args:
            sequence: 输入序列，可以是numpy数组或PyTorch张量
            
        Returns:
            裁剪后的序列，保持与输入相同的类型
        """
        # 转换为numpy数组进行处理
        np_sequence = self.to_numpy(sequence)
        
        # 获取序列长度
        seq_len = len(np_sequence)
        
        # 确定裁剪长度
        if self.length[0] == self.length[1]:
            crop_len = self.length[0]
        else:
            crop_len = random.randint(self.length[0], min(self.length[1], seq_len))
            
        # 确定裁剪位置
        if self.mode == 'random':
            start = random.randint(0, seq_len - crop_len)
        else:  # center
            start = (seq_len - crop_len) // 2
            
        # 执行裁剪
        cropped = np_sequence[start:start + crop_len]
        
        # 转换回原始类型
        return self.to_original_type(cropped, sequence)


@TransformFactory.register('sequence_noise')
class SequenceNoise(BaseTransform):
    """
    序列噪声数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化序列噪声。
        
        Args:
            config: 配置字典，包含：
                - noise_type: 噪声类型，'gaussian'或'uniform'
                - scale: 噪声强度
                - prob: 应用概率
        """
        super().__init__(config)
        self.noise_type = self.config.get('noise_type', 'gaussian')
        self.scale = self.config.get('scale', 0.1)
        self.prob = self.config.get('prob', 0.5)
        
    def __call__(self, sequence: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        执行序列噪声添加。
        
        Args:
            sequence: 输入序列，可以是numpy数组或PyTorch张量
            
        Returns:
            添加噪声后的序列，保持与输入相同的类型
        """
        # 转换为numpy数组进行处理
        np_sequence = self.to_numpy(sequence)
        
        if random.random() < self.prob:
            # 生成噪声
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, self.scale, np_sequence.shape)
            else:  # uniform
                noise = np.random.uniform(-self.scale, self.scale, np_sequence.shape)
                
            # 添加噪声
            np_sequence = np_sequence + noise
            
        # 转换回原始类型
        return self.to_original_type(np_sequence, sequence)


@TransformFactory.register('sequence_masking')
class SequenceMasking(BaseTransform):
    """
    序列掩码数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化序列掩码。
        
        Args:
            config: 配置字典，包含：
                - mask_ratio: 掩码比例
                - mask_value: 掩码值
                - min_length: 最小掩码长度
                - max_length: 最大掩码长度
        """
        super().__init__(config)
        self.mask_ratio = self.config.get('mask_ratio', 0.15)
        self.mask_value = self.config.get('mask_value', 0)
        self.min_length = self.config.get('min_length', 1)
        self.max_length = self.config.get('max_length', 10)
        
    def __call__(self, sequence: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        执行序列掩码。
        
        Args:
            sequence: 输入序列，可以是numpy数组或PyTorch张量
            
        Returns:
            掩码后的序列，保持与输入相同的类型
        """
        # 转换为numpy数组进行处理
        np_sequence = self.to_numpy(sequence)
        
        # 计算掩码位置
        seq_len = len(np_sequence)
        mask_positions = []
        current_pos = 0
        
        while current_pos < seq_len:
            # 随机决定是否添加掩码
            if random.random() < self.mask_ratio:
                # 随机选择掩码长度
                mask_length = random.randint(self.min_length, min(self.max_length, seq_len - current_pos))
                mask_positions.extend(range(current_pos, current_pos + mask_length))
                current_pos += mask_length
            else:
                current_pos += 1
                
        # 应用掩码
        if mask_positions:
            if len(np_sequence.shape) == 1:
                np_sequence[mask_positions] = self.mask_value
            else:
                np_sequence[mask_positions, :] = self.mask_value
                
        # 转换回原始类型
        return self.to_original_type(np_sequence, sequence)


@TransformFactory.register('sequence_mixup')
class SequenceMixup(BaseTransform):
    """
    序列混合数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化序列混合。
        
        Args:
            config: 配置字典，包含：
                - alpha: 混合参数
                - prob: 应用概率
        """
        super().__init__(config)
        self.alpha = self.config.get('alpha', 0.2)
        self.prob = self.config.get('prob', 0.5)
        
    def __call__(self, sequences: List[Union[np.ndarray, torch.Tensor]]) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        执行序列混合。
        
        Args:
            sequences: 输入序列列表，每个序列可以是numpy数组或PyTorch张量
            
        Returns:
            混合后的序列列表，保持与输入相同的类型
        """
        if len(sequences) < 2 or random.random() >= self.prob:
            return sequences
            
        # 转换为numpy数组进行处理
        np_sequences = [self.to_numpy(seq) for seq in sequences]
        
        # 随机选择两个序列进行混合
        idx1, idx2 = random.sample(range(len(np_sequences)), 2)
        seq1, seq2 = np_sequences[idx1], np_sequences[idx2]
        
        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 确保序列长度相同
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
        
        # 执行混合
        mixed_seq = lam * seq1 + (1 - lam) * seq2
        
        # 更新序列列表
        np_sequences[idx1] = mixed_seq
        
        # 转换回原始类型
        return [self.to_original_type(seq, orig_seq) 
                for seq, orig_seq in zip(np_sequences, sequences)]


@TransformFactory.register('sequence_cutout')
class SequenceCutout(BaseTransform):
    """
    序列截断数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化序列截断。
        
        Args:
            config: 配置字典，包含：
                - num_cuts: 截断数量
                - min_length: 最小截断长度
                - max_length: 最大截断长度
                - fill_value: 填充值
        """
        super().__init__(config)
        self.num_cuts = self.config.get('num_cuts', 1)
        self.min_length = self.config.get('min_length', 1)
        self.max_length = self.config.get('max_length', 10)
        self.fill_value = self.config.get('fill_value', 0)
        
    def __call__(self, sequence: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        执行序列截断。
        
        Args:
            sequence: 输入序列，可以是numpy数组或PyTorch张量
            
        Returns:
            截断后的序列，保持与输入相同的类型
        """
        # 转换为numpy数组进行处理
        np_sequence = self.to_numpy(sequence)
        
        # 获取序列长度
        seq_len = len(np_sequence)
        
        # 执行多次截断
        for _ in range(self.num_cuts):
            # 随机选择截断长度
            cut_length = random.randint(self.min_length, min(self.max_length, seq_len))
            
            # 随机选择截断位置
            start = random.randint(0, seq_len - cut_length)
            
            # 执行截断
            if len(np_sequence.shape) == 1:
                np_sequence[start:start + cut_length] = self.fill_value
            else:
                np_sequence[start:start + cut_length, :] = self.fill_value
                
        # 转换回原始类型
        return self.to_original_type(np_sequence, sequence)


# 导出所有数据增强类
__all__ = [
    'SequenceCrop',
    'SequenceNoise',
    'SequenceMasking',
    'SequenceMixup',
    'SequenceCutout'
] 