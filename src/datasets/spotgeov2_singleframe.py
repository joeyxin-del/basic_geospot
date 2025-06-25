import os
import json
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from .base import DatasetBase
from ..transforms.base import Compose
import torch
import numpy as np

class SpotGEOv2_SingleFrame(DatasetBase):
    """
    SpotGEOv2单帧数据集适配类。
    将序列数据展开为单帧数据，适用于单帧空间暗弱目标检测任务。
    """
    def __init__(
        self,
        root_dir: str,
        annotation_path: str,
        transform: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        :param root_dir: 数据集根目录（train或test文件夹）
        :param annotation_path: 标注文件路径（train_anno.json或test_anno.json）
        :param transform: 数据增强方法或数据增强方法列表
        :param config: 其他配置参数
        """
        super().__init__(config or {})
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        
        # 处理transform参数
        if isinstance(transform, list):
            self.transform = Compose(transform)
        else:
            self.transform = transform
        
        # 加载序列和标注
        self.sequences = self._load_sequences()
        self.annotations = self._load_annotations()
        
        # 创建单帧索引
        self.frame_index = self._create_frame_index()
        
        # 打印数据集信息
        print(f"单帧数据集信息:")
        print(f"- 总序列数: {len(self.sequences)}")
        print(f"- 总帧数: {len(self.frame_index)}")
        if len(self.sequences) > 0:
            sequence_name = self.sequences[0]
            sequence_dir = os.path.join(self.root_dir, sequence_name)
            image_files = [f for f in os.listdir(sequence_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"- 第一个序列中的图片数量: {len(image_files)}")

    def _load_sequences(self) -> List[str]:
        """
        加载所有序列文件夹名称。
        """
        return [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

    def _load_annotations(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        加载标注文件。
        标注文件格式为JSON数组，每个元素包含sequence_id、frame、num_objects和object_coords等信息。
        """
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 将标注按sequence_id分组
        sequence_annotations = {}
        for anno in annotations:
            seq_id = anno['sequence_id']
            if seq_id not in sequence_annotations:
                sequence_annotations[seq_id] = []
            sequence_annotations[seq_id].append(anno)
        
        # 对每个序列的标注按帧号排序
        for seq_id in sequence_annotations:
            sequence_annotations[seq_id].sort(key=lambda x: x['frame'])
            
        return sequence_annotations

    def _create_frame_index(self) -> List[Tuple[str, int, Dict[str, Any]]]:
        """
        创建帧索引，将序列数据展开为单帧数据。
        
        Returns:
            List[Tuple[sequence_name, frame_idx, annotation]]
        """
        frame_index = []
        
        for sequence_name in self.sequences:
            sequence_dir = os.path.join(self.root_dir, sequence_name)
            
            # 获取该序列的所有图像文件
            image_files = sorted([
                f for f in os.listdir(sequence_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
            
            # 获取该序列的标注
            seq_id = int(sequence_name)
            sequence_annotations = self.annotations.get(seq_id, [])
            
            # 为每一帧创建索引
            for frame_idx, image_file in enumerate(image_files):
                # 查找对应的标注
                annotation = None
                for anno in sequence_annotations:
                    if anno['frame'] == frame_idx:
                        annotation = anno
                        break
                
                # 如果没有找到标注，创建一个空标注
                if annotation is None:
                    annotation = {
                        'sequence_id': seq_id,
                        'frame': frame_idx,
                        'num_objects': 0,
                        'object_coords': []
                    }
                
                frame_index.append((sequence_name, frame_idx, annotation))
        
        return frame_index

    def __len__(self) -> int:
        """
        返回总帧数。
        """
        return len(self.frame_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        获取指定帧的数据和标注。
        
        Returns: {
            'image': PIL.Image,
            'label': dict,
            'sequence_name': str,
            'frame_idx': int
        }
        """
        sequence_name, frame_idx, annotation = self.frame_index[index]
        
        # 构建图像路径
        sequence_dir = os.path.join(self.root_dir, sequence_name)
        image_files = sorted([
            f for f in os.listdir(sequence_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # 加载图像
        image_path = os.path.join(sequence_dir, image_files[frame_idx])
        image = Image.open(image_path).convert('L')
        
        # 转换为tensor并归一化
        image = torch.from_numpy(np.array(image)).float()
        if image.dim() == 2:  # 如果是2D张量，添加通道维度
            image = image.unsqueeze(0)
        image = image / 255.0  # 归一化到[0,1]
        
        sample = {
            'image': image,
            'label': annotation,
            'sequence_name': sequence_name,
            'frame_idx': frame_idx
        }
        
        # 应用额外的数据增强
        if self.transform:
            if isinstance(self.transform, list):
                # 如果transform是列表，依次应用每个转换
                for t in self.transform:
                    sample = t(sample)
            else:
                # 如果transform是单个转换，直接应用
                sample = self.transform(sample)
            
        return sample

    def get_sequence_info(self, index: int) -> Tuple[str, int]:
        """
        获取指定索引对应的序列信息。
        
        Args:
            index: 帧索引
            
        Returns:
            Tuple[sequence_name, frame_idx]
        """
        sequence_name, frame_idx, _ = self.frame_index[index]
        return sequence_name, frame_idx

    def get_frames_by_sequence(self, sequence_name: str) -> List[int]:
        """
        获取指定序列的所有帧索引。
        
        Args:
            sequence_name: 序列名称
            
        Returns:
            该序列的所有帧在数据集中的索引列表
        """
        indices = []
        for i, (seq_name, frame_idx, _) in enumerate(self.frame_index):
            if seq_name == sequence_name:
                indices.append(i)
        return indices 