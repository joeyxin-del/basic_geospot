import os
import json
from typing import Any, Dict, List, Optional
from PIL import Image
from .base import DatasetBase

class SpotGEOv2Dataset(DatasetBase):
    """
    SpotGEOv2序列数据集适配类。
    支持序列样本读取、标注解析，适用于空间暗弱目标检测任务。
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
        :param transform: 数据增强方法
        :param config: 其他配置参数
        """
        super().__init__(config or {})
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.transform = transform
        self.sequences = self._load_sequences()
        self.annotations = self._load_annotations()

    def _load_sequences(self) -> List[str]:
        """
        加载所有序列文件夹名称。
        """
        return [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

    def _load_annotations(self) -> List[Dict[str, Any]]:
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
        return sequence_annotations

    def __len__(self) -> int:
        """
        返回序列数量。
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        获取指定序列的数据和标注。
        返回：{
            'images': List[PIL.Image],
            'labels': List[dict],
            'sequence_name': str
        }
        """
        sequence_name = self.sequences[index]
        sequence_dir = os.path.join(self.root_dir, sequence_name)
        image_files = sorted([
            os.path.join(sequence_dir, f)
            for f in os.listdir(sequence_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        images = [Image.open(img_path).convert('RGB') for img_path in image_files]
        # 获取该序列的所有标注
        sequence_labels = self.annotations.get(int(sequence_name), [])
        # 按帧号排序标注
        sequence_labels.sort(key=lambda x: x['frame'])
        sample = {
            'images': images,
            'labels': sequence_labels,
            'sequence_name': sequence_name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample 