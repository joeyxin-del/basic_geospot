import os
import json
import pytest
import torch
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from src.datasets.spotgeov2 import SpotGEOv2Dataset
from ...conftest import TestDataGenerator, test_data_dir, data_generator

class TestSpotGEOv2Dataset:
    """SpotGEOv2数据集测试类"""
    
    @pytest.fixture(scope="function")  # 改为function作用域
    def dataset_dir(self, test_data_dir):
        """创建测试数据集目录"""
        # 为每个测试创建独立的目录
        dataset_dir = test_data_dir / f"spotgeov2_{id(self)}"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        return dataset_dir
        
    @pytest.fixture(scope="function")  # 改为function作用域
    def annotation_file(self, dataset_dir):
        """创建测试标注文件"""
        annotation_path = dataset_dir / "test_anno.json"
        
        # 创建测试序列
        sequences = []
        for seq_id in range(2):  # 创建2个序列
            seq_dir = dataset_dir / str(seq_id)
            os.makedirs(seq_dir, exist_ok=True)
            
            # 为每个序列创建5帧图像
            for frame in range(5):
                # 创建空白图像
                img = Image.new('RGB', (640, 480), color='black')
                img_path = seq_dir / f"frame_{frame:04d}.png"
                img.save(img_path)
                
                # 创建标注
                annotation = {
                    "sequence_id": seq_id,
                    "frame": frame,
                    "num_objects": 2,
                    "object_coords": [
                        [100, 100, 200, 200],  # 第一个目标
                        [300, 300, 400, 400]   # 第二个目标
                    ]
                }
                sequences.append(annotation)
                
        # 保存标注文件
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(sequences, f, indent=2)
            
        return annotation_path
        
    @pytest.fixture(scope="function")  # 改为function作用域
    def dataset(self, dataset_dir, annotation_file):
        """创建数据集实例"""
        return SpotGEOv2Dataset(
            root_dir=str(dataset_dir),
            annotation_path=str(annotation_file)
        )
        
    def test_dataset_initialization(self, dataset):
        """测试数据集初始化"""
        assert len(dataset) == 2  # 应该有2个序列
        assert len(dataset.sequences) == 2
        assert len(dataset.annotations) == 2
        
    def test_getitem(self, dataset):
        """测试数据获取"""
        sample = dataset[0]  # 获取第一个序列
        
        # 检查返回的样本结构
        assert "images" in sample
        assert "labels" in sample
        assert "sequence_name" in sample
        
        # 检查图像
        assert len(sample["images"]) == 5  # 5帧
        assert all(isinstance(img, Image.Image) for img in sample["images"])
        assert all(img.size == (640, 480) for img in sample["images"])
        
        # 检查标注
        assert len(sample["labels"]) == 5  # 5帧的标注
        assert all(isinstance(label, dict) for label in sample["labels"])
        assert all("object_coords" in label for label in sample["labels"])
        
    def test_sequence_consistency(self, dataset):
        """测试序列一致性"""
        for i in range(len(dataset)):
            sample = dataset[i]
            sequence_name = sample["sequence_name"]
            
            # 检查序列名称与索引的一致性
            assert int(sequence_name) == i
            
            # 检查标注中的序列ID
            assert all(label["sequence_id"] == i for label in sample["labels"])
            
    def test_annotation_format(self, dataset):
        """测试标注格式"""
        sample = dataset[0]
        label = sample["labels"][0]  # 第一帧的标注
        
        # 检查标注字段
        assert "sequence_id" in label
        assert "frame" in label
        assert "num_objects" in label
        assert "object_coords" in label
        
        # 检查目标坐标格式
        coords = label["object_coords"]
        assert len(coords) == label["num_objects"]
        assert all(len(coord) == 4 for coord in coords)  # 每个目标应该有4个坐标值
        
    def test_transform(self, dataset):
        """测试数据增强"""
        # 创建一个简单的转换函数
        def dummy_transform(sample):
            # 确保图像转换为张量时保持正确的形状
            sample["images"] = [
                torch.from_numpy(np.array(img).transpose(2, 0, 1))  # 转换为CHW格式
                for img in sample["images"]
            ]
            return sample
            
        # 创建带转换的数据集
        dataset_with_transform = SpotGEOv2Dataset(
            root_dir=str(dataset.root_dir),
            annotation_path=str(dataset.annotation_path),
            transform=dummy_transform
        )
        
        # 测试转换后的数据格式
        sample = dataset_with_transform[0]
        assert all(isinstance(img, torch.Tensor) for img in sample["images"])
        assert all(img.shape == (3, 480, 640) for img in sample["images"])
        
    def test_invalid_sequence(self, dataset):
        """测试无效序列处理"""
        with pytest.raises(IndexError):
            _ = dataset[100]  # 访问不存在的序列
            
    def test_empty_sequence(self, test_data_dir):
        """测试空序列处理"""
        # 为这个测试创建独立的目录
        empty_dataset_dir = test_data_dir / f"empty_sequence_{id(self)}"
        if empty_dataset_dir.exists():
            shutil.rmtree(empty_dataset_dir)
        os.makedirs(empty_dataset_dir)
        
        # 创建一个空序列的标注
        empty_annotation = [{
            "sequence_id": 0,  # 使用0作为序列ID
            "frame": 0,
            "num_objects": 0,
            "object_coords": []
        }]
        
        # 创建空序列目录
        empty_seq_dir = empty_dataset_dir / "0"  # 使用0作为目录名
        os.makedirs(empty_seq_dir, exist_ok=True)
        
        # 创建一帧空图像
        img = Image.new('RGB', (640, 480), color='black')
        img_path = empty_seq_dir / "frame_0000.png"
        img.save(img_path)
        
        # 保存空序列标注
        empty_anno_path = empty_dataset_dir / "empty_anno.json"
        with open(empty_anno_path, 'w', encoding='utf-8') as f:
            json.dump(empty_annotation, f)
            
        # 创建数据集实例
        empty_dataset = SpotGEOv2Dataset(
            root_dir=str(empty_dataset_dir),
            annotation_path=str(empty_anno_path)
        )
        
        # 测试空序列
        assert len(empty_dataset) == 1  # 应该有一个序列（空序列）
        sample = empty_dataset[0]
        assert sample["sequence_name"] == "0"  # 检查序列名称
        assert len(sample["images"]) == 1  # 应该有一帧图像
        assert len(sample["labels"]) == 1  # 应该有一帧标注
        assert sample["labels"][0]["num_objects"] == 0  # 标注中应该没有目标
        assert len(sample["labels"][0]["object_coords"]) == 0  # 目标坐标列表应该为空 