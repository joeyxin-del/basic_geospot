import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation
)

class TestTransforms:
    """数据增强模块测试类"""
    
    @pytest.fixture
    def sample_image(self):
        """创建测试图像"""
        # 创建一个3通道的测试图像，尺寸为640x480
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
        
    @pytest.fixture
    def sample_tensor(self):
        """创建测试张量"""
        # 创建一个3通道的测试张量，尺寸为3x480x640
        return torch.randn(3, 480, 640)
        
    def test_compose(self, sample_image):
        """测试组合转换"""
        transforms = Compose([
            Resize((320, 240)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transformed = transforms(sample_image)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 240, 320)
        
    def test_to_tensor(self, sample_image):
        """测试转换为张量"""
        transform = ToTensor()
        transformed = transform(sample_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 480, 640)
        assert transformed.dtype == torch.float32
        assert transformed.min() >= 0 and transformed.max() <= 1
        
    def test_normalize(self, sample_tensor):
        """测试标准化"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Normalize(mean=mean, std=std)
        
        transformed = transform(sample_tensor)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == sample_tensor.shape
        
        # 检查每个通道的均值和标准差
        for i in range(3):
            channel = transformed[i]
            assert abs(channel.mean().item()) < 0.1  # 均值应该接近0
            assert abs(channel.std().item() - 1.0) < 0.1  # 标准差应该接近1
            
    def test_resize(self, sample_image):
        """测试调整大小"""
        target_size = (320, 240)
        transform = Resize(target_size)
        
        transformed = transform(sample_image)
        assert isinstance(transformed, Image.Image)
        assert transformed.size == target_size
        
    def test_random_horizontal_flip(self, sample_image):
        """测试随机水平翻转"""
        transform = RandomHorizontalFlip(p=1.0)  # 设置概率为1确保翻转
        sample = {
            "images": [sample_image],
            "labels": [{
                "object_coords": [[100, 100, 200, 200]]  # [x1, y1, x2, y2]
            }]
        }
        transformed = transform(sample)
        assert isinstance(transformed["images"][0], Image.Image)
        assert transformed["images"][0].size == sample_image.size
        original_box = sample["labels"][0]["object_coords"][0]
        flipped_box = transformed["labels"][0]["object_coords"][0]
        w = sample_image.width
        assert flipped_box[0] == w - 1 - original_box[2]  # x1
        assert flipped_box[2] == w - 1 - original_box[0]  # x2
        assert flipped_box[1] == original_box[1]  # y1
        assert flipped_box[3] == original_box[3]  # y2
        
    def test_random_vertical_flip(self, sample_image):
        """测试随机垂直翻转"""
        transform = RandomVerticalFlip(p=1.0)  # 设置概率为1确保翻转
        sample = {
            "images": [sample_image],
            "labels": [{
                "object_coords": [[100, 100, 200, 200]]  # [x1, y1, x2, y2]
            }]
        }
        transformed = transform(sample)
        assert isinstance(transformed["images"][0], Image.Image)
        assert transformed["images"][0].size == sample_image.size
        original_box = sample["labels"][0]["object_coords"][0]
        flipped_box = transformed["labels"][0]["object_coords"][0]
        h = sample_image.height
        assert flipped_box[0] == original_box[0]  # x1
        assert flipped_box[2] == original_box[2]  # x2
        assert flipped_box[1] == h - 1 - original_box[3]  # y1
        assert flipped_box[3] == h - 1 - original_box[1]  # y2
        
    def test_random_rotation(self, sample_image):
        """测试随机旋转"""
        transform = RandomRotation(degrees=90, p=1.0)  # 设置概率为1确保旋转
        
        # 创建一个带有目标框的图像数据
        sample = {
            "images": [sample_image],
            "labels": [{
                "object_coords": [[100, 100, 200, 200]]  # [x1, y1, x2, y2]
            }]
        }
        
        transformed = transform(sample)
        assert isinstance(transformed["images"][0], Image.Image)
        assert transformed["images"][0].size == sample_image.size
        
        # 检查目标框是否在有效范围内
        flipped_box = transformed["labels"][0]["object_coords"][0]
        assert all(0 <= coord <= max(sample_image.size) for coord in flipped_box)
        assert flipped_box[0] <= flipped_box[2]  # x1 <= x2
        assert flipped_box[1] <= flipped_box[3]  # y1 <= y2
        
    def test_transform_chain(self, sample_image):
        """测试转换链"""
        transforms = Compose([
            Resize((320, 240)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=30, p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建一个带有目标框的图像数据
        sample = {
            "images": [sample_image],
            "labels": [{
                "object_coords": [[100, 100, 200, 200]]  # [x1, y1, x2, y2]
            }]
        }
        
        transformed = transforms(sample)
        assert isinstance(transformed["images"][0], torch.Tensor)
        assert transformed["images"][0].shape == (3, 240, 320)
        assert "labels" in transformed
        assert "object_coords" in transformed["labels"][0]
        
    def test_invalid_input(self):
        """测试无效输入处理"""
        # 测试ToTensor的无效输入
        transform = ToTensor()
        
        # 测试None输入
        with pytest.raises(ValueError):
            transform(None)
            
        # 测试无效图像类型
        with pytest.raises(ValueError):
            transform("invalid_image")
            
        # 测试无效numpy数组形状
        with pytest.raises(ValueError):
            transform(np.zeros((100, 100, 4)))  # 4通道图像
            
        # 测试无效张量形状
        with pytest.raises(ValueError):
            transform(torch.zeros((4, 100, 100)))  # 4通道张量
            
        # 测试Normalize的无效输入
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 测试非张量输入
        with pytest.raises(ValueError):
            normalize(np.zeros((3, 100, 100)))
            
        # 测试无效张量形状
        with pytest.raises(ValueError):
            normalize(torch.zeros((100, 100)))  # 缺少通道维度
            
        # 测试无效字典输入
        with pytest.raises(ValueError):
            normalize({"invalid_key": []})
            
        # 测试RandomHorizontalFlip的无效输入
        flip = RandomHorizontalFlip(p=1.0)
        
        # 测试无效输入类型
        with pytest.raises(ValueError):
            flip(np.zeros((100, 100, 3)))
            
        # 测试无效字典输入
        with pytest.raises(ValueError):
            flip({"invalid_key": []})
            
        # 测试RandomVerticalFlip的无效输入
        flip = RandomVerticalFlip(p=1.0)
        
        # 测试无效输入类型
        with pytest.raises(ValueError):
            flip(np.zeros((100, 100, 3)))
            
        # 测试无效字典输入
        with pytest.raises(ValueError):
            flip({"invalid_key": []})
            
        # 测试RandomRotation的无效输入
        rotation = RandomRotation(degrees=30, p=1.0)
        
        # 测试无效输入类型
        with pytest.raises(ValueError):
            rotation("invalid_image")
            
        # 测试无效字典输入
        with pytest.raises(ValueError):
            rotation({"invalid_key": []}) 