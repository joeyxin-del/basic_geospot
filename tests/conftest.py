import os
import pytest
import numpy as np
import torch
from typing import Dict, Any, Generator
from pathlib import Path

# 设置测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """返回测试数据目录路径"""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def device() -> torch.device:
    """返回可用的设备（GPU或CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def random_seed() -> int:
    """设置随机种子，确保测试的可重复性"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

@pytest.fixture(scope="function")
def sample_image() -> torch.Tensor:
    """生成一个示例图像张量"""
    return torch.randn(3, 480, 640)  # 模拟RGB图像

@pytest.fixture(scope="function")
def sample_batch() -> Dict[str, torch.Tensor]:
    """生成一个示例批次数据"""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 480, 640),
        "targets": torch.randn(batch_size, 30, 4),  # 假设每张图像最多30个目标
        "masks": torch.ones(batch_size, 480, 640, dtype=torch.bool)
    }

@pytest.fixture(scope="function")
def sample_config() -> Dict[str, Any]:
    """生成一个示例配置字典"""
    return {
        "model": {
            "name": "test_model",
            "backbone": "resnet50",
            "num_classes": 1
        },
        "dataset": {
            "name": "spotgeov2",
            "root_dir": str(TEST_DATA_DIR / "dataset"),
            "batch_size": 4
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "adam"
        }
    }

class TestDataGenerator:
    """测试数据生成器基类"""
    
    @staticmethod
    def create_dummy_sequence(
        num_frames: int = 5,
        image_size: tuple = (480, 640),
        num_objects: int = 3
    ) -> Dict[str, Any]:
        """创建模拟序列数据"""
        sequence = {
            "images": [torch.randn(3, *image_size) for _ in range(num_frames)],
            "annotations": []
        }
        
        # 为每一帧生成随机目标
        for frame_idx in range(num_frames):
            frame_annos = []
            for obj_idx in range(num_objects):
                # 生成随机边界框 [x1, y1, x2, y2]
                x1 = np.random.randint(0, image_size[1] - 100)
                y1 = np.random.randint(0, image_size[0] - 100)
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(50, 100)
                
                frame_annos.append({
                    "frame": frame_idx,
                    "object_id": obj_idx,
                    "bbox": [x1, y1, x2, y2],
                    "score": np.random.random()
                })
            sequence["annotations"].append(frame_annos)
            
        return sequence

@pytest.fixture(scope="session")
def data_generator() -> TestDataGenerator:
    """返回测试数据生成器实例"""
    return TestDataGenerator()

def assert_tensor_equal(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """断言两个张量相等"""
    assert torch.allclose(t1, t2, rtol=rtol, atol=atol), f"Tensors not equal:\n{t1}\n{t2}"

def assert_dict_structure(d1: Dict, d2: Dict) -> None:
    """断言两个字典具有相同的结构"""
    assert set(d1.keys()) == set(d2.keys()), f"Dictionary keys not equal: {d1.keys()} vs {d2.keys()}"
    for k in d1.keys():
        if isinstance(d1[k], dict):
            assert_dict_structure(d1[k], d2[k])
        elif isinstance(d1[k], (list, tuple)):
            assert len(d1[k]) == len(d2[k]), f"Length mismatch for key {k}: {len(d1[k])} vs {len(d2[k])}"
            if d1[k] and isinstance(d1[k][0], dict):
                for i in range(len(d1[k])):
                    assert_dict_structure(d1[k][i], d2[k][i]) 