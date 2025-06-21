import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import tempfile
import shutil
from pathlib import Path

from src.training.trainer import Trainer

class DummyModel(nn.Module):
    """用于测试的虚拟模型"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        return {
            'predictions': {
                'cls': self.conv(x),
                'reg': self.reg_conv(x)
            }
        }
    
    def compute_loss(self, outputs, targets):
        """计算损失"""
        cls_loss = torch.mean((outputs['predictions']['cls'] - targets['cls']) ** 2)
        reg_loss = torch.mean((outputs['predictions']['reg'] - targets['reg']) ** 2)
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'total_loss': cls_loss + reg_loss
        }

class DummyDataset(Dataset):
    """用于测试的虚拟数据集"""
    def __init__(self, num_sequences=2, frames_per_sequence=3):
        self.num_sequences = num_sequences
        self.frames_per_sequence = frames_per_sequence
        
    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        # 创建一个序列的图像和标签
        # 注意：这里直接返回张量，而不是PIL图像
        images = torch.stack([torch.randn(3, 480, 640) for _ in range(self.frames_per_sequence)])
        labels = [
            {
                'frame': i,
                'num_objects': 2,
                'object_coords': [(100, 100), (200, 200)]
            }
            for i in range(self.frames_per_sequence)
        ]
        
        return {
            'images': images,  # 现在是 [frames, channels, height, width]
            'labels': labels,
            'sequence_name': str(idx)
        }

@pytest.fixture
def temp_output_dir():
    """创建临时输出目录"""
    temp_dir = tempfile.mkdtemp()
    # 预先创建所需的子目录
    os.makedirs(os.path.join(temp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'plots'), exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def trainer_components(device):
    """准备训练器所需的组件"""
    model = DummyModel().to(device)
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    
    return {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

def test_trainer_initialization(trainer_components, temp_output_dir, device):
    """测试训练器的初始化"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        scheduler=trainer_components['scheduler'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False,
        max_epochs=2
    )
    
    assert trainer.current_epoch == 0
    assert trainer.best_score == float('inf')
    assert trainer.device == device
    assert os.path.exists(os.path.join(trainer.output_dir, 'checkpoints'))
    assert os.path.exists(os.path.join(trainer.output_dir, 'plots'))

def test_create_target_tensors(trainer_components, temp_output_dir, device):
    """测试目标张量创建功能"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False
    )
    
    labels = [
        {
            'frame': 0,
            'num_objects': 1,
            'object_coords': [(320, 240)]  # 图像中心
        }
    ]
    
    image_size = (60, 80)  # 缩小的热图尺寸
    targets = trainer._create_target_tensors(labels, image_size)
    
    assert 'cls' in targets
    assert 'reg' in targets
    assert targets['cls'].shape == (1, 1, 60, 80)
    assert targets['reg'].shape == (1, 2, 60, 80)

def test_save_load_checkpoint(trainer_components, temp_output_dir, device):
    """测试检查点的保存和加载"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        scheduler=trainer_components['scheduler'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False
    )
    
    # 保存检查点
    trainer.current_epoch = 1
    trainer.best_score = 0.5
    trainer._save_checkpoint(epoch=1, is_best=True)
    
    # 创建新的训练器并加载检查点
    checkpoint_path = os.path.join(trainer.output_dir, 'best.pth')
    assert os.path.exists(checkpoint_path), f"Checkpoint file not found at {checkpoint_path}"
    
    new_trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        scheduler=trainer_components['scheduler'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False,
        resume=checkpoint_path
    )
    
    assert new_trainer.current_epoch == 1
    assert new_trainer.best_score == 0.5

def test_train_epoch(trainer_components, temp_output_dir, device):
    """测试单个训练epoch"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False
    )
    
    metrics = trainer._train_epoch()
    assert 'loss' in metrics
    assert 'time' in metrics
    assert 'lr' in metrics
    assert metrics['loss'] >= 0

def test_validate_epoch(trainer_components, temp_output_dir, device):
    """测试单个验证epoch"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False
    )
    
    metrics = trainer._validate_epoch()
    assert 'loss' in metrics
    assert 'f1_score' in metrics
    assert 'mse' in metrics

def test_full_training_loop(trainer_components, temp_output_dir, device):
    """测试完整的训练循环"""
    trainer = Trainer(
        model=trainer_components['model'],
        train_loader=trainer_components['train_loader'],
        val_loader=trainer_components['val_loader'],
        optimizer=trainer_components['optimizer'],
        scheduler=trainer_components['scheduler'],
        device=device,
        output_dir=temp_output_dir,
        use_swanlab=False,
        max_epochs=2,
        checkpoint_interval=1,
        eval_interval=1
    )
    
    results = trainer.train()
    assert 'best_epoch' in results
    assert 'best_score' in results
    assert 'total_epochs' in results
    assert os.path.exists(os.path.join(trainer.output_dir, 'results.json')) 