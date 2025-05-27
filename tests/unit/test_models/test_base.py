import pytest
import torch
import os
from pathlib import Path

from src.models.base import BaseModel


class DummyModel(BaseModel):
    """用于测试的虚拟模型"""
    def __init__(self, config=None):
        super().__init__(config)
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return {
            'predictions': self.linear(x),
            'features': [x]
        }
        
    def compute_loss(self, predictions, targets):
        return {
            'total_loss': torch.nn.functional.mse_loss(predictions['predictions'], targets['targets'])
        }


class TestBaseModel:
    """基础模型测试类"""
    
    @pytest.fixture
    def model(self):
        """创建测试模型实例"""
        config = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        return DummyModel(config)
        
    @pytest.fixture
    def sample_input(self):
        """创建测试输入数据"""
        return torch.randn(4, 10)  # batch_size=4, input_dim=10
        
    @pytest.fixture
    def sample_target(self):
        """创建测试目标数据"""
        return {
            'targets': torch.randn(4, 2)  # batch_size=4, output_dim=2
        }
        
    def test_model_initialization(self, model):
        """测试模型初始化"""
        assert isinstance(model, BaseModel)
        assert model.config['learning_rate'] == 0.001
        assert model.config['batch_size'] == 32
        assert hasattr(model, 'linear')
        
    def test_forward_pass(self, model, sample_input):
        """测试前向传播"""
        output = model(sample_input)
        assert isinstance(output, dict)
        assert 'predictions' in output
        assert 'features' in output
        assert output['predictions'].shape == (4, 2)
        assert len(output['features']) == 1
        assert output['features'][0].shape == (4, 10)
        
    def test_compute_loss(self, model, sample_input, sample_target):
        """测试损失计算"""
        predictions = model(sample_input)
        losses = model.compute_loss(predictions, sample_target)
        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].ndim == 0  # 标量
        
    def test_save_load_checkpoint(self, model, tmp_path):
        """测试保存和加载检查点"""
        # 保存检查点
        checkpoint_path = tmp_path / "model_checkpoint.pth"
        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # 创建新模型并加载检查点
        new_model = DummyModel()
        new_model.load_checkpoint(str(checkpoint_path))
        
        # 验证配置和参数是否一致
        assert new_model.config == model.config
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
            
    def test_get_trainable_params(self, model):
        """测试获取可训练参数"""
        params = model.get_trainable_params()
        assert isinstance(params, dict)
        assert all(isinstance(param, torch.Tensor) for param in params.values())
        assert all(param.requires_grad for param in params.values())
        
    def test_freeze_unfreeze_layers(self, model):
        """测试参数冻结和解冻"""
        # 冻结linear层
        model.freeze_layers(['linear'])
        for name, param in model.named_parameters():
            if 'linear' in name:
                assert not param.requires_grad
                
        # 解冻linear层
        model.unfreeze_layers(['linear'])
        for name, param in model.named_parameters():
            if 'linear' in name:
                assert param.requires_grad
                
    def test_invalid_checkpoint(self, model):
        """测试无效检查点处理"""
        with pytest.raises(FileNotFoundError):
            model.load_checkpoint("nonexistent_checkpoint.pth")
            
    def test_invalid_input_shape(self, model):
        """测试无效输入形状"""
        invalid_input = torch.randn(4, 5)  # 错误的输入维度
        with pytest.raises(RuntimeError):
            model(invalid_input)
            
    def test_invalid_target_shape(self, model, sample_input):
        """测试无效目标形状"""
        predictions = model(sample_input)
        invalid_target = {
            'targets': torch.randn(4, 3)  # 错误的输出维度
        }
        with pytest.raises(RuntimeError):
            model.compute_loss(predictions, invalid_target) 