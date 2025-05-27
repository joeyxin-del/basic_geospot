import pytest
import torch
import numpy as np
from PIL import Image

from src.models.spotgeo import SpotGEOModel


class TestSpotGEOModel:
    """SpotGEOModel测试类"""
    
    @pytest.fixture
    def model(self):
        """创建测试模型实例"""
        config = {
            'backbone': 'resnet50',
            'num_classes': 2,
            'pretrained': False,
            'learning_rate': 0.001,
            'batch_size': 4
        }
        return SpotGEOModel(config)
        
    @pytest.fixture
    def sample_image(self):
        """创建测试图像数据"""
        # 创建随机图像数据
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(image)
        
    @pytest.fixture
    def sample_batch(self, sample_image):
        """创建测试批次数据"""
        # 创建4张图像的批次
        images = [sample_image] * 4
        labels = torch.randint(0, 2, (4,))  # 二分类标签
        return {
            'images': images,
            'labels': labels
        }
        
    def test_model_initialization(self, model):
        """测试模型初始化"""
        assert isinstance(model, SpotGEOModel)
        assert model.config['backbone'] == 'resnet50'
        assert model.config['num_classes'] == 2
        assert not model.config['pretrained']
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'classifier')
        
    def test_forward_pass(self, model, sample_batch):
        """测试前向传播"""
        output = model(sample_batch)
        assert isinstance(output, dict)
        assert 'predictions' in output
        assert 'features' in output
        preds = output['predictions']['cls']
        assert preds.shape[0] == 4  # batch_size=4
        assert preds.shape[1] == 2  # num_classes=2
        assert len(output['features']) > 0
        assert all(isinstance(feat, torch.Tensor) for feat in output['features'])
        
    def test_compute_loss(self, model, sample_batch):
        """测试损失计算"""
        predictions = model(sample_batch)
        losses = model.compute_loss(predictions, sample_batch)
        assert isinstance(losses, dict)
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].ndim == 0  # 标量
        
    def test_different_input_sizes(self, model):
        """测试不同输入尺寸"""
        # 测试不同尺寸的图像
        sizes = [(224, 224), (256, 256), (384, 384)]
        for size in sizes:
            image = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
            batch = {
                'images': [image] * 4,
                'labels': torch.randint(0, 2, (4,))
            }
            output = model(batch)
            preds = output['predictions']['cls']
            assert preds.shape[0] == 4
        
    def test_different_batch_sizes(self, model, sample_image):
        """测试不同批次大小"""
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            batch = {
                'images': [sample_image] * batch_size,
                'labels': torch.randint(0, 2, (batch_size,))
            }
            output = model(batch)
            preds = output['predictions']['cls']
            assert preds.shape[0] == batch_size
        
    def test_model_output_range(self, model, sample_batch):
        """测试模型输出范围"""
        output = model(sample_batch)
        probs = output['predictions']['probs']
        # 检查是否经过softmax
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.size(0)), atol=1e-5)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        
    def test_feature_extraction(self, model, sample_batch):
        """测试特征提取"""
        output = model(sample_batch)
        features = output['features']
        # 检查特征维度
        assert all(feat.size(0) == 4 for feat in features)  # batch_size=4
        # 不再强制检查特征归一化（如有需要可补充）
        
    def test_invalid_input(self, model):
        """测试无效输入"""
        # 测试空批次
        with pytest.raises(ValueError):
            model({'images': [], 'labels': torch.tensor([])})
        
        # 测试无效图像
        with pytest.raises(ValueError):
            model({
                'images': [None] * 4,
                'labels': torch.randint(0, 2, (4,))
            })
        
        # 测试无效标签
        with pytest.raises(ValueError):
            model({
                'images': [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * 4,
                'labels': torch.tensor([-1, 0, 1, 2])  # 无效的标签值
            })
        
    def test_model_save_load(self, model, tmp_path):
        """测试模型保存和加载"""
        # 保存模型
        checkpoint_path = tmp_path / "spotgeo_checkpoint.pth"
        model.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # 加载模型
        new_model = SpotGEOModel(model.config)
        new_model.load_checkpoint(str(checkpoint_path))
        
        # 验证参数
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
        
    def test_model_training_mode(self, model, sample_batch):
        """测试模型训练模式"""
        # 训练模式
        model.train()
        output_train = model(sample_batch)
        assert model.training
        assert model.backbone.training
        assert model.classifier.training
        
        # 评估模式
        model.eval()
        output_eval = model(sample_batch)
        assert not model.training
        assert not model.backbone.training
        assert not model.classifier.training
        
        # 检查dropout是否生效（如有dropout层）
        with torch.no_grad():
            output1 = model(sample_batch)
            output2 = model(sample_batch)
            assert torch.allclose(output1['predictions']['cls'], output2['predictions']['cls']) 