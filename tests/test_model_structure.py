import os
import torch
import pytest
from torch import nn
from src.models import get_model, list_models
from src.utils import get_logger

logger = get_logger('model_test')

class TestModelStructure:
    """模型结构测试类"""
    
    @pytest.fixture
    def device(self):
        """获取设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_model_registration(self):
        """测试模型注册"""
        # 获取所有已注册的模型
        available_models = list_models()
        logger.info(f"Available models: {available_models}")
        assert len(available_models) > 0, "没有找到已注册的模型"
        
    def test_model_creation(self, device):
        """测试模型创建和基本结构"""
        # 获取所有已注册的模型
        available_models = list_models()
        
        for model_name in available_models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing model: {model_name}")
            
            try:
                # 创建模型实例
                model = get_model(model_name)
                model = model.to(device)
                model.eval()  # 设置为评估模式
                
                # 打印模型结构
                logger.info("\nModel Structure:")
                logger.info("-" * 30)
                logger.info(str(model))
                
                # 打印模型参数统计
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"\nModel Parameters:")
                logger.info(f"Total parameters: {total_params:,}")
                logger.info(f"Trainable parameters: {trainable_params:,}")
                logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
                
                # 测试模型输入输出
                self._test_model_io(model, model_name, device)
                
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                raise
                
    def _test_model_io(self, model: nn.Module, model_name: str, device: torch.device):
        """测试模型输入输出
        
        Args:
            model: 模型实例
            model_name: 模型名称
            device: 设备
        """
        logger.info("\nTesting Model Input/Output:")
        logger.info("-" * 30)
        
        # 根据模型类型设置不同的输入
        if "spotgeo" in model_name.lower():
            # SpotGEO模型输入：序列图像
            batch_size = 2
            seq_length = 5  # 序列长度
            channels = 3    # RGB图像
            height = 480    # 图像高度
            width = 640     # 图像宽度
            
            # 创建输入张量
            x = torch.randn(batch_size, seq_length, channels, height, width).to(device)
            logger.info(f"Input shape: {x.shape}")
            
            # 前向传播
            with torch.no_grad():
                try:
                    output = model(x)
                    
                    # 打印输出信息
                    if isinstance(output, dict):
                        logger.info("\nOutput dictionary:")
                        for key, value in output.items():
                            if isinstance(value, torch.Tensor):
                                logger.info(f"{key}: shape {value.shape}, dtype {value.dtype}")
                            else:
                                logger.info(f"{key}: {type(value)}")
                    else:
                        logger.info(f"\nOutput: shape {output.shape}, dtype {output.dtype}")
                        
                    # 测试模型保存和加载
                    self._test_model_save_load(model, model_name, device)
                    
                except Exception as e:
                    logger.error(f"Error in forward pass: {str(e)}")
                    raise
                    
    def _test_model_save_load(self, model: nn.Module, model_name: str, device: torch.device):
        """测试模型保存和加载
        
        Args:
            model: 模型实例
            model_name: 模型名称
            device: 设备
        """
        logger.info("\nTesting Model Save/Load:")
        logger.info("-" * 30)
        
        # 创建保存目录
        save_dir = "test_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        save_path = os.path.join(save_dir, f"{model_name}_test.pth")
        try:
            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.config if hasattr(model, 'config') else None
            }, save_path)
            logger.info(f"Model saved to: {save_path}")
            
            # 加载模型
            checkpoint = torch.load(save_path, map_location=device)
            new_model = get_model(model_name)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_model = new_model.to(device)
            new_model.eval()
            
            logger.info("Model loaded successfully")
            
            # 清理测试文件
            os.remove(save_path)
            logger.info(f"Test checkpoint removed: {save_path}")
            
        except Exception as e:
            logger.error(f"Error in save/load test: {str(e)}")
            raise
            
    def test_model_gradients(self, device):
        """测试模型梯度计算"""
        logger.info("\nTesting Model Gradients:")
        logger.info("-" * 30)
        
        # 获取所有已注册的模型
        available_models = list_models()
        
        for model_name in available_models:
            try:
                # 创建模型实例
                model = get_model(model_name)
                model = model.to(device)
                model.train()  # 设置为训练模式
                
                # 创建输入
                if "spotgeo" in model_name.lower():
                    x = torch.randn(2, 5, 3, 480, 640).to(device)
                    x.requires_grad = True
                else:
                    continue  # 跳过其他类型的模型
                    
                # 前向传播
                output = model(x)
                
                # 计算损失（使用一个简单的损失函数）
                if isinstance(output, dict):
                    # 如果输出是字典，使用第一个张量
                    loss = output[list(output.keys())[0]].mean()
                else:
                    loss = output.mean()
                    
                # 反向传播
                loss.backward()
                
                # 检查梯度
                has_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_grad = True
                        grad_norm = param.grad.norm().item()
                        logger.info(f"Parameter {name}: gradient norm = {grad_norm:.6f}")
                        
                assert has_grad, f"Model {model_name} has no gradients"
                logger.info(f"Model {model_name} gradient test passed")
                
            except Exception as e:
                logger.error(f"Error testing gradients for {model_name}: {str(e)}")
                raise
                
    def test_model_config(self):
        """测试模型配置"""
        logger.info("\nTesting Model Configurations:")
        logger.info("-" * 30)
        
        # 获取所有已注册的模型
        available_models = list_models()
        
        for model_name in available_models:
            try:
                # 创建模型实例
                model = get_model(model_name)
                
                # 检查模型配置
                if hasattr(model, 'config'):
                    logger.info(f"\nModel {model_name} configuration:")
                    for key, value in model.config.items():
                        logger.info(f"{key}: {value}")
                else:
                    logger.info(f"\nModel {model_name} has no configuration")
                    
            except Exception as e:
                logger.error(f"Error checking config for {model_name}: {str(e)}")
                raise 