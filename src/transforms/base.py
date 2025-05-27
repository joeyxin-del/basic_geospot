from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image


class BaseTransform(ABC):
    """
    数据增强基类，定义所有数据增强必须实现的接口。
    所有具体的数据增强实现都应该继承这个基类。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据增强基类。
        
        Args:
            config: 数据增强配置字典，包含数据增强的各种超参数
        """
        self.config = config or {}
        
    @abstractmethod
    def __call__(self, data: Union[Image.Image, np.ndarray, torch.Tensor, Dict[str, Any]]) -> Union[Image.Image, np.ndarray, torch.Tensor, Dict[str, Any]]:
        """
        应用数据增强。
        
        Args:
            data: 输入数据，可以是以下类型之一：
                - PIL.Image：图像数据
                - np.ndarray：numpy数组
                - torch.Tensor：PyTorch张量
                - Dict[str, Any]：包含多个数据的字典
                
        Returns:
            增强后的数据，类型与输入相同
        """
        pass
    
    def _check_input_type(self, data: Any) -> str:
        """
        检查输入数据类型。
        
        Args:
            data: 输入数据
            
        Returns:
            数据类型字符串，可以是以下之一：
            - 'pil'：PIL.Image
            - 'numpy'：np.ndarray
            - 'torch'：torch.Tensor
            - 'dict'：Dict[str, Any]
            
        Raises:
            TypeError: 如果输入数据类型不支持
        """
        if isinstance(data, Image.Image):
            return 'pil'
        elif isinstance(data, np.ndarray):
            return 'numpy'
        elif isinstance(data, torch.Tensor):
            return 'torch'
        elif isinstance(data, dict):
            return 'dict'
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
            
    def _to_tensor(self, data: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        将输入数据转换为PyTorch张量。
        
        Args:
            data: 输入数据，可以是PIL.Image、np.ndarray或torch.Tensor
            
        Returns:
            PyTorch张量
        """
        if isinstance(data, Image.Image):
            return torch.from_numpy(np.array(data))
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
            
    def _to_numpy(self, data: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        将输入数据转换为numpy数组。
        
        Args:
            data: 输入数据，可以是PIL.Image、np.ndarray或torch.Tensor
            
        Returns:
            numpy数组
        """
        if isinstance(data, Image.Image):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")
            
    def _to_pil(self, data: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        将输入数据转换为PIL图像。
        
        Args:
            data: 输入数据，可以是PIL.Image、np.ndarray或torch.Tensor
            
        Returns:
            PIL图像
            
        Raises:
            ValueError: 如果输入数据无法转换为PIL图像
        """
        if isinstance(data, Image.Image):
            return data
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:  # 灰度图
                return Image.fromarray(data.astype(np.uint8))
            elif data.ndim == 3:  # RGB图
                if data.shape[2] == 3:
                    return Image.fromarray(data.astype(np.uint8))
                elif data.shape[2] == 1:
                    return Image.fromarray(data[:, :, 0].astype(np.uint8))
            raise ValueError(f"Invalid numpy array shape: {data.shape}")
        elif isinstance(data, torch.Tensor):
            if data.ndim == 2:  # 灰度图
                return Image.fromarray(data.cpu().numpy().astype(np.uint8))
            elif data.ndim == 3:  # RGB图
                if data.shape[0] == 3:
                    return Image.fromarray(data.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                elif data.shape[0] == 1:
                    return Image.fromarray(data[0].cpu().numpy().astype(np.uint8))
            raise ValueError(f"Invalid tensor shape: {data.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")


class Compose:
    """
    组合多个数据增强操作，按顺序依次应用。
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class ToTensor:
    """
    将PIL图像或numpy数组转换为PyTorch张量，输出为float32且归一化到[0,1]。
    支持以下输入类型：
    - PIL.Image：RGB或灰度图像
    - np.ndarray：形状为(H,W,C)或(H,W)的数组
    - torch.Tensor：形状为(C,H,W)或(H,W)的张量
    """
    def __call__(self, data):
        import numpy as np
        import torch
        from PIL import Image
        
        if isinstance(data, dict) and "images" in data:
            # 处理字典类型输入
            data = data.copy()
            data["images"] = [self.__call__(img) for img in data["images"]]
            return data
        elif isinstance(data, Image.Image):
            # 处理PIL图像
            arr = np.array(data)
            if arr.ndim == 2:  # 灰度图
                arr = arr[:, :, None]  # 添加通道维度
            arr = arr.transpose((2, 0, 1))  # HWC->CHW
            tensor = torch.from_numpy(arr).float() / 255.0
            return tensor
        elif isinstance(data, np.ndarray):
            # 处理numpy数组
            if data.ndim == 2:  # 灰度图
                data = data[:, :, None]  # 添加通道维度
            elif data.ndim == 3 and data.shape[2] == 1:  # 单通道图
                pass
            elif data.ndim == 3 and data.shape[2] == 3:  # RGB图
                pass
            else:
                raise ValueError(f"不支持的numpy数组形状: {data.shape}")
            data = data.transpose((2, 0, 1))  # HWC->CHW
            tensor = torch.from_numpy(data).float() / 255.0
            return tensor
        elif isinstance(data, torch.Tensor):
            # 处理PyTorch张量
            if data.ndim == 2:  # 灰度图
                data = data.unsqueeze(0)  # 添加通道维度
            elif data.ndim == 3 and data.shape[0] == 1:  # 单通道图
                pass
            elif data.ndim == 3 and data.shape[0] == 3:  # RGB图
                pass
            else:
                raise ValueError(f"不支持的张量形状: {data.shape}")
            return data.float() / 255.0 if data.max() > 1 else data.float()
        else:
            raise ValueError('ToTensor只支持PIL.Image、np.ndarray、torch.Tensor或包含images字段的dict')

class Normalize:
    """
    对张量进行归一化处理，返回新张量。
    输入张量应该在[0,1]范围内，输出张量将根据mean和std进行标准化。
    标准化后的张量均值接近0，标准差接近1。
    """
    def __init__(self, mean, std):
        """
        初始化标准化转换。
        
        Args:
            mean: 每个通道的均值
            std: 每个通道的标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, data):
        import torch
        
        if isinstance(data, dict) and "images" in data:
            # 处理字典类型输入
            data = data.copy()
            data["images"] = [self.__call__(img) for img in data["images"]]
            return data
            
        if not isinstance(data, torch.Tensor):
            raise ValueError('Normalize只支持torch.Tensor或包含images字段的dict')
            
        # 检查输入张量维度
        if data.ndim == 2:  # (H, W)
            raise ValueError('Normalize不支持2D张量，需要3D张量(C,H,W)或4D张量(B,C,H,W)')
            
        # 确保输入张量在[0,1]范围内
        if data.max() > 1.0:
            data = data / 255.0
            
        # 转换为float类型
        data = data.float()
        
        # 创建mean和std张量，并调整维度以匹配输入张量
        mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.tensor(self.std, dtype=data.dtype, device=data.device)
        
        # 检查通道数是否匹配
        if data.shape[-3] != len(self.mean):
            raise ValueError(f'输入张量的通道数({data.shape[-3]})与mean/std的长度({len(self.mean)})不匹配')
        
        # 调整维度以匹配输入张量的通道数
        if data.ndim == 3:  # (C, H, W)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif data.ndim == 4:  # (B, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
            
        # 执行标准化
        normalized = (data - mean) / std
        
        return normalized

class Resize:
    """
    调整PIL图像或numpy数组的尺寸。
    支持dict类型，递归处理images字段。
    """
    def __init__(self, size):
        self.size = size  # (width, height) 或 int

    def __call__(self, data):
        from PIL import Image
        import numpy as np
        if isinstance(data, Image.Image):
            return data.resize(self.size, Image.BILINEAR)
        elif isinstance(data, np.ndarray):
            img = Image.fromarray(data)
            return np.array(img.resize(self.size, Image.BILINEAR))
        elif isinstance(data, dict) and "images" in data:
            data = data.copy()
            data["images"] = [self.__call__(img) for img in data["images"]]
            return data
        else:
            raise ValueError('Resize只支持PIL.Image、np.ndarray或包含images字段的dict')


# 导出基类
__all__ = ['BaseTransform'] 