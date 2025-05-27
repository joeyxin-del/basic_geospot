import random
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .base import BaseTransform
from .factory import TransformFactory


@TransformFactory.register('random_crop')
class RandomCrop(BaseTransform):
    """
    随机裁剪数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化随机裁剪。
        
        Args:
            config: 配置字典，包含：
                - size: 裁剪大小，可以是整数或元组 (height, width)
                - padding: 填充大小，可以是整数或元组 (top, bottom, left, right)
        """
        super().__init__(config)
        self.size = self.config.get('size', 224)
        self.padding = self.config.get('padding', 0)
        
        # 确保size是元组
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
            
        # 确保padding是元组
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding, self.padding, self.padding)
            
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        执行随机裁剪。
        
        Args:
            image: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
            
        Returns:
            裁剪后的图像，保持与输入相同的类型
        """
        # 转换为PIL图像进行处理
        pil_image = self.to_pil(image)
        
        # 添加填充
        if any(self.padding):
            pil_image = Image.new(pil_image.mode, 
                                (pil_image.width + self.padding[2] + self.padding[3],
                                 pil_image.height + self.padding[0] + self.padding[1]),
                                (0, 0, 0))
            pil_image.paste(image, (self.padding[2], self.padding[0]))
            
        # 计算裁剪区域
        width, height = pil_image.size
        target_height, target_width = self.size
        
        # 确保目标尺寸不超过图像尺寸
        target_height = min(target_height, height)
        target_width = min(target_width, width)
        
        # 随机选择裁剪位置
        top = random.randint(0, height - target_height)
        left = random.randint(0, width - target_width)
        
        # 执行裁剪
        cropped = pil_image.crop((left, top, left + target_width, top + target_height))
        
        # 转换回原始类型
        return self.to_original_type(cropped, image)


@TransformFactory.register('random_flip')
class RandomFlip(BaseTransform):
    """
    随机翻转数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化随机翻转。
        
        Args:
            config: 配置字典，包含：
                - horizontal_prob: 水平翻转概率
                - vertical_prob: 垂直翻转概率
        """
        super().__init__(config)
        self.horizontal_prob = self.config.get('horizontal_prob', 0.5)
        self.vertical_prob = self.config.get('vertical_prob', 0.5)
        
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        执行随机翻转。
        
        Args:
            image: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
            
        Returns:
            翻转后的图像，保持与输入相同的类型
        """
        # 转换为PIL图像进行处理
        pil_image = self.to_pil(image)
        
        # 随机水平翻转
        if random.random() < self.horizontal_prob:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            
        # 随机垂直翻转
        if random.random() < self.vertical_prob:
            pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            
        # 转换回原始类型
        return self.to_original_type(pil_image, image)


@TransformFactory.register('random_rotation')
class RandomRotation(BaseTransform):
    """
    随机旋转数据增强。
    对于目标框坐标，会根据旋转角度进行相应的变换。
    """
    def __init__(self, degrees=30, p=0.5, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.p = p
        self.degrees = degrees
        self.expand = False
        self.fill = 0
        if config is not None:
            self.expand = config.get('expand', False)
            self.fill = config.get('fill', 0)
        # 确保degrees是元组
        if isinstance(self.degrees, (int, float)):
            self.degrees = (-self.degrees, self.degrees)

    def __call__(self, data: Union[Image.Image, np.ndarray, torch.Tensor, Dict[str, Any]]) -> Union[Image.Image, np.ndarray, torch.Tensor, Dict[str, Any]]:
        if isinstance(data, dict) and "images" in data:
            data = data.copy()
            if random.random() < self.p:
                # 获取图像尺寸
                w, h = data["images"][0].size
                # 计算旋转中心
                center = (w / 2, h / 2)
                # 随机选择旋转角度
                angle = random.uniform(self.degrees[0], self.degrees[1])
                # 旋转图像
                data["images"] = [img.rotate(angle, expand=self.expand, fillcolor=self.fill) for img in data["images"]]
                # 更新标签坐标
                if "labels" in data:
                    for label in data["labels"]:
                        if "object_coords" in label:
                            new_boxes = []
                            for box in label["object_coords"]:
                                x1, y1, x2, y2 = box
                                # 计算旋转后的坐标
                                # 将坐标转换为相对于中心点的坐标
                                cx1, cy1 = x1 - center[0], y1 - center[1]
                                cx2, cy2 = x2 - center[0], y2 - center[1]
                                # 旋转坐标
                                angle_rad = math.radians(angle)
                                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                                rx1 = cx1 * cos_a - cy1 * sin_a + center[0]
                                ry1 = cx1 * sin_a + cy1 * cos_a + center[1]
                                rx2 = cx2 * cos_a - cy2 * sin_a + center[0]
                                ry2 = cx2 * sin_a + cy2 * cos_a + center[1]
                                # 确保坐标在有效范围内
                                rx1 = max(0, min(w - 1, rx1))
                                ry1 = max(0, min(h - 1, ry1))
                                rx2 = max(0, min(w - 1, rx2))
                                ry2 = max(0, min(h - 1, ry2))
                                # 确保x1 <= x2, y1 <= y2
                                rx1, rx2 = min(rx1, rx2), max(rx1, rx2)
                                ry1, ry2 = min(ry1, ry2), max(ry1, ry2)
                                new_boxes.append([rx1, ry1, rx2, ry2])
                            label["object_coords"] = new_boxes
            return data
        else:
            # 转换为PIL图像进行处理
            pil_image = self._to_pil(data)
            if random.random() < self.p:
                angle = random.uniform(self.degrees[0], self.degrees[1])
                rotated = pil_image.rotate(angle, expand=self.expand, fillcolor=self.fill)
                return self._to_original_type(rotated, data)
            return data


@TransformFactory.register('color_jitter')
class ColorJitter(BaseTransform):
    """
    颜色抖动数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化颜色抖动。
        
        Args:
            config: 配置字典，包含：
                - brightness: 亮度调整范围，可以是单个数字或元组 (min_factor, max_factor)
                - contrast: 对比度调整范围
                - saturation: 饱和度调整范围
                - hue: 色调调整范围
        """
        super().__init__(config)
        self.brightness = self.config.get('brightness', 0.2)
        self.contrast = self.config.get('contrast', 0.2)
        self.saturation = self.config.get('saturation', 0.2)
        self.hue = self.config.get('hue', 0.1)
        
        # 确保参数是元组
        for param in ['brightness', 'contrast', 'saturation', 'hue']:
            value = getattr(self, param)
            if isinstance(value, (int, float)):
                setattr(self, param, (max(0, 1 - value), 1 + value))
                
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        执行颜色抖动。
        
        Args:
            image: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
            
        Returns:
            颜色抖动后的图像，保持与输入相同的类型
        """
        # 转换为PIL图像进行处理
        pil_image = self.to_pil(image)
        
        # 随机调整亮度
        if random.random() < 0.5:
            factor = random.uniform(self.brightness[0], self.brightness[1])
            pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
            
        # 随机调整对比度
        if random.random() < 0.5:
            factor = random.uniform(self.contrast[0], self.contrast[1])
            pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)
            
        # 随机调整饱和度
        if random.random() < 0.5:
            factor = random.uniform(self.saturation[0], self.saturation[1])
            pil_image = ImageEnhance.Color(pil_image).enhance(factor)
            
        # 随机调整色调
        if random.random() < 0.5 and pil_image.mode == 'RGB':
            factor = random.uniform(-self.hue[1], self.hue[1])
            h, s, v = pil_image.convert('HSV').split()
            h = h.point(lambda x: (x + factor * 255) % 255)
            pil_image = Image.merge('HSV', (h, s, v)).convert('RGB')
            
        # 转换回原始类型
        return self.to_original_type(pil_image, image)


@TransformFactory.register('gaussian_blur')
class GaussianBlur(BaseTransform):
    """
    高斯模糊数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化高斯模糊。
        
        Args:
            config: 配置字典，包含：
                - radius: 模糊半径范围，可以是单个数字或元组 (min_radius, max_radius)
                - prob: 应用概率
        """
        super().__init__(config)
        self.radius = self.config.get('radius', 2.0)
        self.prob = self.config.get('prob', 0.5)
        
        # 确保radius是元组
        if isinstance(self.radius, (int, float)):
            self.radius = (0.1, self.radius)
            
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        执行高斯模糊。
        
        Args:
            image: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
            
        Returns:
            模糊后的图像，保持与输入相同的类型
        """
        # 转换为PIL图像进行处理
        pil_image = self.to_pil(image)
        
        # 随机决定是否应用模糊
        if random.random() < self.prob:
            # 随机选择模糊半径
            radius = random.uniform(self.radius[0], self.radius[1])
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            
        # 转换回原始类型
        return self.to_original_type(pil_image, image)


@TransformFactory.register('random_erasing')
class RandomErasing(BaseTransform):
    """
    随机擦除数据增强。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化随机擦除。
        
        Args:
            config: 配置字典，包含：
                - prob: 应用概率
                - scale: 擦除区域比例范围
                - ratio: 擦除区域宽高比范围
                - value: 填充值
        """
        super().__init__(config)
        self.prob = self.config.get('prob', 0.5)
        self.scale = self.config.get('scale', (0.02, 0.33))
        self.ratio = self.config.get('ratio', (0.3, 3.3))
        self.value = self.config.get('value', 0)
        
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        执行随机擦除。
        
        Args:
            image: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
            
        Returns:
            擦除后的图像，保持与输入相同的类型
        """
        # 转换为numpy数组进行处理
        np_image = self.to_numpy(image)
        
        if random.random() < self.prob:
            height, width = np_image.shape[:2]
            
            # 计算擦除区域
            area = height * width
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h < height and w < width:
                # 随机选择擦除位置
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                
                # 执行擦除
                if len(np_image.shape) == 3:
                    np_image[top:top + h, left:left + w, :] = self.value
                else:
                    np_image[top:top + h, left:left + w] = self.value
                    
        # 转换回原始类型
        return self.to_original_type(np_image, image)


class RandomHorizontalFlip(BaseTransform):
    """
    随机水平翻转图像。
    对于目标框坐标，x坐标会进行水平翻转：x' = w - 1 - x
    注意：翻转后需要确保x1 <= x2
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data):
        if isinstance(data, dict) and "images" in data:
            images = data["images"]
            labels = data.get("labels", None)
            if random.random() < self.p:
                flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
                data["images"] = flipped_images
                if labels is not None:
                    for label in labels:
                        if "object_coords" in label:
                            new_boxes = []
                            w = images[0].width
                            for box in label["object_coords"]:
                                x1, y1, x2, y2 = box
                                # 水平翻转坐标：x' = w - 1 - x
                                # 注意：需要先计算新的x1和x2，然后确保x1 <= x2
                                new_x1 = w - 1 - x2
                                new_x2 = w - 1 - x1
                                # 确保坐标在有效范围内
                                new_x1 = max(0, min(w - 1, new_x1))
                                new_x2 = max(0, min(w - 1, new_x2))
                                # 确保x1 <= x2
                                new_x1, new_x2 = min(new_x1, new_x2), max(new_x1, new_x2)
                                new_boxes.append([new_x1, y1, new_x2, y2])
                            label["object_coords"] = new_boxes
            return data
        elif isinstance(data, Image.Image):
            if random.random() < self.p:
                return data.transpose(Image.FLIP_LEFT_RIGHT)
            return data
        else:
            raise ValueError('RandomHorizontalFlip只支持PIL.Image或包含images字段的dict')


class RandomVerticalFlip(BaseTransform):
    """
    随机垂直翻转图像。
    对于目标框坐标，y坐标会进行垂直翻转：y' = h - 1 - y
    注意：翻转后需要确保y1 <= y2
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data):
        if isinstance(data, dict) and "images" in data:
            images = data["images"]
            labels = data.get("labels", None)
            if random.random() < self.p:
                flipped_images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]
                data["images"] = flipped_images
                if labels is not None:
                    for label in labels:
                        if "object_coords" in label:
                            new_boxes = []
                            h = images[0].height
                            for box in label["object_coords"]:
                                x1, y1, x2, y2 = box
                                # 垂直翻转坐标：y' = h - 1 - y
                                # 注意：需要先计算新的y1和y2，然后确保y1 <= y2
                                new_y1 = h - 1 - y2
                                new_y2 = h - 1 - y1
                                # 确保坐标在有效范围内
                                new_y1 = max(0, min(h - 1, new_y1))
                                new_y2 = max(0, min(h - 1, new_y2))
                                # 确保y1 <= y2
                                new_y1, new_y2 = min(new_y1, new_y2), max(new_y1, new_y2)
                                new_boxes.append([x1, new_y1, x2, new_y2])
                            label["object_coords"] = new_boxes
            return data
        elif isinstance(data, Image.Image):
            if random.random() < self.p:
                return data.transpose(Image.FLIP_TOP_BOTTOM)
            return data
        else:
            raise ValueError('RandomVerticalFlip只支持PIL.Image或包含images字段的dict')


# 导出所有数据增强类
__all__ = [
    'RandomCrop',
    'RandomFlip',
    'RandomRotation',
    'ColorJitter',
    'GaussianBlur',
    'RandomErasing'
] 