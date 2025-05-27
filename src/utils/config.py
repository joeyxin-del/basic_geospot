import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Type, TypeVar, get_type_hints

import yaml

logger = None  # 延迟初始化logger

def _get_logger():
    """延迟获取logger实例"""
    global logger
    if logger is None:
        from src.utils.logger import get_logger
        logger = get_logger('config')
    return logger

T = TypeVar('T')

class ConfigError(Exception):
    """配置相关错误的基类"""
    pass

class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass

class ConfigFileError(ConfigError):
    """配置文件错误"""
    pass

@dataclass
class ConfigField:
    """
    配置字段定义。
    用于描述配置项的类型、默认值、验证规则等。
    """
    name: str
    type: Type
    default: Any = None
    required: bool = False
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    regex: Optional[str] = None
    
    def validate(self, value: Any) -> None:
        """
        验证字段值。
        
        Args:
            value: 要验证的值
            
        Raises:
            ConfigValidationError: 如果验证失败
        """
        # 检查类型
        if value is not None and not isinstance(value, self.type):
            raise ConfigValidationError(
                f"Field '{self.name}' must be of type {self.type.__name__}, "
                f"got {type(value).__name__}"
            )
            
        # 检查必填
        if self.required and value is None:
            raise ConfigValidationError(f"Field '{self.name}' is required")
            
        # 检查选项
        if self.choices is not None and value not in self.choices:
            raise ConfigValidationError(
                f"Field '{self.name}' must be one of {self.choices}, "
                f"got {value}"
            )
            
        # 检查数值范围
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ConfigValidationError(
                    f"Field '{self.name}' must be >= {self.min_value}, "
                    f"got {value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ConfigValidationError(
                    f"Field '{self.name}' must be <= {self.max_value}, "
                    f"got {value}"
                )

class ConfigManager:
    """
    配置管理工具。
    支持配置文件的读取、验证、更新和保存。
    """
    def __init__(
        self,
        config_class: Type[T],
        config_file: Optional[str] = None,
        env_prefix: str = "APP_"
    ):
        """
        初始化配置管理器。
        
        Args:
            config_class: 配置类，必须是一个dataclass
            config_file: 配置文件路径，支持.yaml和.json格式
            env_prefix: 环境变量前缀
        """
        if not hasattr(config_class, '__dataclass_fields__'):
            raise TypeError(f"{config_class.__name__} must be a dataclass")
            
        self.config_class = config_class
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.config: Optional[T] = None
        
        # 从配置文件加载配置
        if config_file:
            self.load_config()
        else:
            self.config = self._create_default_config()
            
    def _create_default_config(self) -> T:
        """
        创建默认配置。
        
        Returns:
            配置实例
        """
        logger = _get_logger()  # 在这里获取logger
        # 获取字段定义
        fields = get_type_hints(self.config_class)
        field_values = {}
        
        # 遍历字段，设置默认值
        for field_name, field_type in fields.items():
            field = getattr(self.config_class, field_name, None)
            if isinstance(field, ConfigField):
                # 从环境变量获取值
                env_name = f"{self.env_prefix}{field_name.upper()}"
                env_value = os.getenv(env_name)
                
                if env_value is not None:
                    # 转换环境变量值到正确的类型
                    try:
                        if field_type == bool:
                            value = env_value.lower() in ('true', '1', 'yes')
                        else:
                            value = field_type(env_value)
                        field.validate(value)
                        field_values[field_name] = value
                    except (ValueError, ConfigValidationError) as e:
                        logger.warning(
                            f"Invalid environment variable {env_name}: {e}. "
                            f"Using default value."
                        )
                        field_values[field_name] = field.default
                else:
                    field_values[field_name] = field.default
            else:
                # 使用dataclass的默认值
                field_values[field_name] = getattr(self.config_class, field_name, None)
                
        return self.config_class(**field_values)
        
    def load_config(self) -> None:
        """
        从配置文件加载配置。
        
        Raises:
            ConfigFileError: 如果配置文件不存在或格式错误
            ConfigValidationError: 如果配置验证失败
        """
        logger = _get_logger()  # 在这里获取logger
        if not self.config_file:
            raise ConfigFileError("No config file specified")
            
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found, using default config")
            self.config = self._create_default_config()
            return
            
        # 根据文件扩展名选择加载方法
        ext = os.path.splitext(self.config_file)[1].lower()
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if ext in ('.yaml', '.yml'):
                    config_dict = yaml.safe_load(f)
                elif ext == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigFileError(f"Unsupported config file format: {ext}")
                    
            # 验证配置
            self._validate_config(config_dict)
            
            # 创建配置实例
            self.config = self.config_class(**config_dict)
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigFileError(f"Error parsing config file: {e}")
            
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        保存配置到文件。
        
        Args:
            config_file: 配置文件路径，如果为None则使用初始化时的路径
            
        Raises:
            ConfigFileError: 如果保存失败
        """
        if not self.config:
            raise ConfigError("No config to save")
            
        save_path = config_file or self.config_file
        if not save_path:
            raise ConfigFileError("No config file specified")
            
        # 将配置转换为字典
        config_dict = {
            field: getattr(self.config, field)
            for field in self.config.__dataclass_fields__
        }
        
        # 根据文件扩展名选择保存方法
        ext = os.path.splitext(save_path)[1].lower()
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if ext in ('.yaml', '.yml'):
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
                elif ext == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigFileError(f"Unsupported config file format: {ext}")
                    
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigFileError(f"Error saving config file: {e}")
            
    def _validate_config(self, config_dict: Dict[str, Any]) -> None:
        """
        验证配置字典。
        
        Args:
            config_dict: 配置字典
            
        Raises:
            ConfigValidationError: 如果验证失败
        """
        fields = get_type_hints(self.config_class)
        
        # 检查必填字段
        for field_name, field_type in fields.items():
            field = getattr(self.config_class, field_name, None)
            if isinstance(field, ConfigField):
                value = config_dict.get(field_name)
                field.validate(value)
                
    def update_config(self, **kwargs) -> None:
        """
        更新配置。
        
        Args:
            **kwargs: 要更新的配置项
            
        Raises:
            ConfigValidationError: 如果更新后的配置验证失败
        """
        if not self.config:
            raise ConfigError("No config to update")
            
        # 创建新的配置字典
        config_dict = {
            field: getattr(self.config, field)
            for field in self.config.__dataclass_fields__
        }
        config_dict.update(kwargs)
        
        # 验证更新后的配置
        self._validate_config(config_dict)
        
        # 更新配置
        self.config = self.config_class(**config_dict)
        
    def get_config(self) -> T:
        """
        获取当前配置。
        
        Returns:
            配置实例
        """
        if not self.config:
            raise ConfigError("No config available")
        return self.config


# 导出
__all__ = ['ConfigManager', 'ConfigField', 'ConfigError', 'ConfigValidationError', 'ConfigFileError'] 