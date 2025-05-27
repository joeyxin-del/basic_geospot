import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, Any

class Logger:
    """
    日志工具类，提供灵活的日志记录功能。
    支持多级别日志记录、文件轮转、自定义格式等功能。
    """
    # 日志级别映射
    LEVELS = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    def __init__(
        self,
        name: str,
        level: str = 'info',
        log_dir: str = 'logs',
        log_format: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        rotation: str = 'size'  # 'size' or 'time'
    ):
        """
        初始化日志工具。
        
        Args:
            name: 日志器名称
            level: 日志级别，可选值：'debug', 'info', 'warning', 'error', 'critical'
            log_dir: 日志文件目录
            log_format: 日志格式，如果为None则使用默认格式
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_bytes: 单个日志文件的最大大小（字节）
            backup_count: 保留的日志文件数量
            rotation: 日志轮转方式，'size'按大小轮转，'time'按时间轮转
        """
        self.name = name
        self.level = self.LEVELS.get(level.lower(), logging.INFO)
        self.log_dir = log_dir
        self.log_format = log_format or (
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        self.console_output = console_output
        self.file_output = file_output
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotation = rotation
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(self.log_format)
        
        # 添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        # 添加文件处理器
        if file_output:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名
            log_file = os.path.join(log_dir, f'{name}.log')
            
            # 根据轮转方式创建文件处理器
            if rotation == 'size':
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            else:  # time
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """
        记录日志的底层方法。
        
        Args:
            level: 日志级别
            msg: 日志消息
            *args: 位置参数
            **kwargs: 关键字参数
        """
        self.logger.log(level, msg, *args, **kwargs)
        
    def debug(self, msg: str, *args, **kwargs) -> None:
        """记录调试级别日志"""
        self._log(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """记录信息级别日志"""
        self._log(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        """记录警告级别日志"""
        self._log(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        """记录错误级别日志"""
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs) -> None:
        """记录严重错误级别日志"""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
        
    def exception(self, msg: str, *args, **kwargs) -> None:
        """记录异常信息，自动包含异常堆栈"""
        self.logger.exception(msg, *args, **kwargs)
        
    def set_level(self, level: str) -> None:
        """
        设置日志级别。
        
        Args:
            level: 日志级别，可选值：'debug', 'info', 'warning', 'error', 'critical'
        """
        if level.lower() in self.LEVELS:
            self.level = self.LEVELS[level.lower()]
            self.logger.setLevel(self.level)
        else:
            raise ValueError(f"Invalid log level: {level}")
            
    def get_logger(self) -> logging.Logger:
        """
        获取底层的logging.Logger对象。
        
        Returns:
            logging.Logger对象
        """
        return self.logger


# 创建默认日志器
default_logger = Logger('default')


def get_logger(name: str, **kwargs) -> Logger:
    """
    获取指定名称的日志器。
    如果日志器不存在，则创建一个新的日志器。
    
    Args:
        name: 日志器名称
        **kwargs: 传递给Logger构造函数的参数
        
    Returns:
        Logger实例
    """
    if name == 'default':
        return default_logger
    return Logger(name, **kwargs)


# 导出
__all__ = ['Logger', 'get_logger', 'default_logger'] 