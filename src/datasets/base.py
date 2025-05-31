import os
from typing import Any, Dict, List
from torch.utils.data import Dataset

class DatasetBase(Dataset):
    """
    数据集基类，所有自定义数据集需继承本类。
    主要定义了数据集的基本接口，便于后续扩展和统一管理。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据集基类。
        :param config: 数据集相关配置参数
        """
        self.config = config

    def __len__(self) -> int:
        """
        返回数据集样本数量。
        """
        raise NotImplementedError("子类需实现 __len__ 方法")

    def __getitem__(self, index: int) -> Any:
        """
        获取指定索引的数据样本。
        """
        raise NotImplementedError("子类需实现 __getitem__ 方法") 