import os
import pytest
from src.datasets.spotgeov2 import SpotGEOv2Dataset

def test_spotgeov2_dataset_basic():
    """
    测试SpotGEOv2Dataset的基本功能，包括序列加载和样本读取。
    """
    # 假设有如下测试数据结构
    root_dir = 'datasets/SpotGEOv2/train'  # 需根据实际情况调整
    annotation_path = 'datasets/SpotGEOv2/train_anno.json'
    if not (os.path.exists(root_dir) and os.path.exists(annotation_path)):
        pytest.skip("测试数据不存在，跳过测试")
    
    dataset = SpotGEOv2Dataset(root_dir=root_dir, annotation_path=annotation_path)
    assert len(dataset) > 0, "数据集应包含至少一个序列"
    
    # 打印前5个序列的标注信息
    print("\n=== 数据集前5个序列的标注信息 ===")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n序列 {i+1}: {sample['sequence_name']}")
        print(f"图像数量: {len(sample['images'])}")
        print(f"标注数量: {len(sample['labels'])}")
        if sample['labels']:
            print("标注示例:")
            for j, label in enumerate(sample['labels'][:2]):  # 只打印前2帧的标注
                print(f"  帧 {j+1}:")
                print(f"    帧号: {label['frame']}")
                print(f"    目标数量: {label['num_objects']}")
                print(f"    目标坐标: {label['object_coords']}")
    
    # 基本功能测试
    sample = dataset[0]
    assert 'images' in sample and 'labels' in sample and 'sequence_name' in sample
    assert isinstance(sample['images'], list) and len(sample['images']) > 0
    assert isinstance(sample['labels'], list)
    assert isinstance(sample['sequence_name'], str) 