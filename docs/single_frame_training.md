# 单帧训练模式使用指南

## 概述

单帧训练模式将原本的序列数据展开为单帧数据进行处理，简化了训练逻辑，提高了训练效率。相比于序列训练模式，单帧模式具有以下优势：

- **简化的训练逻辑**：移除了序列处理的嵌套循环
- **更好的内存管理**：每次只处理单张图像
- **更灵活的批处理**：可以设置更大的batch_size
- **更容易调试**：可以直接检查单张图像的预测结果
- **YAML配置管理**：支持通过配置文件管理所有训练参数

## 核心组件

### 1. SpotGEOv2_SingleFrame 数据集类

- **文件位置**: `src/datasets/spotgeov2_singleframe.py`
- **功能**: 将序列数据展开为单帧数据
- **输出格式**:
  ```python
  {
      'image': PIL.Image,           # 单张图像
      'label': dict,               # 单帧标注
      'sequence_name': str,        # 序列名称
      'frame_idx': int            # 帧索引
  }
  ```

### 2. TrainerSingleFrame 训练器类

- **文件位置**: `src/training/trainer_singleframe.py`
- **功能**: 专门用于单帧训练的训练器
- **特点**: 简化的训练循环，移除序列处理逻辑，支持offline模式

### 3. 训练脚本

- **文件位置**: `scripts/train_singleframe.py`
- **功能**: 完整的单帧训练流程，支持YAML配置文件

### 4. 配置文件

- **基础配置**: `configs/singleframe_train.yaml` - 标准训练配置
- **GPU优化**: `configs/singleframe_gpu.yaml` - GPU优化配置
- **调试配置**: `configs/singleframe_debug.yaml` - 快速调试配置

## 使用方法

### 推荐方式：使用YAML配置文件

#### 1. 基本用法
```bash
# 使用默认配置文件
python scripts/train_singleframe.py --config configs/singleframe_train.yaml

# 使用GPU优化配置
python scripts/train_singleframe.py --config configs/singleframe_gpu.yaml

# 使用调试配置
python scripts/train_singleframe.py --config configs/singleframe_debug.yaml
```

#### 2. 配置文件 + 命令行覆盖
```bash
# 使用配置文件，但覆盖某些参数
python scripts/train_singleframe.py \
    --config configs/singleframe_train.yaml \
    --batch_size 32 \
    --epochs 200 \
    --experiment_name my_experiment

# 使用配置文件，但指定不同的数据路径
python scripts/train_singleframe.py \
    --config configs/singleframe_train.yaml \
    --train_dir /path/to/your/train/data \
    --val_dir /path/to/your/val/data
```

### 传统方式：纯命令行参数

```bash
python scripts/train_singleframe.py \
    --train_dir /path/to/train/data \
    --train_anno /path/to/train_anno.json \
    --val_dir /path/to/val/data \
    --val_anno /path/to/val_anno.json \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4
```

## 配置文件详解

### 基础配置文件结构

```yaml
# 数据配置
data:
  train_dir: "data/train"
  train_anno: "data/train_anno.json"
  val_dir: "data/val"
  val_anno: "data/val_anno.json"

# 模型配置
model:
  name: "spotgeo"
  config:
    backbone_channels: [64, 128, 256, 512]
    detection_channels: [256, 128, 64]
    num_classes: 1
    use_bn: true
    dropout: 0.1

# 训练配置
training:
  batch_size: 16
  num_workers: 4
  epochs: 100
  lr: 1e-4
  optimizer: "adam"
  scheduler: "step"
  checkpoint_interval: 10
  eval_interval: 1
  early_stopping_patience: 15

# 输出配置
output:
  output_dir: "outputs"
  experiment_name: "singleframe_v1"

# 设备配置
device: "auto"

# 日志配置
logging:
  use_swanlab: false
  swanlab_project: "spotgeo-singleframe"

# 其他配置
seed: 42
```

### 配置优先级

配置参数的优先级从高到低：
1. **命令行参数** - 最高优先级
2. **YAML配置文件** - 中等优先级  
3. **默认值** - 最低优先级

这意味着你可以：
- 使用配置文件设置大部分参数
- 用命令行参数覆盖特定的配置项
- 未指定的参数会使用合理的默认值

### 预设配置文件

#### 1. 标准配置 (`configs/singleframe_train.yaml`)
- 适用于一般训练场景
- 平衡的批次大小和训练参数
- 关闭swanlab记录

#### 2. GPU优化配置 (`configs/singleframe_gpu.yaml`)
- 适用于GPU内存充足的情况
- 更大的批次大小和学习率
- 启用swanlab记录
- 使用余弦学习率调度

#### 3. 调试配置 (`configs/singleframe_debug.yaml`)
- 适用于快速测试和调试
- 小批次大小和少量训练轮数
- 精简的模型配置
- 频繁的检查点保存

## 完整参数说明

### 数据相关参数
- `data.train_dir`: 训练数据目录路径
- `data.train_anno`: 训练标注文件路径
- `data.val_dir`: 验证数据目录路径
- `data.val_anno`: 验证标注文件路径

### 模型相关参数
- `model.name`: 模型名称 (默认: 'spotgeo')
- `model.config`: 模型特定配置

### 训练相关参数
- `training.batch_size`: 批次大小 (默认: 8)
- `training.num_workers`: 数据加载器工作进程数 (默认: 4)
- `training.epochs`: 训练轮数 (默认: 100)
- `training.lr`: 学习率 (默认: 1e-4)
- `training.optimizer`: 优化器类型 (adam/sgd, 默认: adam)
- `training.scheduler`: 学习率调度器类型 (step/cosine/none, 默认: step)
- `training.checkpoint_interval`: 检查点保存间隔 (默认: 10)
- `training.eval_interval`: 验证间隔 (默认: 1)
- `training.early_stopping_patience`: 早停耐心值 (默认: 10)

### 输出相关参数
- `output.output_dir`: 输出目录 (默认: 'outputs')
- `output.experiment_name`: 实验名称

### 其他参数
- `device`: 训练设备 (cuda/cpu/auto, 默认: auto)
- `logging.use_swanlab`: 是否使用swanlab记录
- `logging.swanlab_project`: swanlab项目名称
- `seed`: 随机种子 (默认: 42)

## 高级用法示例

### 1. 创建自定义配置文件
```yaml
# my_config.yaml
data:
  train_dir: "/path/to/my/train/data"
  train_anno: "/path/to/my/train_anno.json"
  val_dir: "/path/to/my/val/data"
  val_anno: "/path/to/my/val_anno.json"

training:
  batch_size: 24
  epochs: 150
  lr: 2e-4

output:
  experiment_name: "my_custom_experiment"

logging:
  use_swanlab: true
  swanlab_project: "my-project"
```

### 2. 使用配置文件进行实验管理
```bash
# 实验1：基准模型
python scripts/train_singleframe.py --config configs/singleframe_train.yaml

# 实验2：更大批次
python scripts/train_singleframe.py \
    --config configs/singleframe_train.yaml \
    --batch_size 32 \
    --experiment_name "large_batch_experiment"

# 实验3：不同学习率
python scripts/train_singleframe.py \
    --config configs/singleframe_train.yaml \
    --lr 5e-4 \
    --experiment_name "high_lr_experiment"
```

### 3. 批量实验脚本
```bash
#!/bin/bash
# run_experiments.sh

# 不同批次大小的实验
for batch_size in 8 16 32; do
    python scripts/train_singleframe.py \
        --config configs/singleframe_train.yaml \
        --batch_size $batch_size \
        --experiment_name "batch_${batch_size}_experiment"
done
```

## 代码示例

### 1. 直接使用数据集类

```python
from src.datasets.spotgeov2_singleframe import SpotGEOv2_SingleFrame
from torch.utils.data import DataLoader

# 创建数据集
dataset = SpotGEOv2_SingleFrame(
    root_dir='data/train',
    annotation_path='data/train_anno.json'
)

# 创建数据加载器
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 遍历数据
for batch in loader:
    images = batch['image']      # List[PIL.Image] 或 torch.Tensor
    labels = batch['label']      # List[Dict]
    seq_names = batch['sequence_name']  # List[str]
    frame_indices = batch['frame_idx']  # List[int]
    
    print(f"Batch size: {len(images)}")
    print(f"First label: {labels[0]}")
```

### 2. 使用配置文件编程

```python
import yaml
from src.training.trainer_singleframe import TrainerSingleFrame

# 加载配置
with open('configs/singleframe_train.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 修改配置
config['training']['batch_size'] = 32
config['training']['lr'] = 2e-4

# 使用修改后的配置训练
# ... 创建模型、数据加载器等
trainer = TrainerSingleFrame(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    **config['training']  # 使用配置中的训练参数
)
```

## 性能对比

### 内存使用
- **序列模式**: 需要加载完整序列到内存 (批次 × 序列长度 × 图像)
- **单帧模式**: 只需要加载单帧到内存 (批次 × 图像)

### 训练速度
- **序列模式**: 嵌套循环处理，较慢
- **单帧模式**: 直接批处理，更快

### 批次大小
- **序列模式**: 受序列长度限制，通常较小
- **单帧模式**: 不受序列长度限制，可以设置更大

### 配置管理
- **传统方式**: 命令行参数难以管理
- **YAML配置**: 结构化管理，易于版本控制和实验复现

## 注意事项

1. **数据集大小变化**: 单帧模式会将数据集大小从序列数扩展到总帧数

2. **评估方式**: 评估器会自动将单帧结果重新组合成序列结果进行评估

3. **模型兼容性**: 现有的SpotGEO模型完全兼容单帧输入

4. **时序信息**: 单帧模式丢失了帧间的时序关系

5. **配置文件验证**: 脚本会自动验证配置文件中的路径是否存在

## 故障排除

### 常见问题

1. **配置文件错误**
   - 检查YAML语法是否正确
   - 确保所有路径都存在
   - 验证参数类型是否匹配

2. **内存不足**
   - 减小配置文件中的 `batch_size`
   - 减少 `num_workers`

3. **训练速度慢**
   - 增加 `num_workers`
   - 使用GPU训练
   - 增大 `batch_size`

### 调试技巧

1. **使用调试配置**: 先用 `configs/singleframe_debug.yaml` 测试
2. **检查配置**: 观察训练开始时打印的最终配置
3. **渐进式调整**: 从小参数开始，逐步增加复杂度

## 输出结果

训练完成后，会在输出目录生成以下文件：

```
outputs/experiment_name/
├── checkpoints/          # 检查点文件
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── ...
├── plots/               # 训练图表
│   ├── train_loss_*.png
│   ├── val_loss_*.png
│   └── ...
├── evaluations/         # 评估结果
├── best.pth            # 最佳模型
├── last.pth            # 最新模型
└── results.json        # 训练结果摘要
```

## 快速开始

1. **准备数据**: 确保训练和验证数据路径正确
2. **选择配置**: 根据需求选择合适的配置文件
3. **开始训练**:
   ```bash
   python scripts/train_singleframe.py --config configs/singleframe_train.yaml
   ```
4. **监控训练**: 观察输出日志和生成的图表
5. **调整参数**: 根据结果调整配置文件中的参数 