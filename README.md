# GEO Detection 项目

一个基于深度学习的地理目标检测项目，支持序列数据的多目标检测任务。

## 项目特性

- 🎯 **多目标检测**：支持序列数据中的多个目标同时检测
- 📊 **实验跟踪**：集成 Weights & Biases (wandb) 进行实验管理
- ⚙️ **配置管理**：使用 Hydra 进行灵活的配置管理
- 🧪 **完整测试**：包含单元测试和集成测试
- 📈 **性能监控**：内置训练和验证性能监控

## 完整安装流程示例

```bash
# 1. 安装 uv
pip install uv

# 2. 克隆项目
git clone <your-repo-url>
cd 002basic_GEO_Detection

# 3. 创建虚拟环境
uv venv

# 4. 激活环境
uv shell

# 5. 安装 PyTorch（选择适合你硬件的版本）
# GPU 用户（推荐）：
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 用户：
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. 安装其他依赖
uv pip install -r requirements.txt

# 7. 验证安装
uv run python -c "import torch; print('PyTorch安装成功！CUDA可用:', torch.cuda.is_available())"
```

## 数据准备

将你的数据集放在 `datasets/` 目录下，确保数据格式符合项目要求：
```json
{
  "sequence_id": 1,
  "frame": 1,
  "num_objects": 3,
  "object_coords": [[502.4, 237.1], [490.4, 221.8], [140.9, 129.1]]
}
```

## 配置设置

编辑 `configs/default.yaml` 文件，设置你的训练参数：
```yaml
model:
  name: "spotgeo"
  num_classes: 10

training:
  batch_size: 8
  learning_rate: 0.001
  epochs: 100

data:
  train_path: "datasets/train"
  val_path: "datasets/val"
```

## 开始训练

```bash
# 使用 uv 运行训练脚本
uv run python scripts/train.py

# 或者激活环境后运行
uv shell
python scripts/train.py
```

## 监控训练

训练过程中可以通过以下方式监控：
- **终端输出**：实时显示训练进度和指标
- **Wandb 面板**：在线查看详细的训练曲线和指标
- **日志文件**：保存在 `logs/` 目录下

## 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_models.py

# 生成测试覆盖率报告
uv run pytest --cov=src tests/
```

## 代码质量

```bash
# 代码格式化
uv run black src/ scripts/ tests/

# 导入排序
uv run isort src/ scripts/ tests/

# 代码检查
uv run flake8 src/ scripts/ tests/

# 类型检查
uv run mypy src/
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
