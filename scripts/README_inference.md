# 推理脚本使用说明

## 概述

本项目提供了两个脚本用于模型推理：

1. **`eval.py`** - 用于在测试集上评估模型性能
2. **`inference.py`** - 用于对新图像进行推理预测

## 使用方法

### 1. 评估脚本 (eval.py)

用于在测试集上评估模型性能：

```bash
python scripts/eval.py --checkpoint /path/to/your/model.pth --output-dir /path/to/output
```

**参数说明：**
- `--checkpoint`: 模型权重文件路径（必需）
- `--output-dir`: 评估结果输出目录（可选）

**功能：**
- 加载训练好的模型权重
- 在测试集上进行预测
- 计算评估指标（精确率、召回率、F1分数等）
- 生成评估报告和可视化图表

### 2. 推理脚本 (inference.py)

用于对新图像进行推理预测：

```bash
# 单张图像推理
python scripts/inference.py --checkpoint /path/to/your/model.pth --input /path/to/image.jpg

# 批量推理（目录）
python scripts/inference.py --checkpoint /path/to/your/model.pth --input /path/to/image/directory

# 自定义参数
python scripts/inference.py \
    --checkpoint /path/to/your/model.pth \
    --input /path/to/image.jpg \
    --output /path/to/output \
    --conf-thresh 0.5 \
    --topk 100

# 指定设备和模型
python scripts/inference.py \
    --checkpoint /path/to/your/model.pth \
    --input /path/to/image.jpg \
    --device cuda \
    --model-name spotgeo
```

**参数说明：**
- `--checkpoint`: 模型权重文件路径（必需）
- `--input`: 输入图像路径或目录路径（必需）
- `--output`: 输出结果保存路径（可选，默认：outputs/inference）
- `--conf-thresh`: 置信度阈值，默认0.5（可选）
- `--topk`: 每帧最多输出目标数，默认100（可选）
- `--device`: 计算设备，默认auto（可选：auto, cpu, cuda）
- `--model-name`: 模型名称，默认spotgeo（可选）
- `--model-config`: 模型配置文件路径（可选）

**功能：**
- 支持单张图像和批量推理
- 自动预处理图像
- 输出预测坐标
- 生成可视化结果
- 保存JSON格式的详细结果
- 不依赖配置文件，完全通过命令行参数控制

## 输出结果

### 评估脚本输出
- `evaluation_YYYYMMDD_HHMMSS.json`: 评估指标结果
- `evaluation_metrics.png`: 评估指标可视化图表
- `config.yaml`: 使用的配置文件

### 推理脚本输出
- `{filename}_results.json`: 推理结果（包含预测坐标）
- `{filename}/`: 每张图像的专用文件夹，包含：
  - `01_original.jpg`: 原始输入图像
  - `02_prediction.jpg`: 预测结果图（黑色背景上的蓝色标点）
  - `03_ground_truth.jpg`: 真实标注图（黑色背景上的白色标点，如果有GT数据）
  - `04_comparison.jpg`: 对比图（原图上同时显示预测和GT，白色为GT，蓝色为预测）
  - `info.txt`: 结果信息文件（包含图像信息、目标数量、参数等）

## 结果格式

### 推理结果JSON格式
```json
{
  "image_path": "/path/to/image.jpg",
  "predictions": {
    "frame": 0,
    "num_objects": 3,
    "object_coords": [
      [123.45, 67.89],
      [234.56, 78.90],
      [345.67, 89.01]
    ]
  },
  "model_outputs": {
    "cls_shape": [1, 1, 15, 20],
    "reg_shape": [1, 2, 15, 20]
  }
}
```

## 注意事项

1. **模型权重文件**：确保使用训练好的模型权重文件（.pth格式）
2. **图像格式**：支持常见的图像格式（jpg, png, bmp, tiff等）
3. **设备要求**：脚本会自动检测并使用可用的GPU/CPU
4. **内存要求**：批量推理时注意内存使用量
5. **参数格式**：使用传统的 `--key value` 格式传递参数
6. **独立性**：推理脚本不依赖配置文件，完全通过命令行参数控制

## 示例

```bash
# 评估模型在测试集上的性能
python scripts/eval.py --checkpoint checkpoints/best_model.pth

# 对单张图像进行推理
python scripts/inference.py --checkpoint checkpoints/best_model.pth --input test_images/sample.jpg

# 批量推理并调整参数
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input test_images/ \
    --conf-thresh 0.7 \
    --topk 50

# 指定输出目录和设备
python scripts/inference.py \
    --checkpoint outputs/singleframe_gpu_v2/best.pth \
    --input datasets/SpotGEOv2/test/1/1.png \
    --output ./inference_results \
    --device cuda

# 使用自定义模型配置
python scripts/inference.py \
    --checkpoint outputs/singleframe_gpu_v2/best.pth \
    --input datasets/SpotGEOv2/test/1/1.png \
    --model-name spotgeo \
    --model-config configs/model_config.yaml
```

## 常见问题

### 1. 设备选择
- `--device auto`: 自动选择GPU（如果可用）或CPU
- `--device cuda`: 强制使用GPU
- `--device cpu`: 强制使用CPU

### 2. 模型配置
如果您的模型需要特殊配置，可以通过 `--model-config` 参数指定YAML配置文件。

### 3. 输出目录
如果不指定 `--output` 参数，结果会保存在 `outputs/inference/` 目录下。

### 4. 批量推理
当 `--input` 指向目录时，脚本会自动处理目录中的所有图像文件。 