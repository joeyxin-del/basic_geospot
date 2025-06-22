# SwanLab集成使用指南

本文档介绍如何在GEO检测项目中使用SwanLab进行实验管理和训练曲线记录。

## 1. 安装SwanLab

```bash
pip install swanlab
```

## 2. 基本使用

### 2.1 启用SwanLab

在创建训练器时，设置以下参数来启用SwanLab：

```python
from src.training.trainer_singleframe import TrainerSingleFrame

trainer = TrainerSingleFrame(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    use_swanlab=True,  # 启用SwanLab
    swanlab_project='spotgeo-singleframe',  # 项目名称
    swanlab_mode='cloud',  # 模式：cloud/offline/local/disabled
    log_batch_metrics=True,  # 记录每个batch的指标
    log_gradients=True,      # 记录梯度信息
    # ... 其他参数
)
```

### 2.2 SwanLab模式说明

- **cloud**: 在线模式，数据上传到SwanLab云端
- **offline**: 离线模式，数据保存在本地
- **local**: 本地模式，启动本地SwanLab服务器
- **disabled**: 禁用模式，不记录任何数据

## 3. 记录的指标

### 3.1 Batch级别指标

每个训练和验证batch都会记录以下指标：

```python
# 训练batch指标
{
    'batch_loss': 0.1234,           # 当前batch损失
    'batch_lr': 0.0001,             # 当前学习率
    'batch_grad_norm': 1.2345,      # 梯度范数
    'batch_cls_loss': 0.0567,       # 分类损失
    'batch_reg_loss': 0.0667,       # 回归损失
    'epoch': 1,                     # 当前epoch
    'batch_idx': 10                 # batch索引
}

# 验证batch指标
{
    'val_batch_loss': 0.0987,       # 验证batch损失
    'val_epoch': 1,                 # 当前epoch
    'val_batch_idx': 5              # batch索引
}
```

### 3.2 Epoch级别指标

每个epoch结束后记录以下指标：

```python
{
    'epoch': 1,                     # epoch编号
    'train_loss': 0.1234,           # 平均训练损失
    'val_loss': 0.0987,             # 平均验证损失
    'val_score': 0.8765,            # 验证分数
    'val_f1': 0.8234,               # F1分数
    'val_mse': 0.0456,              # 均方误差
    'lr': 0.0001,                   # 学习率
    'train_time': 45.67,            # 训练时间(秒)
    'best_score': 0.8765,           # 最佳分数
    'patience_counter': 0           # 早停计数器
}
```

### 3.3 最终总结指标

训练结束后记录实验总结：

```python
{
    'final_best_epoch': 25,         # 最佳epoch
    'final_best_score': 0.8765,     # 最佳分数
    'final_total_epochs': 30,       # 总训练轮数
    'final_avg_train_loss': 0.1234, # 平均训练损失
    'final_avg_val_loss': 0.0987,   # 平均验证损失
    'final_avg_val_f1': 0.8234,     # 平均F1分数
    'final_best_val_f1': 0.8567,    # 最佳F1分数
    'final_total_training_time': 1367.89  # 总训练时间
}
```

## 4. 配置文件使用

使用YAML配置文件来管理SwanLab设置：

```yaml
# configs/swanlab_example.yaml
swanlab:
  enabled: true
  project: "spotgeo-singleframe"
  mode: "cloud"
  log_batch_metrics: true
  log_gradients: true
```

在训练脚本中加载配置：

```python
import yaml
from src.utils.config import load_config

# 加载配置
config = load_config('configs/swanlab_example.yaml')

# 创建训练器
trainer = TrainerSingleFrame(
    use_swanlab=config['swanlab']['enabled'],
    swanlab_project=config['swanlab']['project'],
    swanlab_mode=config['swanlab']['mode'],
    log_batch_metrics=config['swanlab']['log_batch_metrics'],
    log_gradients=config['swanlab']['log_gradients'],
    # ... 其他参数
)
```

## 5. 查看实验结果

### 5.1 在线模式

1. 访问 [https://swanlab.ai](https://swanlab.ai)
2. 登录你的账户
3. 找到对应的项目
4. 查看实验详情和训练曲线

### 5.2 离线模式

查看本地保存的日志文件：

```bash
# 日志文件位置
outputs/experiment_name/timestamp/

# 主要文件
- swanlab.json          # SwanLab配置
- metrics.csv           # 指标数据
- config.yaml           # 实验配置
```

### 5.3 本地模式

1. 启动本地SwanLab服务器
2. 访问 `http://localhost:5000`
3. 查看实验界面

## 6. 高级功能

### 6.1 自定义指标记录

在训练过程中添加自定义指标：

```python
# 在训练循环中添加
custom_metrics = {
    'custom_metric': your_calculation(),
    'model_parameter_count': sum(p.numel() for p in model.parameters())
}
trainer._log_swanlab_metrics(custom_metrics, step=current_step)
```

### 6.2 实验对比

SwanLab支持多个实验的对比：

```python
# 不同配置的实验
experiments = [
    {'lr': 0.001, 'batch_size': 16},
    {'lr': 0.0001, 'batch_size': 32},
    {'lr': 0.00001, 'batch_size': 64}
]

for i, config in enumerate(experiments):
    trainer = TrainerSingleFrame(
        experiment_name=f"exp_{i}_{config['lr']}_{config['batch_size']}",
        # ... 其他参数
    )
    trainer.train()
```

### 6.3 模型版本管理

SwanLab可以跟踪模型版本：

```python
# 保存模型检查点时记录到SwanLab
if is_best_model:
    swanlab.log({
        'model_checkpoint': 'best_model.pth',
        'model_performance': best_score
    })
```

## 7. 故障排除

### 7.1 常见问题

**Q: SwanLab初始化失败**
A: 检查网络连接和API密钥设置

**Q: 指标记录失败**
A: 确保指标值为数值类型，避免记录复杂对象

**Q: 离线模式文件过大**
A: 考虑减少batch级别指标记录频率

### 7.2 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在训练器中启用调试
trainer = TrainerSingleFrame(
    # ... 其他参数
    debug=True  # 如果支持
)
```

## 8. 最佳实践

1. **合理设置记录频率**: 避免记录过于频繁的batch指标
2. **使用有意义的实验名称**: 便于后续查找和对比
3. **记录关键超参数**: 确保实验可重现
4. **定期备份数据**: 特别是离线模式的实验数据
5. **团队协作**: 使用统一的项目命名规范

## 9. 示例代码

完整的使用示例请参考：
- `test_swanlab.py`: 基本集成测试
- `configs/swanlab_example.yaml`: 配置文件示例
- `scripts/train_singleframe.py`: 训练脚本示例 