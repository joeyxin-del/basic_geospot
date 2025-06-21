#!/usr/bin/env python3
"""测试SwanLab API"""

import sys
import os

# 设置环境变量来避免交互式提示
os.environ['SWANLAB_DISABLE_PROMPT'] = '1'

try:
    import swanlab
    print("SwanLab导入成功")
    
    # 检查swanlab.init的参数
    import inspect
    sig = inspect.signature(swanlab.init)
    print(f"swanlab.init参数: {sig}")
    
    # 尝试不同的初始化方式
    print("\n尝试初始化SwanLab...")
    
    # 方式1：基本初始化（使用experiment_name而不是name）
    try:
        run = swanlab.init(project="test-project", experiment_name="test-experiment")
        print("✅ 基本初始化成功")
        print(f"   运行ID: {run.id}")
        swanlab.finish()
    except Exception as e:
        print(f"❌ 基本初始化失败: {e}")
    
    # 方式2：带config的初始化
    try:
        run = swanlab.init(
            project="test-project", 
            experiment_name="test-experiment",
            config={"lr": 0.001, "batch_size": 32}
        )
        print("✅ 带config初始化成功")
        print(f"   运行ID: {run.id}")
        swanlab.finish()
    except Exception as e:
        print(f"❌ 带config初始化失败: {e}")
    
    # 方式3：尝试mode参数 - offline模式
    try:
        run = swanlab.init(
            project="test-project", 
            experiment_name="test-experiment",
            mode="offline"
        )
        print("✅ offline模式初始化成功")
        print(f"   运行ID: {run.id}")
        swanlab.finish()
    except Exception as e:
        print(f"❌ offline模式初始化失败: {e}")
    
    # 方式4：尝试local模式
    try:
        run = swanlab.init(
            project="test-project", 
            experiment_name="test-experiment",
            mode="local"
        )
        print("✅ local模式初始化成功")
        print(f"   运行ID: {run.id}")
        swanlab.finish()
    except Exception as e:
        print(f"❌ local模式初始化失败: {e}")
    
    # 方式5：测试日志记录功能
    try:
        run = swanlab.init(
            project="test-project", 
            experiment_name="test-metrics",
            mode="offline"
        )
        print("✅ 开始测试日志记录...")
        
        # 记录一些指标
        swanlab.log({"accuracy": 0.85, "loss": 0.15})
        swanlab.log({"accuracy": 0.87, "loss": 0.13})
        swanlab.log({"accuracy": 0.89, "loss": 0.11})
        
        print("✅ 指标记录成功")
        swanlab.finish()
    except Exception as e:
        print(f"❌ 日志记录测试失败: {e}")
    
    # 方式6：测试禁用模式
    try:
        run = swanlab.init(
            project="test-project", 
            experiment_name="test-disabled",
            mode="disabled"
        )
        print("✅ disabled模式初始化成功")
        print(f"   运行ID: {run.id}")
        swanlab.finish()
    except Exception as e:
        print(f"❌ disabled模式初始化失败: {e}")
        
    print("\n🎉 SwanLab API测试完成！")
        
except ImportError:
    print("❌ SwanLab未安装")
    print("请运行: pip install swanlab")
except Exception as e:
    print(f"❌ 测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc() 