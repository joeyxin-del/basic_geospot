# pytest使用指南

## 1. pytest简介

### 1.1 什么是pytest
pytest是Python的一个功能强大的测试框架，它提供了简单灵活的方式来编写测试用例，并具有丰富的插件生态系统。相比Python自带的unittest框架，pytest提供了更简洁的语法和更强大的功能。

### 1.2 为什么选择pytest
- 语法简洁：使用简单的assert语句进行断言
- 自动发现：自动发现和运行测试用例
- 丰富的插件：支持参数化、fixture、测试报告等
- 详细的失败信息：提供清晰的错误追踪
- 活跃的社区：持续更新和维护

### 1.3 主要特点
- 支持函数式测试
- 支持参数化测试
- 支持fixture（测试夹具）
- 支持测试跳过和预期失败
- 支持测试报告生成
- 支持测试覆盖率统计

## 2. 基础使用

### 2.1 安装方法
```bash
# 使用pip安装
pip install pytest

# 使用conda安装
conda install pytest
```

### 2.2 基本语法
```python
# test_example.py
def test_simple():
    assert 1 + 1 == 2

def test_string():
    assert "hello" + " world" == "hello world"
```

### 2.3 常用命令
```bash
# 运行所有测试
pytest

# 运行指定测试文件
pytest test_file.py

# 运行指定测试函数
pytest test_file.py::test_function

# 显示详细输出
pytest -v

# 显示打印输出
pytest -s
```

### 2.4 测试文件命名规则
- 测试文件应以`test_`开头或`_test`结尾
- 例如：`test_models.py`或`models_test.py`

### 2.5 测试函数命名规则
- 测试函数应以`test_`开头
- 例如：`test_model_initialization()`

## 3. 常用功能

### 3.1 断言（assert）的使用
```python
def test_assertions():
    # 基本断言
    assert 1 == 1
    
    # 包含断言
    assert "hello" in "hello world"
    
    # 异常断言
    import pytest
    with pytest.raises(ValueError):
        int("invalid")
```

### 3.2 参数化测试
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected
```

### 3.3 夹具（fixture）的使用
```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### 3.4 跳过测试和预期失败
```python
import pytest

@pytest.mark.skip(reason="功能尚未实现")
def test_future_feature():
    assert False

@pytest.mark.xfail
def test_known_failure():
    assert False
```

### 3.5 测试报告生成
```bash
# 生成HTML报告
pytest --html=report.html

# 生成JUnit XML报告
pytest --junitxml=report.xml
```

## 4. 项目中的实践

### 4.1 测试目录结构 