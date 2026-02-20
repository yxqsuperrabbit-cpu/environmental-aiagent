# AirGuardian - 空气质量预测与智能分析系统

<div align="center">

🌍 **基于深度学习和大语言模型的智能空气质量预测系统**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

##  项目简介

AirGuardian 是一个集成了 LSTM 深度学习模型和大语言模型（LLM）的智能空气质量预测与分析系统。系统能够基于历史空气质量数据预测未来 24-72 小时的 PM2.5 浓度，并通过 LLM 生成智能分析报告和健康建议。

## 核心特性

- 🔮 **智能预测**: 使用 LSTM 时间序列模型预测未来 24-72 小时的 PM2.5 浓度
- 📊 **置信区间**: 提供预测值的置信区间和不确定性范围
- 🧠 **AI 分析**: 集成 Ollama + DeepSeek-R1 模型生成智能分析报告
- 📄 **双版本报告**: 自动生成政府版和市民版分析文档
- ⚠️ **健康预警**: 基于空气质量等级的智能健康风险预警系统
- 📈 **可视化界面**: Streamlit Web 界面提供交互式数据展示
- 🚀 **GPU 加速**: 支持 NVIDIA GPU 加速推理

##  快速开始

### 环境要求

- Python 3.9 或更高版本
- uv (推荐) 或 pip 包管理器
- Ollama (用于 LLM 分析功能)
- NVIDIA GPU 或 Apple Silicon (可选，用于加速推理)

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yxqsuperrabbit-cpu/environmental-aiagent.git
cd environmental-aiagent
```

#### 2. 创建虚拟环境

使用 uv (推荐):
```bash
uv venv
```

Windows 激活:
```bash
.venv\Scripts\activate
```

Linux/Mac 激活:
```bash
source .venv/bin/activate
```

#### 3. 安装依赖

```bash
# 安装运行时依赖
uv sync

# 如果需要GPU监控功能（可选）
uv add --optional acceleration
```

#### 4. 配置环境变量

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件，配置 Ollama 等参数
```

`.env` 配置示例:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:latest
MODEL_PATH=models/
DATA_PATH=data/
DEBUG=False
LOG_LEVEL=INFO
```

#### 5. 安装和配置 Ollama

```bash
# 安装 Ollama (访问 https://ollama.ai)

# 拉取 DeepSeek-R1 模型
ollama pull deepseek-r1:latest

# 启动 Ollama 服务
ollama serve
```

### 运行应用

#### 启动 Web 界面

```bash
# 使用启动脚本
python run_app.py

# 或直接使用 Streamlit
streamlit run src/web_interface/streamlit_app.py
```

访问 http://localhost:8503 查看 Web 界面。

##  数据格式要求

### 输入数据格式

系统需要包含以下字段的 CSV 数据：

| 字段名 | 类型 | 描述 | 单位 |
|--------|------|------|------|
| timestamp | datetime | 时间戳 | - |
| pm25 | float | PM2.5浓度 | µg/m³ |
| temperature | float | 温度 | °C |
| humidity | float | 湿度 | % |
| wind_speed | float | 风速 | m/s |
| wind_direction | float | 风向 | 度 |

### 数据示例

```csv
timestamp,pm25,temperature,humidity,wind_speed,wind_direction
2024-12-22 00:00:00,45.2,18.5,65,3.2,180
2024-12-22 01:00:00,48.1,18.2,67,3.5,185
2024-12-22 02:00:00,52.3,17.8,70,3.8,190
```

##  配置说明

### 模型配置

系统使用预训练的 LSTM 模型，配置参数已优化。如需调整，可在源代码中修改相关参数。

### Ollama 配置

确保 Ollama 服务正在运行，并且已下载 DeepSeek-R1 模型（当然也可以用其他模型，更改配置即可）：

```bash
# 检查 Ollama 状态
ollama list

# 如果模型未下载，执行：
ollama pull deepseek-r1:latest
```

##  使用说明

1. **启动应用**: 运行 `python run_app.py`
2. **上传数据**: 在 Web 界面上传符合格式要求的 CSV 数据文件
3. **查看预测**: 系统自动生成未来 24-72 小时的 PM2.5 浓度预测
4. **获取分析**: 查看 AI 生成的智能分析报告和健康建议
5. **下载报告**: 可下载 PDF 格式的分析报告

##  技术支持

如遇到问题，请检查：
1. Python 环境和依赖包是否正确安装
2. Ollama 服务是否正常运行
3. DeepSeek-R1 模型是否已下载
4. 数据格式是否符合要求


