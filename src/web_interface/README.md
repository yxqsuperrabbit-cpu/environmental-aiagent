# AirGuardian Web界面模块

## 概述

本模块实现了AirGuardian系统的Streamlit Web界面，提供用户友好的交互式界面用于空气质量预测、分析和文档生成。

## 功能特性

### 🎯 核心功能
- **数据输入**: 支持CSV文件上传、手动输入和示例数据
- **预测执行**: 集成LSTM模型进行空气质量预测
- **智能分析**: 基于LLM的预测结果分析和建议生成
- **文档导出**: 生成市民版和政府版报告，支持PDF导出
- **可视化展示**: 丰富的图表和热力图展示

### 📊 界面组件
1. **数据输入与预测标签页**
   - 多种数据输入方式
   - 实时预测执行
   - 预测结果展示

2. **分析报告标签页**
   - 历史情况分析
   - 预测情况分析
   - 健康预警信息
   - 政府和市民建议

3. **文档生成标签页**
   - 市民版文档生成
   - 政府版文档生成
   - PDF一键导出

4. **可视化展示标签页**
   - 预测趋势图
   - 置信区间图
   - 空气质量等级分布
   - 历史数据趋势
   - 交互式仪表板

5. **热力图展示标签页**
   - 区域空气质量热力图
   - 支持自定义数据上传
   - 交互式地图展示

## 模块结构

```
src/web_interface/
├── __init__.py              # 模块初始化
├── streamlit_app.py         # 主应用程序
├── web_interface.py         # 界面组件类
├── document_generator.py    # 文档生成器
├── visualization.py         # 可视化组件
└── README.md               # 本文档
```

## 主要类

### AirGuardianApp
主应用程序类，负责整体界面布局和功能协调。

**主要方法**:
- `run()`: 运行Streamlit应用
- `_render_sidebar()`: 渲染侧边栏
- `_render_main_content()`: 渲染主要内容区域
- `_render_prediction_tab()`: 渲染预测标签页
- `_render_analysis_tab()`: 渲染分析标签页
- `_render_document_tab()`: 渲染文档标签页
- `_render_visualization_tab()`: 渲染可视化标签页
- `_render_heatmap_tab()`: 渲染热力图标签页

### WebInterface
标准化的界面组件类，提供可重用的界面元素。

**主要方法**:
- `render_data_input()`: 渲染数据输入界面
- `display_analysis()`: 展示分析结果
- `generate_citizen_document()`: 生成市民版文档
- `generate_government_document()`: 生成政府版文档
- `export_pdf()`: 导出PDF文档

### DocumentGenerator
文档生成器类，负责生成不同类型的报告文档。

**主要方法**:
- `generate_citizen_document()`: 生成市民版文档
- `generate_government_document()`: 生成政府版文档
- `export_pdf()`: 导出PDF文档

### AirQualityVisualizer
可视化组件类，提供各种图表和可视化功能。

**主要方法**:
- `render_prediction_charts()`: 渲染预测图表
- `render_air_quality_heatmap()`: 渲染空气质量热力图
- `render_historical_trends()`: 渲染历史趋势图
- `render_comparison_charts()`: 渲染对比图表
- `render_interactive_dashboard()`: 渲染交互式仪表板

## 使用方法

### 启动应用

```bash
# 方法1: 使用启动脚本
python run_app.py

# 方法2: 直接使用streamlit
streamlit run src/web_interface/streamlit_app.py

# 方法3: 使用Python模块
python -m src.web_interface.streamlit_app
```

### 数据输入格式

CSV文件应包含以下列：
- `timestamp`: 时间戳 (YYYY-MM-DD HH:MM格式)
- `pm25`: PM2.5浓度 (µg/m³)
- `temperature`: 温度 (°C)
- `humidity`: 湿度 (%)
- `wind_speed`: 风速 (m/s)
- `wind_direction`: 风向 (度)

### 热力图数据格式

热力图CSV文件应包含以下列：
- `latitude`: 纬度
- `longitude`: 经度
- `pm25`: PM2.5浓度 (µg/m³)
- `location`: 监测点名称 (可选)

## 依赖要求

### 必需依赖
- `streamlit >= 1.28.0`: Web界面框架
- `pandas >= 2.0.0`: 数据处理
- `numpy >= 1.24.0`: 数值计算

### 可选依赖
- `plotly >= 5.15.0`: 交互式图表 (推荐)
- `reportlab >= 4.0.0`: PDF生成功能

### 系统模块依赖
- `src.prediction_engine`: 预测引擎
- `src.llm_analyzer`: LLM分析器
- `src.config`: 配置管理

## 配置说明

应用使用以下配置项（通过`src.config`模块）：
- `MODEL_PATH`: 模型存储路径
- `OLLAMA_BASE_URL`: Ollama服务地址
- `OLLAMA_MODEL`: 使用的LLM模型名称

## 功能特点

### 🎨 用户体验
- 响应式设计，支持不同屏幕尺寸
- 直观的标签页布局
- 实时状态反馈
- 友好的错误提示

### 📊 数据可视化
- 多种图表类型支持
- 交互式图表操作
- 自定义颜色方案
- 导出功能支持

### 📄 文档生成
- 专业的报告格式
- 差异化内容设计
- PDF导出支持
- 多语言支持（中文）

### 🗺️ 地理可视化
- 空气质量热力图
- 自定义数据支持
- 交互式地图操作
- 多种地图样式

## 故障排除

### 常见问题

1. **图表不显示**
   - 确保安装了plotly库：`pip install plotly`
   - 检查浏览器JavaScript是否启用

2. **PDF导出失败**
   - 确保安装了reportlab库：`pip install reportlab`
   - 检查系统字体支持

3. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件完整性

4. **LLM分析失败**
   - 检查Ollama服务是否运行
   - 验证网络连接状态

### 性能优化

1. **大数据集处理**
   - 使用数据采样减少计算量
   - 启用Streamlit缓存机制

2. **图表渲染优化**
   - 限制数据点数量
   - 使用适当的图表类型

## 开发指南

### 添加新功能

1. **新增标签页**
   ```python
   def _render_new_tab(self):
       st.header("新功能")
       # 实现新功能逻辑
   ```

2. **新增可视化组件**
   ```python
   def render_new_chart(self, data):
       # 实现新图表逻辑
       pass
   ```

3. **新增文档类型**
   ```python
   def generate_new_document(self, report):
       # 实现新文档生成逻辑
       return document_content
   ```

### 测试

运行测试脚本验证功能：
```bash
python test_streamlit_app.py
```

## 版本历史

- **v1.0**: 初始版本，实现基础功能
  - 数据输入和预测
  - 分析报告展示
  - 文档生成和导出
  - 基础可视化功能
  - 热力图展示

## 许可证

本模块是AirGuardian项目的一部分，遵循项目整体许可证。