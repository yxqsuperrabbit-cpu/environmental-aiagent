"""
数据要求文档生成模块
根据需求1.1，生成详细的数据要求和处理要求说明
"""
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import os

class DataRequirementsGenerator:
    """数据要求文档生成器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化数据要求文档生成器
        
        Args:
            output_dir: 输出目录，默认为项目docs目录
        """
        if output_dir is None:
            from src.config import Config
            output_dir = Config.DOCS_PATH
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_data_requirements(self) -> str:
        """
        生成数据要求文档
        
        Returns:
            str: 生成的markdown文档内容
        """
        # 生成markdown内容
        markdown_content = self._create_markdown_template()
        
        # 保存到文件
        output_file = self.output_dir / "data_requirements.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return markdown_content
    
    def _create_markdown_template(self) -> str:
        """
        创建数据要求markdown模板
        
        Returns:
            str: markdown模板内容
        """
        template = f"""# AirGuardian 数据要求文档

## 文档信息
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **版本**: 1.0
- **目标**: LSTM时间序列预测模型训练
- **LLM模型**: deepseek-r1 (本地Ollama部署)

## 1. 数据格式要求

### 1.1 基本数据格式
训练数据必须为CSV格式，包含以下必需字段：

| 字段名 | 数据类型 | 单位 | 描述 | 示例 |
|--------|----------|------|------|------|
| timestamp | datetime | - | 时间戳，ISO 8601格式 | 2024-01-01T00:00:00 |
| pm25 | float | µg/m³ | PM2.5浓度值 | 35.5 |
| temperature | float | °C | 温度 | 25.3 |
| humidity | float | % | 相对湿度 | 65.2 |
| wind_speed | float | m/s | 风速 | 3.2 |
| wind_direction | float | 度 | 风向角度 | 180.0 |

### 1.2 数据文件结构
```
data/
├── training_data.csv          # 训练数据
├── validation_data.csv        # 验证数据
└── test_data.csv             # 测试数据
```

### 1.3 时间序列要求
- **时间间隔**: 1小时
- **最小数据量**: 连续30天（720小时）
- **推荐数据量**: 连续365天（8760小时）
- **时间完整性**: 不允许时间间隔缺失

## 2. 数据质量标准

### 2.1 数据完整性要求
- **缺失值比例**: 每个字段缺失值不超过5%
- **连续缺失**: 连续缺失时间不超过3小时
- **时间连续性**: 时间序列必须连续，无跳跃

### 2.2 数据有效性范围

#### PM2.5浓度值
- **有效范围**: 0 - 500 µg/m³
- **异常值定义**: 超出范围或连续12小时无变化
- **处理方式**: 标记为异常值，需要插值处理

#### 气象数据
- **温度范围**: -40°C 到 50°C
- **湿度范围**: 0% 到 100%
- **风速范围**: 0 到 50 m/s
- **风向范围**: 0° 到 360°

### 2.3 数据一致性检查
- **时间戳格式**: 必须为UTC时间，ISO 8601格式
- **数值精度**: PM2.5保留1位小数，气象数据保留2位小数
- **编码格式**: UTF-8编码

## 3. 数据预处理要求

### 3.1 数据清洗步骤
1. **异常值检测**: 使用3σ原则识别异常值
2. **缺失值处理**: 
   - 短期缺失（<3小时）：线性插值
   - 长期缺失（≥3小时）：使用历史同期均值
3. **重复值处理**: 删除完全重复的记录
4. **时间对齐**: 确保所有数据点时间戳对齐到整点

### 3.2 特征工程要求
1. **时间特征提取**:
   - 小时（0-23）
   - 星期几（0-6）
   - 月份（1-12）
   - 季节（1-4）

2. **滞后特征**:
   - PM2.5前1-6小时滞后值
   - 气象参数前1-3小时滞后值

3. **统计特征**:
   - 过去24小时PM2.5均值、最大值、最小值
   - 过去24小时气象参数均值

### 3.3 数据标准化
- **方法**: Z-score标准化
- **应用范围**: 所有数值特征
- **保存要求**: 保存标准化参数用于预测时反标准化

## 4. 训练数据集划分

### 4.1 数据集比例
- **训练集**: 70%（时间序列前70%）
- **验证集**: 15%（时间序列中间15%）
- **测试集**: 15%（时间序列最后15%）

### 4.2 时间窗口设置
- **输入序列长度**: 24小时（24个时间点）
- **预测序列长度**: 72小时（24、48、72小时预测点）
- **滑动窗口步长**: 1小时

## 5. 模型性能要求

### 5.1 精度指标
- **主要指标**: 平均绝对误差（MAE）< 15 µg/m³
- **辅助指标**: 
  - 均方根误差（RMSE）< 20 µg/m³
  - 平均绝对百分比误差（MAPE）< 25%
  - 决定系数（R²）> 0.7

### 5.2 预测时间范围
- **24小时预测**: MAE < 12 µg/m³
- **48小时预测**: MAE < 15 µg/m³
- **72小时预测**: MAE < 18 µg/m³

## 6. 数据存储和管理

### 6.1 文件命名规范
```
YYYY-MM-DD_HH-mm-ss_data_type.csv
例如: 2024-01-01_00-00-00_training.csv
```

### 6.2 版本控制
- 每次数据更新需要记录版本号
- 保留数据处理日志
- 建立数据血缘关系追踪

### 6.3 备份策略
- 原始数据必须备份
- 处理后数据分别存储
- 定期验证数据完整性

## 7. 数据验证检查清单

### 7.1 格式验证
- [ ] CSV文件格式正确
- [ ] 字段名称和数量正确
- [ ] 数据类型符合要求
- [ ] 编码格式为UTF-8

### 7.2 质量验证
- [ ] 缺失值比例在允许范围内
- [ ] 数值范围符合有效性要求
- [ ] 时间序列连续性检查通过
- [ ] 异常值已标记和处理

### 7.3 完整性验证
- [ ] 训练、验证、测试集划分正确
- [ ] 特征工程步骤完成
- [ ] 标准化参数已保存
- [ ] 数据集大小满足最小要求

## 8. 常见问题和解决方案

### 8.1 数据质量问题
**问题**: 数据缺失率过高
**解决方案**: 
1. 检查数据源质量
2. 考虑使用多源数据融合
3. 调整时间窗口大小

**问题**: 异常值过多
**解决方案**:
1. 检查传感器校准状态
2. 使用更严格的异常值检测方法
3. 考虑使用鲁棒性更强的模型

### 8.2 性能问题
**问题**: 模型精度不达标
**解决方案**:
1. 增加训练数据量
2. 调整特征工程策略
3. 优化模型超参数
4. 考虑集成学习方法

## 9. 联系信息
如有数据相关问题，请联系：
- 技术支持：[技术团队邮箱]
- 数据质量：[数据团队邮箱]

---
*本文档由AirGuardian系统自动生成，最后更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return template
    
    def get_data_format_requirements(self) -> Dict[str, Dict]:
        """
        获取数据格式要求的结构化信息
        
        Returns:
            Dict: 数据格式要求字典
        """
        return {
            "required_fields": {
                "timestamp": {
                    "type": "datetime",
                    "format": "ISO 8601",
                    "example": "2024-01-01T00:00:00",
                    "description": "时间戳"
                },
                "pm25": {
                    "type": "float",
                    "unit": "µg/m³",
                    "range": [0, 500],
                    "example": 35.5,
                    "description": "PM2.5浓度值"
                },
                "temperature": {
                    "type": "float",
                    "unit": "°C",
                    "range": [-40, 50],
                    "example": 25.3,
                    "description": "温度"
                },
                "humidity": {
                    "type": "float",
                    "unit": "%",
                    "range": [0, 100],
                    "example": 65.2,
                    "description": "相对湿度"
                },
                "wind_speed": {
                    "type": "float",
                    "unit": "m/s",
                    "range": [0, 50],
                    "example": 3.2,
                    "description": "风速"
                },
                "wind_direction": {
                    "type": "float",
                    "unit": "度",
                    "range": [0, 360],
                    "example": 180.0,
                    "description": "风向角度"
                }
            },
            "quality_standards": {
                "missing_value_threshold": 0.05,  # 5%
                "continuous_missing_hours": 3,
                "min_data_days": 30,
                "recommended_data_days": 365
            },
            "performance_requirements": {
                "mae_threshold": 15.0,  # µg/m³
                "rmse_threshold": 20.0,
                "mape_threshold": 0.25,
                "r2_threshold": 0.7
            }
        }
    
    def validate_data_format(self, data_path: str) -> Dict[str, bool]:
        """
        验证数据格式是否符合要求
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            Dict: 验证结果
        """
        # 这里可以实现实际的数据验证逻辑
        # 目前返回基本的验证结构
        return {
            "file_exists": os.path.exists(data_path),
            "format_valid": True,  # 需要实际实现
            "fields_complete": True,  # 需要实际实现
            "quality_acceptable": True  # 需要实际实现
        }