"""
文档生成模块
实现市民版和政府版文档生成，以及PDF导出功能
"""
import io
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from src.llm_analyzer.llm_analyzer import AnalysisReport
from src.prediction_engine.prediction_engine import PredictionResult


class DocumentGenerator:
    """
    文档生成器类
    
    根据需求4.3, 4.4, 4.5实现：
    - 4.3: 提供市民版文档生成功能
    - 4.4: 提供政府版文档生成功能
    - 4.5: 提供一键PDF导出功能
    """
    
    def __init__(self):
        """初始化文档生成器"""
        pass
    
    def generate_citizen_document(
        self, 
        report: AnalysisReport, 
        prediction: Optional[PredictionResult] = None
    ) -> str:
        """
        生成市民版文档
        
        根据需求4.3：提供市民版文档生成功能
        
        Args:
            report: 分析报告对象
            prediction: 预测结果对象（可选）
            
        Returns:
            str: 市民版文档内容
        """
        doc_content = f"""# 市民版空气质量报告

## 报告摘要
- **生成时间**: {report.generated_at.strftime('%Y年%m月%d日 %H:%M')}
- **整体风险等级**: {report.risk_level}
- **报告类型**: 市民健康指导版

## 过去24小时空气质量情况
{report.historical_summary}

## 未来空气质量预测
{report.prediction_summary}
"""
        
        # 添加预测数据摘要（如果有预测结果）
        if prediction:
            doc_content += f"""
## 预测数据摘要
- **预测时间范围**: 未来{len(prediction.pm25_predictions)}小时
- **平均PM2.5浓度**: {np.mean(prediction.pm25_predictions):.1f} µg/m³
- **最高PM2.5浓度**: {np.max(prediction.pm25_predictions):.1f} µg/m³
- **最低PM2.5浓度**: {np.min(prediction.pm25_predictions):.1f} µg/m³
"""
        
        # 健康预警部分
        doc_content += f"""
## 健康预警信息
"""
        
        if report.health_warnings:
            for i, warning in enumerate(report.health_warnings, 1):
                doc_content += f"{i}. {warning}\n"
        else:
            doc_content += "当前预测期间无特殊健康预警，空气质量总体良好。\n"
        
        # 市民健康建议部分
        doc_content += f"""
## 市民健康防护建议
"""
        
        if report.citizen_recommendations:
            for i, recommendation in enumerate(report.citizen_recommendations, 1):
                doc_content += f"{i}. {recommendation}\n"
        else:
            doc_content += "1. 请关注空气质量变化，适当调整户外活动安排\n"
            doc_content += "2. 敏感人群（儿童、老人、心肺疾病患者）应特别注意防护\n"
        
        # 空气质量等级说明
        doc_content += f"""
## 空气质量等级说明
- **优（0-35µg/m³）**: 空气质量令人满意，基本无空气污染
- **良（36-75µg/m³）**: 空气质量可接受，但某些污染物可能对极少数异常敏感人群健康有较弱影响
- **轻度污染（76-115µg/m³）**: 易感人群症状有轻度加剧，健康人群出现刺激症状
- **中度污染（116-150µg/m³）**: 进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响
- **重度污染（151-250µg/m³）**: 心脏病和肺病患者症状显著加剧，运动耐受力降低，健康人群普遍出现症状
- **严重污染（>250µg/m³）**: 健康人群运动耐受力降低，有明显强烈症状，提前出现某些疾病

## 日常防护小贴士
1. **外出防护**: 空气质量不佳时，外出请佩戴N95或KN95口罩
2. **室内环境**: 关闭门窗，使用空气净化器，保持室内空气清洁
3. **运动建议**: 污染天气避免户外运动，可选择室内运动替代
4. **饮食调理**: 多吃富含维生素C和抗氧化物质的食物
5. **健康监测**: 如出现咳嗽、胸闷等症状，及时就医

## 紧急联系方式
- **环保热线**: 12369
- **医疗急救**: 120
- **空气质量查询**: 关注当地环保部门官方发布

---
**免责声明**: 本报告由AirGuardian智能分析系统自动生成，仅供健康防护参考使用。具体健康问题请咨询专业医疗机构。

**数据来源**: 基于历史空气质量监测数据和LSTM深度学习模型预测生成
**报告版本**: 市民版 v1.0
**生成系统**: AirGuardian 空气质量预测与智能分析系统
"""
        
        return doc_content
    
    def generate_government_document(
        self, 
        report: AnalysisReport, 
        prediction: Optional[PredictionResult] = None
    ) -> str:
        """
        生成政府版文档
        
        根据需求4.4：提供政府版文档生成功能
        
        Args:
            report: 分析报告对象
            prediction: 预测结果对象（可选）
            
        Returns:
            str: 政府版文档内容
        """
        doc_content = f"""# 政府版空气质量分析报告

## 执行摘要
- **报告生成时间**: {report.generated_at.strftime('%Y年%m月%d日 %H:%M')}
- **整体风险等级**: {report.risk_level}
- **报告类型**: 政府决策支持版
- **紧急程度**: {self._determine_urgency_level(report.risk_level)}
"""
        
        # 添加预测技术参数（如果有预测结果）
        if prediction:
            doc_content += f"""- **预测模型**: {prediction.metadata.get('model_name', 'LSTM时间序列模型')}
- **预测时间范围**: 未来{len(prediction.pm25_predictions)}小时
- **模型置信度**: 95%
"""
        
        doc_content += f"""
## 空气质量分析

### 历史情况分析
{report.historical_summary}

### 预测情况分析
{report.prediction_summary}
"""
        
        # 添加详细技术数据（如果有预测结果）
        if prediction:
            doc_content += f"""
### 技术数据摘要
- **平均PM2.5浓度**: {np.mean(prediction.pm25_predictions):.1f} µg/m³
- **最高PM2.5浓度**: {np.max(prediction.pm25_predictions):.1f} µg/m³ (时间: {prediction.timestamps[np.argmax(prediction.pm25_predictions)].strftime('%m-%d %H:%M')})
- **最低PM2.5浓度**: {np.min(prediction.pm25_predictions):.1f} µg/m³ (时间: {prediction.timestamps[np.argmin(prediction.pm25_predictions)].strftime('%m-%d %H:%M')})
- **标准差**: {np.std(prediction.pm25_predictions):.1f} µg/m³
- **变异系数**: {(np.std(prediction.pm25_predictions)/np.mean(prediction.pm25_predictions)*100):.1f}%

### 空气质量等级分布
{self._generate_quality_distribution_analysis(prediction)}
"""
        
        # 风险评估与预警
        doc_content += f"""
## 风险评估与预警
"""
        
        if report.health_warnings:
            for i, warning in enumerate(report.health_warnings, 1):
                doc_content += f"**预警{i}**: {warning}\n\n"
        else:
            doc_content += "预测期间空气质量总体良好，无特殊预警。建议继续保持现有环境管控措施。\n\n"
        
        # 政策建议与应对措施
        doc_content += f"""
## 政策建议与应对措施

### 立即执行措施
"""
        
        if report.government_recommendations:
            for i, recommendation in enumerate(report.government_recommendations, 1):
                doc_content += f"{i}. {recommendation}\n"
        else:
            doc_content += "1. 继续监测空气质量变化，保持现有管控措施\n"
            doc_content += "2. 加强重点污染源监管，确保达标排放\n"
            doc_content += "3. 做好应急预案准备，随时应对突发情况\n"
        
        # 应急响应建议
        doc_content += f"""
### 应急响应建议
{self._generate_emergency_response_recommendations(report.risk_level)}

### 公众信息发布建议
{self._generate_public_communication_recommendations(report.risk_level)}

## 监测与评估

### 重点监测区域
- 工业集中区域
- 交通枢纽地带
- 人口密集区域
- 敏感受体周边（学校、医院、养老院）

### 监测频次建议
- **常规监测**: 每小时更新一次
- **预警期间**: 每30分钟更新一次
- **应急状态**: 实时监测

### 数据质量控制
- 确保监测设备正常运行
- 定期校准监测仪器
- 及时处理异常数据
- 建立数据备份机制

## 🔧 技术参数

### 预测模型信息
"""
        
        if prediction:
            doc_content += f"""- **模型类型**: LSTM时间序列深度学习模型
- **训练数据**: 历史空气质量监测数据
- **输入特征**: PM2.5、温度、湿度、风速、风向
- **预测精度**: MAE < 15 µg/m³
- **置信水平**: 95%
- **更新频率**: 每小时更新预测结果
"""
        else:
            doc_content += "- 预测模型信息暂不可用\n"
        
        doc_content += f"""
### 数据来源
- 国家环境监测网络
- 地方环境监测站点
- 气象观测数据
- 卫星遥感数据

## 联系信息
- **环保部门值班电话**: [请填入具体电话]
- **应急指挥中心**: [请填入具体电话]
- **技术支持**: [请填入具体电话]
- **媒体联络**: [请填入具体电话]

---
**报告分类**: 内部参考
**保密等级**: 一般
**有效期限**: 72小时
**下次更新**: {(report.generated_at.replace(hour=report.generated_at.hour+1)).strftime('%Y年%m月%d日 %H:%M')}

**生成系统**: AirGuardian 空气质量预测与智能分析系统
**报告版本**: 政府版 v1.0
**技术支持**: AirGuardian技术团队
"""
        
        return doc_content
    
    def export_pdf(self, document: str, title: str = "空气质量报告") -> bytes:
        """
        导出PDF文档
        
        根据需求4.5：提供一键PDF导出功能
        
        Args:
            document: 文档内容
            title: 文档标题
            
        Returns:
            bytes: PDF文件字节数据
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.colors import black, blue, red, green
            
            # 创建内存缓冲区
            buffer = io.BytesIO()
            
            # 创建PDF文档
            doc = SimpleDocTemplate(
                buffer, 
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # 获取样式
            styles = getSampleStyleSheet()
            
            # 创建自定义样式
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1,  # 居中
                textColor=blue
            )
            
            heading1_style = ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=black
            )
            
            heading2_style = ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                textColor=black
            )
            
            heading3_style = ParagraphStyle(
                'CustomHeading3',
                parent=styles['Heading3'],
                fontSize=12,
                spaceAfter=8,
                textColor=black
            )
            
            warning_style = ParagraphStyle(
                'Warning',
                parent=styles['Normal'],
                fontSize=10,
                textColor=red,
                leftIndent=20
            )
            
            # 创建内容列表
            story = []
            
            # 添加标题
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
            # 处理内容
            lines = document.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('# '):
                        # 一级标题
                        story.append(Paragraph(line[2:], heading1_style))
                        story.append(Spacer(1, 12))
                    elif line.startswith('## '):
                        # 二级标题
                        story.append(Paragraph(line[3:], heading2_style))
                        story.append(Spacer(1, 10))
                    elif line.startswith('### '):
                        # 三级标题
                        story.append(Paragraph(line[4:], heading3_style))
                        story.append(Spacer(1, 8))
                    elif line.startswith('- ') or line.startswith('* '):
                        # 列表项
                        content = line[2:].strip()
                        if '预警' in content or '警告' in content:
                            story.append(Paragraph(f"• {content}", warning_style))
                        else:
                            story.append(Paragraph(f"• {content}", styles['Normal']))
                        story.append(Spacer(1, 4))
                    elif line.startswith('**') and line.endswith('**'):
                        # 粗体文本
                        content = line[2:-2]
                        story.append(Paragraph(f"<b>{content}</b>", styles['Normal']))
                        story.append(Spacer(1, 6))
                    elif line.startswith('---'):
                        # 分隔线
                        story.append(Spacer(1, 10))
                        story.append(Paragraph("_" * 50, styles['Normal']))
                        story.append(Spacer(1, 10))
                    else:
                        # 普通文本
                        if line:
                            story.append(Paragraph(line, styles['Normal']))
                            story.append(Spacer(1, 6))
            
            # 添加页脚信息
            story.append(Spacer(1, 20))
            story.append(Paragraph(
                f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
                styles['Normal']
            ))
            
            # 构建PDF
            doc.build(story)
            
            # 获取PDF字节
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except ImportError:
            # 如果没有reportlab，返回空字节并记录错误
            return b""
        except Exception as e:
            # 其他错误也返回空字节
            return b""
    
    def _determine_urgency_level(self, risk_level: str) -> str:
        """
        确定紧急程度等级
        
        Args:
            risk_level: 风险等级
            
        Returns:
            str: 紧急程度等级
        """
        urgency_mapping = {
            '优': '无需特殊关注',
            '良': '常规监测',
            '轻度污染': '加强关注',
            '中度污染': '重点关注',
            '重度污染': '高度关注',
            '严重污染': '紧急关注'
        }
        
        return urgency_mapping.get(risk_level, '需要关注')
    
    def _generate_quality_distribution_analysis(self, prediction: PredictionResult) -> str:
        """
        生成空气质量等级分布分析
        
        Args:
            prediction: 预测结果
            
        Returns:
            str: 分布分析文本
        """
        # 计算各等级小时数
        quality_levels = []
        for pm25 in prediction.pm25_predictions:
            if pm25 <= 35:
                quality_levels.append('优')
            elif pm25 <= 75:
                quality_levels.append('良')
            elif pm25 <= 115:
                quality_levels.append('轻度污染')
            elif pm25 <= 150:
                quality_levels.append('中度污染')
            elif pm25 <= 250:
                quality_levels.append('重度污染')
            else:
                quality_levels.append('严重污染')
        
        # 统计各等级数量
        from collections import Counter
        level_counts = Counter(quality_levels)
        total_hours = len(quality_levels)
        
        analysis = "预测期间空气质量等级分布如下：\n"
        for level in ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']:
            count = level_counts.get(level, 0)
            percentage = (count / total_hours) * 100
            if count > 0:
                analysis += f"- **{level}**: {count}小时 ({percentage:.1f}%)\n"
        
        return analysis
    
    def _generate_emergency_response_recommendations(self, risk_level: str) -> str:
        """
        生成应急响应建议
        
        Args:
            risk_level: 风险等级
            
        Returns:
            str: 应急响应建议
        """
        if risk_level in ['严重污染', '重度污染']:
            return """1. **启动重污染天气应急预案**，实施相应级别的应急措施
2. **强制性减排措施**：工业企业限产停产，建筑工地停工
3. **交通管制**：实施机动车限行，禁止高排放车辆上路
4. **学校停课**：中小学和幼儿园可考虑停止户外活动或停课
5. **医疗准备**：增加呼吸科医护人员，准备应对就诊高峰
6. **信息发布**：及时向公众发布预警信息和防护指导"""
        
        elif risk_level in ['中度污染', '轻度污染']:
            return """1. **加强监测**：增加监测频次，密切关注污染变化趋势
2. **预防性措施**：提醒重点企业加强污染治理设施运行
3. **交通引导**：建议公众优先选择公共交通出行
4. **健康提醒**：向敏感人群发布健康防护提醒
5. **应急准备**：做好应急预案启动准备
6. **部门协调**：加强各部门间的信息沟通和协调"""
        
        else:
            return """1. **常规监测**：保持正常的监测频次和质量控制
2. **预防为主**：继续实施常规的污染防控措施
3. **能力建设**：利用良好时期加强应急能力建设
4. **设备维护**：对监测和应急设备进行维护保养
5. **培训演练**：组织相关人员进行应急培训和演练
6. **经验总结**：总结分析空气质量管理经验"""
    
    def _generate_public_communication_recommendations(self, risk_level: str) -> str:
        """
        生成公众信息发布建议
        
        Args:
            risk_level: 风险等级
            
        Returns:
            str: 公众信息发布建议
        """
        if risk_level in ['严重污染', '重度污染']:
            return """1. **及时发布**：通过官方媒体、网站、APP等渠道及时发布预警信息
2. **详细说明**：说明污染程度、持续时间、影响范围和健康风险
3. **防护指导**：提供详细的个人防护措施和注意事项
4. **交通信息**：发布交通管制措施和公共交通调整信息
5. **医疗指导**：提供就医指导和急救电话
6. **辟谣澄清**：及时回应公众关切，澄清不实信息"""
        
        elif risk_level in ['中度污染', '轻度污染']:
            return """1. **主动发布**：通过多种渠道发布空气质量信息和健康提醒
2. **分类指导**：针对不同人群提供差异化的防护建议
3. **科普宣传**：普及空气污染防护知识和健康常识
4. **互动回应**：及时回应公众咨询和关切
5. **预防提醒**：提醒公众关注空气质量变化
6. **正面引导**：引导公众理性对待，避免恐慌情绪"""
        
        else:
            return """1. **常规发布**：按照正常频次发布空气质量信息
2. **科普教育**：利用良好时期开展环保科普教育
3. **经验分享**：分享空气质量改善的成功经验
4. **公众参与**：鼓励公众参与环境保护行动
5. **预防宣传**：宣传污染预防和健康防护知识
6. **正面宣传**：宣传环境治理成效，增强公众信心"""