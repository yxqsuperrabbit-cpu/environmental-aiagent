"""
Web界面模块
实现WebInterface类，提供标准化的界面组件和文档生成功能
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import io

from src.llm_analyzer.llm_analyzer import AnalysisReport


class WebInterface:
    """
    Web界面类
    
    根据设计文档定义的WebInterface接口实现：
    - 提供数据输入界面
    - 展示分析结果
    - 生成和导出文档
    - 可选的可视化功能
    """
    
    def __init__(self):
        """初始化Web界面"""
        pass
    
    def render_data_input(self) -> Optional[pd.DataFrame]:
        """
        渲染数据输入界面
        
        Returns:
            Optional[pd.DataFrame]: 用户输入的数据，如果输入无效则返回None
        """
        st.subheader("数据输入")
        
        # 数据输入方式选择
        input_method = st.radio(
            "选择数据输入方式",
            ["上传CSV文件", "手动输入数据", "使用示例数据"],
            horizontal=True
        )
        
        if input_method == "上传CSV文件":
            return self._handle_file_upload()
        elif input_method == "手动输入数据":
            return self._handle_manual_input()
        elif input_method == "使用示例数据":
            return self._generate_sample_data()
        
        return None
    
    def display_analysis(self, report: AnalysisReport) -> None:
        """
        展示分析结果
        
        Args:
            report: 分析报告对象
        """
        if not report:
            st.warning("暂无分析报告")
            return
        
        # 报告摘要
        st.subheader("分析报告摘要")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("风险等级", report.risk_level)
        with col2:
            st.metric("生成时间", report.generated_at.strftime('%Y-%m-%d %H:%M'))
        
        # 历史情况分析
        with st.expander("过去24小时情况", expanded=True):
            st.write(report.historical_summary)
        
        # 预测情况分析
        with st.expander("未来预测情况", expanded=True):
            st.write(report.prediction_summary)
        
        # 健康预警
        if report.health_warnings:
            with st.expander("健康预警", expanded=True):
                for warning in report.health_warnings:
                    st.warning(warning)
        
        # 政府建议
        if report.government_recommendations:
            with st.expander("政府建议", expanded=False):
                for i, recommendation in enumerate(report.government_recommendations, 1):
                    st.write(f"{i}. {recommendation}")
        
        # 市民建议
        if report.citizen_recommendations:
            with st.expander("市民建议", expanded=False):
                for i, recommendation in enumerate(report.citizen_recommendations, 1):
                    st.write(f"{i}. {recommendation}")
    
    def generate_citizen_document(self, report: AnalysisReport) -> str:
        """
        生成市民版文档
        
        Args:
            report: 分析报告对象
            
        Returns:
            str: 市民版文档内容
        """
        doc_content = f"""
# 市民版空气质量报告

## 报告摘要
- **生成时间**: {report.generated_at.strftime('%Y年%m月%d日 %H:%M')}
- **整体风险等级**: {report.risk_level}

## 过去24小时情况
{report.historical_summary}

## 未来预测情况
{report.prediction_summary}

## 健康预警
"""
        
        if report.health_warnings:
            for warning in report.health_warnings:
                doc_content += f"- {warning}\n"
        else:
            doc_content += "- 暂无特殊健康预警\n"
        
        doc_content += f"""
## 市民健康建议
"""
        
        if report.citizen_recommendations:
            for i, recommendation in enumerate(report.citizen_recommendations, 1):
                doc_content += f"{i}. {recommendation}\n"
        else:
            doc_content += "1. 请关注空气质量变化，适当调整户外活动\n"
        
        doc_content += f"""
---
*本报告由AirGuardian系统自动生成，仅供参考。如有疑问，请咨询相关专业人士。*
"""
        
        return doc_content
    
    def generate_government_document(self, report: AnalysisReport) -> str:
        """
        生成政府版文档
        
        Args:
            report: 分析报告对象
            
        Returns:
            str: 政府版文档内容
        """
        doc_content = f"""
# 政府版空气质量报告

## 执行摘要
- **报告生成时间**: {report.generated_at.strftime('%Y年%m月%d日 %H:%M')}
- **整体风险等级**: {report.risk_level}

## 空气质量分析

### 历史情况分析
{report.historical_summary}

### 预测情况分析
{report.prediction_summary}

## 风险评估与预警
"""
        
        if report.health_warnings:
            for warning in report.health_warnings:
                doc_content += f"- {warning}\n"
        else:
            doc_content += "- 预测期间空气质量总体良好，无特殊预警\n"
        
        doc_content += f"""
## 政策建议与应对措施
"""
        
        if report.government_recommendations:
            for i, recommendation in enumerate(report.government_recommendations, 1):
                doc_content += f"{i}. {recommendation}\n"
        else:
            doc_content += "1. 继续监测空气质量变化，保持现有管控措施\n"
        
        doc_content += f"""
---
*本报告由AirGuardian智能分析系统生成，供政府决策参考使用。*
"""
        
        return doc_content
    
    def export_pdf(self, document: str) -> bytes:
        """
        导出PDF文档
        
        Args:
            document: 文档内容
            
        Returns:
            bytes: PDF文件字节数据
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            # 创建内存缓冲区
            buffer = io.BytesIO()
            
            # 创建PDF文档
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # 创建内容列表
            story = []
            
            # 处理内容
            lines = document.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('# '):
                        # 一级标题
                        story.append(Paragraph(line[2:], styles['Heading1']))
                    elif line.startswith('## '):
                        # 二级标题
                        story.append(Paragraph(line[3:], styles['Heading2']))
                    elif line.startswith('### '):
                        # 三级标题
                        story.append(Paragraph(line[4:], styles['Heading3']))
                    elif line.startswith('- '):
                        # 列表项
                        story.append(Paragraph(line[2:], styles['Normal']))
                    else:
                        # 普通文本
                        story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            # 构建PDF
            doc.build(story)
            
            # 获取PDF字节
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except ImportError:
            st.error("缺少reportlab库，无法生成PDF")
            return b""
        except Exception as e:
            st.error(f"PDF生成失败: {str(e)}")
            return b""
    
    def _handle_file_upload(self) -> Optional[pd.DataFrame]:
        """处理文件上传"""
        uploaded_file = st.file_uploader(
            "上传包含过去24小时数据的CSV文件",
            type=['csv'],
            help="CSV文件应包含以下列：timestamp, pm25, temperature, humidity, wind_speed, wind_direction"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # 验证必需列
                required_columns = ['timestamp', 'pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"CSV文件缺少必需的列: {', '.join(missing_columns)}")
                    return None
                
                # 转换时间戳
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 验证数据长度
                if len(df) < 24:
                    st.warning(f"数据不足24小时（当前：{len(df)}小时），预测精度可能受影响")
                
                st.success(f"成功加载 {len(df)} 条数据记录")
                return df
                
            except Exception as e:
                st.error(f"文件读取失败: {str(e)}")
                return None
        
        return None
    
    def _handle_manual_input(self) -> Optional[pd.DataFrame]:
        """处理手动输入"""
        st.info("请输入过去24小时的空气质量数据（至少需要24个数据点）")
        
        # 创建数据输入表格
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = self._create_empty_dataframe(24)
        
        # 数据编辑器
        edited_data = st.data_editor(
            st.session_state.manual_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "时间戳",
                    help="数据记录时间",
                    format="YYYY-MM-DD HH:mm"
                ),
                "pm25": st.column_config.NumberColumn(
                    "PM2.5 (µg/m³)",
                    help="PM2.5浓度",
                    min_value=0,
                    max_value=500,
                    step=0.1
                ),
                "temperature": st.column_config.NumberColumn(
                    "温度 (°C)",
                    help="环境温度",
                    min_value=-50,
                    max_value=50,
                    step=0.1
                ),
                "humidity": st.column_config.NumberColumn(
                    "湿度 (%)",
                    help="相对湿度",
                    min_value=0,
                    max_value=100,
                    step=1
                ),
                "wind_speed": st.column_config.NumberColumn(
                    "风速 (m/s)",
                    help="风速",
                    min_value=0,
                    max_value=50,
                    step=0.1
                ),
                "wind_direction": st.column_config.NumberColumn(
                    "风向 (度)",
                    help="风向角度",
                    min_value=0,
                    max_value=360,
                    step=1
                )
            }
        )
        
        # 验证数据完整性
        if len(edited_data) >= 24:
            # 检查是否有空值
            if edited_data.isnull().any().any():
                st.warning("数据中存在空值，请填写完整")
                return None
            else:
                st.session_state.manual_data = edited_data
                return edited_data
        else:
            st.warning(f"需要至少24小时的数据（当前：{len(edited_data)}小时）")
            return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """生成示例数据"""
        import numpy as np
        from datetime import timedelta
        
        # 生成过去24小时的时间戳
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = [end_time - timedelta(hours=i) for i in range(24, 0, -1)]
        
        # 生成模拟的空气质量数据
        np.random.seed(42)  # 确保可重现性
        
        # 基础PM2.5值，模拟日变化
        base_pm25 = 50 + 20 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 10, 24)
        base_pm25 = np.maximum(base_pm25, 5)  # 确保不为负值
        
        # 温度变化
        base_temp = 20 + 5 * np.sin(np.linspace(0, 2*np.pi, 24) - np.pi/2) + np.random.normal(0, 2, 24)
        
        # 湿度变化
        base_humidity = 60 + 20 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi) + np.random.normal(0, 5, 24)
        base_humidity = np.clip(base_humidity, 10, 90)
        
        # 风速变化
        base_wind_speed = 3 + 2 * np.random.exponential(1, 24)
        base_wind_speed = np.clip(base_wind_speed, 0.1, 15)
        
        # 风向变化
        base_wind_direction = np.random.uniform(0, 360, 24)
        
        # 创建DataFrame
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'pm25': base_pm25.round(1),
            'temperature': base_temp.round(1),
            'humidity': base_humidity.round(0).astype(int),
            'wind_speed': base_wind_speed.round(1),
            'wind_direction': base_wind_direction.round(0).astype(int)
        })
        
        st.info("已生成示例数据（过去24小时）")
        return sample_data
    
    def _create_empty_dataframe(self, rows: int) -> pd.DataFrame:
        """创建空的数据框架"""
        from datetime import timedelta
        
        # 生成默认时间戳（过去24小时）
        end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = [end_time - timedelta(hours=i) for i in range(rows, 0, -1)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'pm25': [None] * rows,
            'temperature': [None] * rows,
            'humidity': [None] * rows,
            'wind_speed': [None] * rows,
            'wind_direction': [None] * rows
        })