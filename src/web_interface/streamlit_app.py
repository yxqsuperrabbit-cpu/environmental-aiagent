"""
Streamlit Web界面主应用
实现AirGuardian系统的用户界面，使用集成的系统架构
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import io
import base64
from pathlib import Path

# 导入集成系统
from src.air_guardian_system import AirGuardianSystem, PipelineResult
from src.prediction_engine import PredictionResult
from src.web_interface.document_generator import DocumentGenerator
from src.web_interface.visualization import AirQualityVisualizer
from src.config import Config


class AirGuardianApp:
    """
    AirGuardian Streamlit应用主类
    
    使用集成的AirGuardianSystem实现完整的数据流管道
    """
    
    def __init__(self):
        """初始化应用"""
        # 初始化集成系统
        if 'air_guardian_system' not in st.session_state:
            with st.spinner("正在初始化AirGuardian系统..."):
                st.session_state.air_guardian_system = AirGuardianSystem()
        
        self.system = st.session_state.air_guardian_system
        self.visualizer = AirQualityVisualizer()
        
        # 初始化会话状态
        if 'pipeline_result' not in st.session_state:
            st.session_state.pipeline_result = None
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = None
        if 'system_status' not in st.session_state:
            st.session_state.system_status = None
    
    def run(self):
        """运行Streamlit应用"""
        # 设置页面配置
        st.set_page_config(
            page_title="AirGuardian - 空气质量预测与智能分析系统",
            layout="wide"
        )
        
        # 主标题
        st.title("AirGuardian 空气质量预测与智能分析系统")
        st.markdown("---")
        
        # 侧边栏配置
        self._render_sidebar()
        
        # 主界面内容
        self._render_main_content()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("系统配置")
        
        # 系统状态概览
        st.sidebar.subheader("系统状态")
        system_status = self.system.get_system_status()
        st.session_state.system_status = system_status
        
        if system_status.is_fully_ready:
            st.sidebar.success("系统完全就绪")
        else:
            st.sidebar.warning("系统部分就绪")
        
        # 详细状态
        with st.sidebar.expander("详细状态", expanded=False):
            st.write(f"**模型训练器**: {'就绪' if system_status.model_trainer_ready else '未就绪'}")
            st.write(f"**预测引擎**: {'就绪' if system_status.prediction_engine_ready else '未就绪'}")
            st.write(f"**LLM分析器**: {'就绪' if system_status.llm_analyzer_ready else '未就绪'}")
            st.write(f"**文档生成器**: {'就绪' if system_status.document_generator_ready else '未就绪'}")
            
            if system_status.current_model:
                st.write(f"**当前模型**: {system_status.current_model}")
            
            if system_status.error_messages:
                st.write("**错误信息**:")
                for error in system_status.error_messages:
                    st.write(f"• {error}")
        
        # 模型管理
        st.sidebar.subheader("模型管理")
        available_models = self.system.get_available_models()
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "选择预测模型",
                available_models,
                index=0 if system_status.current_model in available_models else 0,
                help="选择用于空气质量预测的LSTM模型"
            )
            
            # 重新加载模型按钮
            if st.sidebar.button("重新加载模型"):
                with st.sidebar.spinner("正在重新加载模型..."):
                    success = self.system.reload_model(selected_model)
                    if success:
                        st.sidebar.success(f"模型 {selected_model} 重新加载成功")
                        st.rerun()
                    else:
                        st.sidebar.error(f"模型 {selected_model} 重新加载失败")
        else:
            st.sidebar.warning("未找到可用的预测模型")
            st.sidebar.info("请先训练模型或检查模型目录")
        
        # 模型信息
        model_info = self.system.get_model_info()
        if model_info and model_info.get('model_loaded'):
            with st.sidebar.expander("模型详情"):
                metadata = model_info.get('model_metadata', {})
                st.write(f"**模型类型**: {metadata.get('model_type', 'N/A')}")
                st.write(f"**输入维度**: {metadata.get('input_size', 'N/A')}")
                st.write(f"**隐藏层大小**: {metadata.get('hidden_size', 'N/A')}")
                st.write(f"**层数**: {metadata.get('num_layers', 'N/A')}")
                if 'final_val_loss' in metadata:
                    st.write(f"**验证损失**: {metadata['final_val_loss']:.4f}")
        
        # 系统操作
        st.sidebar.subheader("系统操作")
        
        if st.sidebar.button("重置系统"):
            with st.sidebar.spinner("正在重置系统..."):
                self.system.reset_system()
                st.sidebar.success("系统已重置")
                st.rerun()
        
        if st.sidebar.button("健康检查"):
            with st.sidebar.spinner("正在执行健康检查..."):
                health_status = self.system.health_check()
                st.sidebar.json(health_status)
    
    def _render_main_content(self):
        """渲染主要内容区域"""
        # 创建标签页 - 简化为3个主要标签页
        tab1, tab2, tab3 = st.tabs([
            "数据输入与预测", 
            "分析报告", 
            "可视化展示"
        ])
        
        with tab1:
            self._render_prediction_tab()
        
        with tab2:
            self._render_analysis_tab()
        
        with tab3:
            self._render_visualization_tab()
    
    def _render_prediction_tab(self):
        """渲染预测标签页"""
        st.header("空气质量预测")
        
        # 检查系统状态
        system_status = st.session_state.system_status or self.system.get_system_status()
        if not system_status.prediction_engine_ready:
            st.error("预测引擎未就绪，请检查模型加载状态")
            return
        
        # 数据输入方式选择
        st.subheader("数据输入方式")
        input_method = st.radio(
            "选择数据输入方式",
            ["上传CSV文件", "手动输入数据", "使用示例数据"],
            horizontal=True
        )
        
        historical_data = None
        
        if input_method == "上传CSV文件":
            historical_data = self._handle_file_upload()
        elif input_method == "手动输入数据":
            historical_data = self._handle_manual_input()
        elif input_method == "使用示例数据":
            historical_data = self._generate_sample_data()
            st.info("已生成示例数据（过去24小时）")
        
        # 显示输入数据预览
        if historical_data is not None:
            st.subheader("输入数据预览")
            
            # 数据预览选项
            col1, col2 = st.columns([3, 1])
            with col1:
                preview_option = st.selectbox(
                    "选择预览方式",
                    ["最近10条", "最早10条", "全部数据", "自定义范围"],
                    index=0
                )
            with col2:
                if preview_option == "自定义范围":
                    max_rows = len(historical_data)
                    show_rows = st.number_input(
                        "显示行数", 
                        min_value=1, 
                        max_value=max_rows, 
                        value=min(20, max_rows),
                        step=1
                    )
            
            # 根据选择显示数据
            if preview_option == "最近10条":
                display_data = historical_data.tail(10)
                st.caption("显示最近10条数据记录")
            elif preview_option == "最早10条":
                display_data = historical_data.head(10)
                st.caption("显示最早10条数据记录")
            elif preview_option == "全部数据":
                display_data = historical_data
                st.caption(f"显示全部{len(historical_data)}条数据记录")
            else:  # 自定义范围
                display_data = historical_data.tail(show_rows)
                st.caption(f"显示最近{len(display_data)}条数据记录")
            
            # 显示数据表格
            st.dataframe(
                display_data, 
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "时间戳",
                        format="MM-DD HH:mm"
                    ),
                    "pm25": st.column_config.NumberColumn(
                        "PM2.5 (µg/m³)",
                        format="%.1f"
                    ),
                    "temperature": st.column_config.NumberColumn(
                        "温度 (°C)",
                        format="%.1f"
                    ),
                    "humidity": st.column_config.NumberColumn(
                        "湿度 (%)",
                        format="%d"
                    ),
                    "wind_speed": st.column_config.NumberColumn(
                        "风速 (m/s)",
                        format="%.1f"
                    ),
                    "wind_direction": st.column_config.NumberColumn(
                        "风向 (度)",
                        format="%d"
                    )
                }
            )
            
            # 数据统计信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据点数", len(historical_data))
            with col2:
                st.metric("平均PM2.5", f"{historical_data['pm25'].mean():.1f} µg/m³")
            with col3:
                st.metric("最高PM2.5", f"{historical_data['pm25'].max():.1f} µg/m³")
            with col4:
                st.metric("最低PM2.5", f"{historical_data['pm25'].min():.1f} µg/m³")
            
            # 数据质量检查
            with st.expander("数据质量检查", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    # 检查缺失值
                    missing_data = historical_data.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning("发现缺失数据:")
                        for col, count in missing_data.items():
                            if count > 0:
                                st.write(f"• {col}: {count} 个缺失值")
                    else:
                        st.success("数据完整，无缺失值")
                
                with col2:
                    # 检查数据范围
                    st.write("**数据范围检查:**")
                    
                    # PM2.5范围检查
                    pm25_min, pm25_max = historical_data['pm25'].min(), historical_data['pm25'].max()
                    if pm25_min < 0 or pm25_max > 500:
                        st.warning(f"PM2.5值异常: {pm25_min:.1f} - {pm25_max:.1f}")
                    else:
                        st.success(f"PM2.5范围正常: {pm25_min:.1f} - {pm25_max:.1f}")
                    
                    # 温度范围检查
                    temp_min, temp_max = historical_data['temperature'].min(), historical_data['temperature'].max()
                    if temp_min < -50 or temp_max > 50:
                        st.warning(f"温度值异常: {temp_min:.1f} - {temp_max:.1f}")
                    else:
                        st.success(f"温度范围正常: {temp_min:.1f} - {temp_max:.1f}")
                    
                    # 湿度范围检查
                    humidity_min, humidity_max = historical_data['humidity'].min(), historical_data['humidity'].max()
                    if humidity_min < 0 or humidity_max > 100:
                        st.warning(f"湿度值异常: {humidity_min:.0f} - {humidity_max:.0f}")
                    else:
                        st.success(f"湿度范围正常: {humidity_min:.0f} - {humidity_max:.0f}")
            
            # 执行完整管道按钮
            col1, col2 = st.columns(2)
            with col1:
                generate_docs = st.checkbox("生成文档", value=True, help="是否生成市民版和政府版文档")
            
            with col2:
                if st.button("执行完整分析管道", type="primary", use_container_width=True):
                    # 保存历史数据到会话状态
                    st.session_state.historical_data = historical_data
                    self._execute_full_pipeline(historical_data, generate_docs)
        
        # 显示管道执行结果
        if st.session_state.pipeline_result:
            self._display_pipeline_results()
    
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
        
        return sample_data
    
    def _create_empty_dataframe(self, rows: int) -> pd.DataFrame:
        """创建空的数据框架"""
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
    
    def _execute_full_pipeline(self, historical_data: pd.DataFrame, generate_documents: bool = True):
        """执行完整的分析管道"""
        try:
            with st.spinner("正在执行完整分析管道..."):
                # 显示进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 步骤1: 预测
                status_text.text("步骤1/3: 执行空气质量预测...")
                progress_bar.progress(33)
                
                # 步骤2: 分析
                status_text.text("步骤2/3: 执行智能分析...")
                progress_bar.progress(66)
                
                # 步骤3: 文档生成（可选）
                if generate_documents:
                    status_text.text("步骤3/3: 生成文档...")
                progress_bar.progress(100)
                
                # 执行完整管道
                pipeline_result = self.system.full_pipeline(
                    historical_data=historical_data,
                    generate_documents=generate_documents
                )
                
                # 清除进度显示
                progress_bar.empty()
                status_text.empty()
                
                # 保存结果
                st.session_state.pipeline_result = pipeline_result
                
                if pipeline_result.success:
                    st.success(f"完整管道执行成功！耗时: {pipeline_result.execution_time:.2f} 秒")
                else:
                    st.error(f"管道执行失败: {pipeline_result.error_message}")
                
        except Exception as e:
            st.error(f"管道执行异常: {str(e)}")
            st.session_state.pipeline_result = None
    
    def _display_pipeline_results(self):
        """显示管道执行结果"""
        result = st.session_state.pipeline_result
        if not result:
            return
        
        st.subheader("管道执行结果")
        
        if result.success:
            # 执行摘要
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("执行状态", "成功")
            with col2:
                st.metric("执行时间", f"{result.execution_time:.2f}s")
            with col3:
                if result.prediction_result:
                    st.metric("预测小时数", len(result.prediction_result.pm25_predictions))
                else:
                    st.metric("预测小时数", "N/A")
            with col4:
                docs_generated = "是" if (result.citizen_document and result.government_document) else "否"
                st.metric("文档已生成", docs_generated)
            
            # 预测结果摘要
            if result.prediction_result:
                st.subheader("预测结果摘要")
                prediction = result.prediction_result
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均PM2.5", f"{np.mean(prediction.pm25_predictions):.1f} µg/m³")
                with col2:
                    st.metric("最高PM2.5", f"{np.max(prediction.pm25_predictions):.1f} µg/m³")
                with col3:
                    st.metric("最低PM2.5", f"{np.min(prediction.pm25_predictions):.1f} µg/m³")
            
            # 分析结果摘要
            if result.analysis_report:
                st.subheader("分析结果摘要")
                report = result.analysis_report
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("风险等级", report.risk_level)
                with col2:
                    st.metric("健康预警", f"{len(report.health_warnings)} 条")
                with col3:
                    st.metric("建议总数", f"{len(report.government_recommendations + report.citizen_recommendations)} 条")
        
        else:
            st.error(f"管道执行失败: {result.error_message}")
            if result.execution_time:
                st.info(f"执行时间: {result.execution_time:.2f} 秒")
    
    def _render_analysis_tab(self):
        """渲染分析报告标签页"""
        st.header("智能分析报告")
        
        pipeline_result = st.session_state.pipeline_result
        if not pipeline_result or not pipeline_result.analysis_report:
            st.info("请先在\"数据输入与预测\"标签页执行完整分析管道")
            return
        
        report = pipeline_result.analysis_report
        
        # 报告摘要
        st.subheader("报告摘要")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("风险等级", report.risk_level)
        with col2:
            st.metric("生成时间", report.generated_at.strftime('%Y-%m-%d %H:%M'))
        with col3:
            if pipeline_result.execution_time:
                st.metric("分析耗时", f"{pipeline_result.execution_time:.2f}s")
        
        # 历史情况分析
        st.subheader("过去24小时情况")
        st.write(report.historical_summary)
        
        # 预测情况分析
        st.subheader("未来预测情况")
        st.write(report.prediction_summary)
        
        # 健康预警
        if report.health_warnings:
            st.subheader("健康预警")
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
        
        # 文档生成和下载
        self._render_document_section(pipeline_result)
        
        # 分析元数据
        with st.expander("分析详情", expanded=False):
            metadata = report.metadata
            st.json({
                "分析模型": metadata.get('model_used', 'N/A'),
                "分析版本": metadata.get('analysis_version', 'N/A'),
                "生成时间": report.generated_at.isoformat(),
                "风险等级": report.risk_level,
                "预警数量": len(report.health_warnings),
                "政府建议数量": len(report.government_recommendations),
                "市民建议数量": len(report.citizen_recommendations)
            })
    
    def _render_document_section(self, pipeline_result):
        """渲染文档生成部分（作为expander）"""
        with st.expander("文档生成与导出", expanded=False):
            if not pipeline_result or not pipeline_result.analysis_report:
                st.info("需要分析报告才能生成文档")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("市民版文档")
                if pipeline_result.citizen_document:
                    # 文档已生成，提供下载
                    document_generator = DocumentGenerator()
                    pdf_bytes = document_generator.export_pdf(pipeline_result.citizen_document, "市民版空气质量报告")
                    if pdf_bytes:
                        st.download_button(
                            "下载市民版PDF",
                            pdf_bytes,
                            file_name=f"市民版空气质量报告_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # 在expander中显示文档内容
                    with st.expander("查看市民版文档内容", expanded=False):
                        st.text_area("", pipeline_result.citizen_document, height=300, label_visibility="collapsed")
                else:
                    if st.button("生成市民版文档", use_container_width=True):
                        with st.spinner("正在生成市民版文档..."):
                            document_generator = DocumentGenerator()
                            citizen_doc = document_generator.generate_citizen_document(
                                pipeline_result.analysis_report,
                                pipeline_result.prediction_result
                            )
                            # 更新session state
                            pipeline_result.citizen_document = citizen_doc
                            st.rerun()
            
            with col2:
                st.subheader("政府版文档")
                if pipeline_result.government_document:
                    # 文档已生成，提供下载
                    document_generator = DocumentGenerator()
                    pdf_bytes = document_generator.export_pdf(pipeline_result.government_document, "政府版空气质量报告")
                    if pdf_bytes:
                        st.download_button(
                            "下载政府版PDF",
                            pdf_bytes,
                            file_name=f"政府版空气质量报告_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # 在expander中显示文档内容
                    with st.expander("查看政府版文档内容", expanded=False):
                        st.text_area("", pipeline_result.government_document, height=300, label_visibility="collapsed")
                else:
                    if st.button("生成政府版文档", use_container_width=True):
                        with st.spinner("正在生成政府版文档..."):
                            document_generator = DocumentGenerator()
                            government_doc = document_generator.generate_government_document(
                                pipeline_result.analysis_report,
                                pipeline_result.prediction_result
                            )
                            # 更新session state
                            pipeline_result.government_document = government_doc
                            st.rerun()
    
    def _render_visualization_tab(self):
        """渲染可视化标签页"""
        st.header("可视化展示")
        
        pipeline_result = st.session_state.pipeline_result
        
        # 基础预测图表
        if pipeline_result and pipeline_result.prediction_result:
            self.visualizer.render_prediction_charts(pipeline_result.prediction_result)
        else:
            st.info("请先执行完整分析管道以查看预测可视化")
        
        # 历史数据趋势（在expander中）
        if st.session_state.historical_data is not None:
            with st.expander("历史数据趋势", expanded=False):
                self.visualizer.render_historical_trends(st.session_state.historical_data)
    
    def _plot_prediction_trend(self, prediction: PredictionResult):
        """绘制预测趋势图 - 保留兼容性"""
        # 委托给可视化模块
        self.visualizer._plot_prediction_trend(prediction)
    
    def _plot_confidence_intervals(self, prediction: PredictionResult):
        """绘制置信区间图 - 保留兼容性"""
        # 委托给可视化模块
        self.visualizer._plot_confidence_intervals(prediction)
    
    def _plot_air_quality_distribution(self, prediction: PredictionResult):
        """绘制空气质量等级分布 - 保留兼容性"""
        # 委托给可视化模块
        self.visualizer._plot_air_quality_distribution(prediction)


def main():
    """主函数"""
    app = AirGuardianApp()
    app.run()


if __name__ == "__main__":
    main()