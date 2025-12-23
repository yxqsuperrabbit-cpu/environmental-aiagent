"""
可视化模块
实现空气质量数据的可视化功能，包括热力图和预测结果图表展示
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from src.prediction_engine.prediction_engine import PredictionResult


class AirQualityVisualizer:
    """
    空气质量可视化器类
    
    根据需求3.7, 4.1实现：
    - 3.7: 提供可视化图表展示预测结果（可选功能）
    - 4.1: 显示监测区域的空气质量热力图（可选功能）
    """
    
    def __init__(self):
        """初始化可视化器"""
        pass
    
    def render_prediction_charts(self, prediction: PredictionResult) -> None:
        """
        渲染预测结果图表
        
        根据需求3.7：提供可视化图表展示预测结果
        
        Args:
            prediction: 预测结果对象
        """
        if not prediction:
            st.info("请先执行预测以查看可视化图表")
            return
        
        # 预测趋势图
        st.subheader("PM2.5预测趋势")
        self._plot_prediction_trend(prediction)
        
        # 置信区间图
        st.subheader("预测置信区间")
        self._plot_confidence_intervals(prediction)
        
        # 空气质量等级分布
        st.subheader("空气质量等级分布")
        self._plot_air_quality_distribution(prediction)
        
        # 时间序列分解图
        st.subheader("预测数据分析")
        self._plot_prediction_analysis(prediction)
    
    def render_air_quality_heatmap(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        渲染空气质量热力图
        
        根据需求4.1：显示监测区域的空气质量热力图（可选功能）
        
        Args:
            data: 空气质量数据（可选，如果为None则生成示例数据）
        """
        st.subheader("空气质量热力图")
        
        if data is None:
            # 生成示例热力图数据
            data = self._generate_sample_heatmap_data()
            st.info("当前显示的是示例热力图数据")
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # 创建热力图
            fig = go.Figure(data=go.Heatmap(
                z=data['pm25'].values.reshape(10, 10),  # 假设10x10网格
                x=[f"经度{i}" for i in range(10)],
                y=[f"纬度{i}" for i in range(10)],
                colorscale=[
                    [0, 'green'],      # 优
                    [0.14, 'lightgreen'],  # 良
                    [0.3, 'yellow'],   # 轻度污染
                    [0.5, 'orange'],   # 中度污染
                    [0.7, 'red'],      # 重度污染
                    [1, 'darkred']     # 严重污染
                ],
                colorbar=dict(
                    title="PM2.5 (µg/m³)",
                    titleside="right"
                ),
                hovertemplate='经度: %{x}<br>纬度: %{y}<br>PM2.5: %{z:.1f} µg/m³<extra></extra>'
            ))
            
            fig.update_layout(
                title="区域空气质量热力图",
                xaxis_title="经度",
                yaxis_title="纬度",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加图例说明
            st.markdown("""
            **空气质量等级说明：**
            - 绿色：优（0-35µg/m³）
            - 黄绿色：良（36-75µg/m³）
            - 黄色：轻度污染（76-115µg/m³）
            - 橙色：中度污染（116-150µg/m³）
            - 红色：重度污染（151-250µg/m³）
            - 深红色：严重污染（>250µg/m³）
            """)
            
        except ImportError:
            st.error("缺少plotly库，无法显示热力图")
            # 提供备用的简单表格显示
            st.subheader("区域空气质量数据表")
            st.dataframe(data, use_container_width=True)
        except Exception as e:
            st.error(f"热力图生成失败: {str(e)}")
    
    def render_historical_trends(self, historical_data: pd.DataFrame) -> None:
        """
        渲染历史趋势图
        
        Args:
            historical_data: 历史数据
        """
        if historical_data is None or len(historical_data) == 0:
            st.info("请先输入历史数据以查看趋势图")
            return
        
        st.subheader("历史数据趋势")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('PM2.5浓度', '温度', '湿度', '风速'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # PM2.5趋势
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['pm25'],
                    mode='lines+markers',
                    name='PM2.5',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 温度趋势
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['temperature'],
                    mode='lines+markers',
                    name='温度',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # 湿度趋势
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['humidity'],
                    mode='lines+markers',
                    name='湿度',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # 风速趋势
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['wind_speed'],
                    mode='lines+markers',
                    name='风速',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_xaxes(title_text="时间", row=1, col=1)
            fig.update_xaxes(title_text="时间", row=1, col=2)
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_xaxes(title_text="时间", row=2, col=2)
            
            fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
            fig.update_yaxes(title_text="温度 (°C)", row=1, col=2)
            fig.update_yaxes(title_text="湿度 (%)", row=2, col=1)
            fig.update_yaxes(title_text="风速 (m/s)", row=2, col=2)
            
            fig.update_layout(
                title="历史环境数据趋势",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示趋势图")
        except Exception as e:
            st.error(f"趋势图生成失败: {str(e)}")
    
    def render_comparison_charts(self, prediction: PredictionResult, historical_data: pd.DataFrame) -> None:
        """
        渲染历史与预测对比图
        
        Args:
            prediction: 预测结果
            historical_data: 历史数据
        """
        if not prediction or historical_data is None:
            st.info("需要历史数据和预测结果才能显示对比图")
            return
        
        st.subheader("历史与预测对比")
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # 添加历史数据
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['pm25'],
                mode='lines+markers',
                name='历史数据',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # 添加预测数据
            fig.add_trace(go.Scatter(
                x=[ts.strftime('%Y-%m-%d %H:%M') for ts in prediction.timestamps],
                y=prediction.pm25_predictions,
                mode='lines+markers',
                name='预测数据',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # 添加分界线
            last_historical_time = historical_data['timestamp'].iloc[-1]
            fig.add_vline(
                x=last_historical_time,
                line_dash="dot",
                line_color="gray",
                annotation_text="预测起点"
            )
            
            # 添加空气质量等级参考线
            fig.add_hline(y=35, line_dash="dash", line_color="green", 
                         annotation_text="优良分界线")
            fig.add_hline(y=75, line_dash="dash", line_color="yellow", 
                         annotation_text="轻度污染分界线")
            fig.add_hline(y=115, line_dash="dash", line_color="orange", 
                         annotation_text="中度污染分界线")
            
            fig.update_layout(
                title="历史数据与预测结果对比",
                xaxis_title="时间",
                yaxis_title="PM2.5浓度 (µg/m³)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示对比图")
        except Exception as e:
            st.error(f"对比图生成失败: {str(e)}")
    
    def _plot_prediction_trend(self, prediction: PredictionResult) -> None:
        """绘制预测趋势图"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # 添加预测线
            fig.add_trace(go.Scatter(
                x=[ts.strftime('%m-%d %H:%M') for ts in prediction.timestamps],
                y=prediction.pm25_predictions,
                mode='lines+markers',
                name='PM2.5预测值',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                hovertemplate='时间: %{x}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
            ))
            
            # 添加空气质量等级参考线
            fig.add_hline(y=35, line_dash="dash", line_color="green", 
                         annotation_text="优良分界线 (35µg/m³)")
            fig.add_hline(y=75, line_dash="dash", line_color="yellow", 
                         annotation_text="轻度污染分界线 (75µg/m³)")
            fig.add_hline(y=115, line_dash="dash", line_color="orange", 
                         annotation_text="中度污染分界线 (115µg/m³)")
            fig.add_hline(y=150, line_dash="dash", line_color="red", 
                         annotation_text="重度污染分界线 (150µg/m³)")
            
            fig.update_layout(
                title="PM2.5浓度预测趋势",
                xaxis_title="时间",
                yaxis_title="PM2.5浓度 (µg/m³)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示图表")
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")
    
    def _plot_confidence_intervals(self, prediction: PredictionResult) -> None:
        """绘制置信区间图"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            timestamps_str = [ts.strftime('%m-%d %H:%M') for ts in prediction.timestamps]
            
            # 添加置信区间填充
            upper_bounds = [ci[1] for ci in prediction.confidence_intervals]
            lower_bounds = [ci[0] for ci in prediction.confidence_intervals]
            
            fig.add_trace(go.Scatter(
                x=timestamps_str + timestamps_str[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%置信区间',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # 添加预测线
            fig.add_trace(go.Scatter(
                x=timestamps_str,
                y=prediction.pm25_predictions,
                mode='lines',
                name='预测值',
                line=dict(color='blue', width=2),
                hovertemplate='时间: %{x}<br>预测值: %{y:.1f} µg/m³<extra></extra>'
            ))
            
            fig.update_layout(
                title="预测置信区间",
                xaxis_title="时间",
                yaxis_title="PM2.5浓度 (µg/m³)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示图表")
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")
    
    def _plot_air_quality_distribution(self, prediction: PredictionResult) -> None:
        """绘制空气质量等级分布"""
        try:
            import plotly.express as px
            
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
            level_counts = pd.Series(quality_levels).value_counts()
            
            # 创建饼图
            fig = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title="预测期间空气质量等级分布",
                color_discrete_map={
                    '优': 'green',
                    '良': 'lightgreen',
                    '轻度污染': 'yellow',
                    '中度污染': 'orange',
                    '重度污染': 'red',
                    '严重污染': 'darkred'
                }
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='等级: %{label}<br>小时数: %{value}<br>占比: %{percent}<extra></extra>'
            )
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示统计表格
            st.subheader("等级统计")
            stats_df = pd.DataFrame({
                '空气质量等级': level_counts.index,
                '小时数': level_counts.values,
                '占比': [f"{(count/len(quality_levels)*100):.1f}%" for count in level_counts.values]
            })
            st.dataframe(stats_df, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示图表")
        except Exception as e:
            st.error(f"图表生成失败: {str(e)}")
    
    def _plot_prediction_analysis(self, prediction: PredictionResult) -> None:
        """绘制预测数据分析图"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('预测值分布', '不确定性变化', '小时变化模式', '预测精度评估'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 预测值分布直方图
            fig.add_trace(
                go.Histogram(
                    x=prediction.pm25_predictions,
                    nbinsx=20,
                    name='预测值分布',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # 不确定性变化
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(prediction.uncertainty_ranges))),
                    y=prediction.uncertainty_ranges,
                    mode='lines+markers',
                    name='不确定性',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # 小时变化模式
            hours = [ts.hour for ts in prediction.timestamps]
            hourly_avg = {}
            for hour, pm25 in zip(hours, prediction.pm25_predictions):
                if hour not in hourly_avg:
                    hourly_avg[hour] = []
                hourly_avg[hour].append(pm25)
            
            avg_by_hour = {hour: np.mean(values) for hour, values in hourly_avg.items()}
            
            fig.add_trace(
                go.Scatter(
                    x=list(avg_by_hour.keys()),
                    y=list(avg_by_hour.values()),
                    mode='lines+markers',
                    name='小时平均值',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # 预测精度评估（置信区间宽度）
            ci_widths = [ci[1] - ci[0] for ci in prediction.confidence_intervals]
            time_periods = ['0-24h', '24-48h', '48-72h']
            period_widths = []
            
            for i in range(0, len(ci_widths), 24):
                period_data = ci_widths[i:i+24]
                if period_data:
                    period_widths.append(np.mean(period_data))
            
            # 确保有足够的数据
            while len(period_widths) < len(time_periods):
                period_widths.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=time_periods[:len(period_widths)],
                    y=period_widths,
                    name='置信区间宽度',
                    marker_color='orange'
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_xaxes(title_text="PM2.5浓度 (µg/m³)", row=1, col=1)
            fig.update_xaxes(title_text="预测步数", row=1, col=2)
            fig.update_xaxes(title_text="小时", row=2, col=1)
            fig.update_xaxes(title_text="时间段", row=2, col=2)
            
            fig.update_yaxes(title_text="频次", row=1, col=1)
            fig.update_yaxes(title_text="不确定性 (µg/m³)", row=1, col=2)
            fig.update_yaxes(title_text="平均PM2.5 (µg/m³)", row=2, col=1)
            fig.update_yaxes(title_text="置信区间宽度 (µg/m³)", row=2, col=2)
            
            fig.update_layout(
                title="预测数据深度分析",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("缺少plotly库，无法显示分析图")
        except Exception as e:
            st.error(f"分析图生成失败: {str(e)}")
    
    def _generate_sample_heatmap_data(self) -> pd.DataFrame:
        """生成示例热力图数据"""
        np.random.seed(42)
        
        # 生成10x10网格的示例数据
        n_points = 100
        latitudes = np.random.uniform(39.8, 40.2, n_points)  # 北京纬度范围
        longitudes = np.random.uniform(116.2, 116.6, n_points)  # 北京经度范围
        
        # 生成PM2.5数据，模拟城市中心污染较重的情况
        center_lat, center_lon = 40.0, 116.4
        distances = np.sqrt((latitudes - center_lat)**2 + (longitudes - center_lon)**2)
        base_pm25 = 80 - distances * 100  # 距离中心越远污染越轻
        pm25_values = base_pm25 + np.random.normal(0, 15, n_points)
        pm25_values = np.maximum(pm25_values, 10)  # 确保不为负值
        
        return pd.DataFrame({
            'latitude': latitudes,
            'longitude': longitudes,
            'pm25': pm25_values,
            'location': [f"监测点{i+1}" for i in range(n_points)]
        })
    
    def render_interactive_dashboard(self, prediction: PredictionResult, historical_data: pd.DataFrame) -> None:
        """
        渲染交互式仪表板
        
        Args:
            prediction: 预测结果
            historical_data: 历史数据
        """
        st.subheader("交互式数据仪表板")
        
        # 创建选项卡
        tab1, tab2, tab3 = st.tabs(["数据概览", "详细分析", "趋势对比"])
        
        with tab1:
            if prediction:
                # 关键指标卡片
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "平均PM2.5",
                        f"{np.mean(prediction.pm25_predictions):.1f} µg/m³",
                        delta=f"{np.mean(prediction.pm25_predictions) - 50:.1f}"
                    )
                with col2:
                    st.metric(
                        "最高PM2.5",
                        f"{np.max(prediction.pm25_predictions):.1f} µg/m³"
                    )
                with col3:
                    st.metric(
                        "预测时长",
                        f"{len(prediction.pm25_predictions)}小时"
                    )
                with col4:
                    risk_hours = sum(1 for pm25 in prediction.pm25_predictions if pm25 > 75)
                    st.metric(
                        "污染小时数",
                        f"{risk_hours}小时",
                        delta=f"{(risk_hours/len(prediction.pm25_predictions)*100):.1f}%"
                    )
                
                # 快速趋势图
                self._plot_prediction_trend(prediction)
        
        with tab2:
            if prediction:
                # 时间段选择器
                time_range = st.selectbox(
                    "选择分析时间段",
                    ["全部时间", "未来24小时", "未来48小时", "未来72小时"]
                )
                
                # 根据选择过滤数据
                if time_range == "未来24小时":
                    end_idx = min(24, len(prediction.pm25_predictions))
                elif time_range == "未来48小时":
                    end_idx = min(48, len(prediction.pm25_predictions))
                elif time_range == "未来72小时":
                    end_idx = min(72, len(prediction.pm25_predictions))
                else:
                    end_idx = len(prediction.pm25_predictions)
                
                # 显示选定时间段的详细分析
                selected_predictions = prediction.pm25_predictions[:end_idx]
                selected_timestamps = prediction.timestamps[:end_idx]
                
                st.write(f"**{time_range}分析结果：**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- 平均浓度：{np.mean(selected_predictions):.1f} µg/m³")
                    st.write(f"- 最高浓度：{np.max(selected_predictions):.1f} µg/m³")
                    st.write(f"- 最低浓度：{np.min(selected_predictions):.1f} µg/m³")
                with col2:
                    pollution_hours = sum(1 for pm25 in selected_predictions if pm25 > 75)
                    st.write(f"- 污染小时数：{pollution_hours}小时")
                    st.write(f"- 污染占比：{(pollution_hours/len(selected_predictions)*100):.1f}%")
                    st.write(f"- 数据点数：{len(selected_predictions)}个")
        
        with tab3:
            if prediction and historical_data is not None:
                self.render_comparison_charts(prediction, historical_data)
            else:
                st.info("需要历史数据和预测结果才能显示趋势对比")