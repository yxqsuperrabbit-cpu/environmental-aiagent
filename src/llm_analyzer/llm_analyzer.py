"""
LLM分析模块
实现基于Ollama的智能分析功能，包括预测结果分析、报告生成和建议提供
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
import json
import logging

from src.config import Config


@dataclass
class AnalysisReport:
    """
    分析报告数据类
    根据设计文档定义的分析报告格式
    """
    historical_summary: str  # 过去24小时情况
    prediction_summary: str  # 未来24小时预测
    health_warnings: List[str]  # 健康预警
    government_recommendations: List[str]  # 政府建议
    citizen_recommendations: List[str]  # 市民建议
    risk_level: str  # 风险等级
    generated_at: datetime  # 生成时间
    metadata: Dict[str, Any]  # 分析元数据


class LLMAnalyzer:
    """
    LLM分析器类
    负责使用Ollama API分析预测结果并生成智能报告
    
    根据需求3.1-3.6实现：
    - 3.1: 基于预测数据生成全面的分析总结
    - 3.2: 包含过去24小时的空气质量情况描述
    - 3.3: 包含未来24小时的空气质量预测情况
    - 3.4: 生成相应的预警信息
    - 3.5: 提供政府政策建议
    - 3.6: 提供市民健康指导
    """
    
    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化LLM分析器
        
        Args:
            base_url: Ollama服务地址
            model_name: 使用的模型名称
        """
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model_name = model_name or Config.OLLAMA_MODEL
        self.timeout = 60  # 请求超时时间（秒）
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 健康风险阈值配置
        self.health_thresholds = {
            'good': 35,           # 优
            'moderate': 75,       # 良
            'unhealthy_sensitive': 115,  # 轻度污染
            'unhealthy': 150,     # 中度污染
            'very_unhealthy': 250,  # 重度污染
            'hazardous': float('inf')  # 严重污染
        }
    
    def analyze_prediction(self, prediction_data: Dict[str, Any]) -> AnalysisReport:
        """
        分析预测结果并生成综合报告
        
        根据需求3.1：基于预测数据生成全面的分析总结
        
        Args:
            prediction_data: 格式化的预测数据（来自PredictionEngine.format_for_llm）
            
        Returns:
            AnalysisReport: 完整的分析报告
        """
        try:
            # 生成历史情况描述（需求3.2）
            historical_summary = self._generate_historical_summary(prediction_data)
            
            # 生成预测情况描述（需求3.3）
            prediction_summary = self._generate_prediction_summary(prediction_data)
            
            # 检查健康预警（需求3.4）
            health_warnings = self.check_health_warnings(prediction_data)
            
            # 生成政府建议（需求3.5）
            government_recommendations = self.generate_government_advice(prediction_data)
            
            # 生成市民建议（需求3.6）
            citizen_recommendations = self.generate_citizen_advice(prediction_data)
            
            # 确定整体风险等级
            risk_level = self._determine_overall_risk_level(prediction_data)
            
            # 创建分析报告
            report = AnalysisReport(
                historical_summary=historical_summary,
                prediction_summary=prediction_summary,
                health_warnings=health_warnings,
                government_recommendations=government_recommendations,
                citizen_recommendations=citizen_recommendations,
                risk_level=risk_level,
                generated_at=datetime.now(),
                metadata={
                    'model_used': self.model_name,
                    'analysis_version': '1.0',
                    'prediction_source': prediction_data.get('prediction_summary', {}),
                    'risk_analysis': prediction_data.get('risk_analysis', {})
                }
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析预测结果时出错: {str(e)}")
            # 返回基础报告
            return self._create_fallback_report(prediction_data, str(e))
    
    def _generate_historical_summary(self, prediction_data: Dict[str, Any]) -> str:
        """
        生成过去24小时情况描述
        
        根据需求3.2：包含过去24小时的空气质量情况描述
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 历史情况描述
        """
        try:
            # 构建历史分析提示
            prompt = self._build_historical_analysis_prompt(prediction_data)
            
            # 调用LLM生成分析
            response = self._call_ollama_api(prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"生成历史摘要时出错: {str(e)}")
            return f"历史数据分析暂时不可用。错误信息: {str(e)}"
    
    def _generate_prediction_summary(self, prediction_data: Dict[str, Any]) -> str:
        """
        生成未来24小时预测情况描述
        
        根据需求3.3：包含未来24小时的空气质量预测情况
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 预测情况描述
        """
        try:
            # 构建预测分析提示
            prompt = self._build_prediction_analysis_prompt(prediction_data)
            
            # 调用LLM生成分析
            response = self._call_ollama_api(prompt)
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"生成预测摘要时出错: {str(e)}")
            return f"预测分析暂时不可用。错误信息: {str(e)}"
    
    def check_health_warnings(self, prediction_data: Dict[str, Any]) -> List[str]:
        """
        检查健康预警
        
        根据需求3.4：当空气质量存在健康风险时，生成相应的预警信息
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            List[str]: 健康预警信息列表
        """
        warnings = []
        
        try:
            # 获取统计摘要
            stats = prediction_data.get('statistical_summary', {})
            risk_analysis = prediction_data.get('risk_analysis', {})
            hourly_predictions = prediction_data.get('hourly_predictions', [])
            
            max_pm25 = stats.get('max_pm25', 0)
            mean_pm25 = stats.get('mean_pm25', 0)
            unhealthy_hours = risk_analysis.get('unhealthy_hours', 0)
            
            # 检查严重污染预警
            if max_pm25 > self.health_thresholds['very_unhealthy']:
                warnings.append("🚨 严重污染预警：预测期内PM2.5浓度将超过250µg/m³，所有人群应避免户外活动")
                warnings.append("🏠 建议：关闭门窗，使用空气净化器，避免一切户外运动")
            
            # 检查重度污染预警
            elif max_pm25 > self.health_thresholds['unhealthy']:
                warnings.append("⚠️ 重度污染预警：预测期内PM2.5浓度将超过150µg/m³，建议减少户外活动")
                warnings.append("😷 建议：外出时佩戴N95口罩，减少户外运动时间")
            
            # 检查中度污染预警
            elif max_pm25 > self.health_thresholds['unhealthy_sensitive']:
                warnings.append("⚠️ 中度污染预警：预测期内PM2.5浓度将超过115µg/m³，敏感人群应减少户外活动")
                warnings.append("👥 敏感人群（儿童、老人、心肺疾病患者）应特别注意防护")
            
            # 检查持续污染预警
            if unhealthy_hours > 12:  # 超过12小时不健康
                warnings.append("⏰ 持续污染预警：预测期内将有超过12小时的不健康空气质量")
                warnings.append("📅 建议调整户外活动计划，选择空气质量较好的时段")
            
            # 检查平均浓度预警
            if mean_pm25 > self.health_thresholds['unhealthy_sensitive']:
                warnings.append("📊 整体空气质量预警：预测期内平均PM2.5浓度较高，建议关注空气质量变化")
            
            # 检查夜间污染预警
            night_hours_pollution = self._check_night_pollution(hourly_predictions)
            if night_hours_pollution:
                warnings.append("🌙 夜间污染预警：夜间时段空气质量较差，建议关闭门窗")
            
            # 检查早高峰污染预警
            morning_peak_pollution = self._check_morning_peak_pollution(hourly_predictions)
            if morning_peak_pollution:
                warnings.append("🚗 早高峰污染预警：上午7-9点空气质量较差，建议调整出行时间")
            
            # 检查运动时段预警
            exercise_warnings = self._check_exercise_time_warnings(hourly_predictions)
            warnings.extend(exercise_warnings)
            
            # 如果没有特殊预警，检查是否需要一般性提醒
            if not warnings and max_pm25 > self.health_thresholds['moderate']:
                warnings.append("💡 空气质量提醒：预测期内空气质量可能达到轻度污染，敏感人群请注意防护")
            
            # 如果空气质量良好，给出积极提醒
            if not warnings and max_pm25 <= self.health_thresholds['good']:
                warnings.append("✅ 空气质量良好：预测期内空气质量优良，适合户外活动")
            
        except Exception as e:
            self.logger.error(f"检查健康预警时出错: {str(e)}")
            warnings.append(f"健康预警系统暂时不可用。错误信息: {str(e)}")
        
        return warnings
    
    def generate_government_advice(self, prediction_data: Dict[str, Any]) -> List[str]:
        """
        生成政府建议
        
        根据需求3.5：提供交通管制、工业排放控制、公共活动调整等政策建议
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            List[str]: 政府建议列表
        """
        try:
            # 构建政府建议提示
            prompt = self._build_government_advice_prompt(prediction_data)
            
            # 调用LLM生成建议
            response = self._call_ollama_api(prompt)
            
            # 解析响应为建议列表
            advice_list = self._parse_advice_response(response)
            
            return advice_list
            
        except Exception as e:
            self.logger.error(f"生成政府建议时出错: {str(e)}")
            return [f"政府建议生成暂时不可用。错误信息: {str(e)}"]
    
    def generate_citizen_advice(self, prediction_data: Dict[str, Any]) -> List[str]:
        """
        生成市民建议
        
        根据需求3.6：提供外出防护措施、室内活动建议、敏感人群特别提醒等健康指导
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            List[str]: 市民建议列表
        """
        try:
            # 构建市民建议提示
            prompt = self._build_citizen_advice_prompt(prediction_data)
            
            # 调用LLM生成建议
            response = self._call_ollama_api(prompt)
            
            # 解析响应为建议列表
            advice_list = self._parse_advice_response(response)
            
            return advice_list
            
        except Exception as e:
            self.logger.error(f"生成市民建议时出错: {str(e)}")
            return [f"市民建议生成暂时不可用。错误信息: {str(e)}"]
    
    def _call_ollama_api(self, prompt: str) -> str:
        """
        调用Ollama API
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: API响应内容
            
        Raises:
            Exception: 当API调用失败时
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                raise Exception(f"API调用失败，状态码: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise Exception("API调用超时")
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到Ollama服务")
        except Exception as e:
            raise Exception(f"API调用出错: {str(e)}")
    
    def _build_historical_analysis_prompt(self, prediction_data: Dict[str, Any]) -> str:
        """
        构建历史分析提示
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 历史分析提示
        """
        stats = prediction_data.get('statistical_summary', {})
        
        prompt = f"""
作为空气质量专家，请基于以下数据分析过去24小时的空气质量情况：

预测数据统计：
- 平均PM2.5浓度: {stats.get('mean_pm25', 'N/A')} µg/m³
- 最低PM2.5浓度: {stats.get('min_pm25', 'N/A')} µg/m³
- 最高PM2.5浓度: {stats.get('max_pm25', 'N/A')} µg/m³

请提供一个简洁的历史情况分析，包括：
1. 过去24小时空气质量的总体状况
2. 主要的空气质量变化趋势
3. 可能的影响因素分析

请用中文回答，控制在200字以内。
"""
        return prompt.strip()
    
    def _build_prediction_analysis_prompt(self, prediction_data: Dict[str, Any]) -> str:
        """
        构建预测分析提示
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 预测分析提示
        """
        stats = prediction_data.get('statistical_summary', {})
        risk_analysis = prediction_data.get('risk_analysis', {})
        daily_summaries = prediction_data.get('daily_summaries', {})
        
        prompt = f"""
作为空气质量专家，请基于以下预测数据分析未来24-72小时的空气质量情况：

统计摘要：
- 平均PM2.5浓度: {stats.get('mean_pm25', 'N/A')} µg/m³
- 最低PM2.5浓度: {stats.get('min_pm25', 'N/A')} µg/m³
- 最高PM2.5浓度: {stats.get('max_pm25', 'N/A')} µg/m³

风险分析：
- 主要风险等级: {risk_analysis.get('dominant_risk_level', 'N/A')}
- 不健康小时数: {risk_analysis.get('unhealthy_hours', 'N/A')}小时

每日摘要：
{self._format_daily_summaries(daily_summaries)}

请提供一个详细的预测分析，包括：
1. 未来24-72小时空气质量的整体趋势
2. 重点时段的空气质量状况
3. 可能的变化原因和影响因素

请用中文回答，控制在300字以内。
"""
        return prompt.strip()
    
    def _build_government_advice_prompt(self, prediction_data: Dict[str, Any]) -> str:
        """
        构建政府建议提示
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 政府建议提示
        """
        stats = prediction_data.get('statistical_summary', {})
        risk_analysis = prediction_data.get('risk_analysis', {})
        
        prompt = f"""
作为环境政策专家，请基于以下空气质量预测数据为政府部门提供政策建议：

预测统计：
- 最高PM2.5浓度: {stats.get('max_pm25', 'N/A')} µg/m³
- 平均PM2.5浓度: {stats.get('mean_pm25', 'N/A')} µg/m³
- 主要风险等级: {risk_analysis.get('dominant_risk_level', 'N/A')}
- 不健康小时数: {risk_analysis.get('unhealthy_hours', 'N/A')}小时

请提供具体的政府政策建议，包括但不限于：
1. 交通管制措施
2. 工业排放控制
3. 公共活动调整
4. 应急响应措施
5. 公众信息发布

请以列表形式回答，每条建议独立成行，用"- "开头。控制在10条建议以内。
"""
        return prompt.strip()
    
    def _build_citizen_advice_prompt(self, prediction_data: Dict[str, Any]) -> str:
        """
        构建市民建议提示
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 市民建议提示
        """
        stats = prediction_data.get('statistical_summary', {})
        risk_analysis = prediction_data.get('risk_analysis', {})
        
        prompt = f"""
作为健康专家，请基于以下空气质量预测数据为市民提供健康防护建议：

预测统计：
- 最高PM2.5浓度: {stats.get('max_pm25', 'N/A')} µg/m³
- 平均PM2.5浓度: {stats.get('mean_pm25', 'N/A')} µg/m³
- 主要风险等级: {risk_analysis.get('dominant_risk_level', 'N/A')}
- 不健康小时数: {risk_analysis.get('unhealthy_hours', 'N/A')}小时

请提供具体的市民健康建议，包括但不限于：
1. 外出防护措施
2. 室内活动建议
3. 敏感人群特别提醒
4. 运动和户外活动指导
5. 健康监测建议

请以列表形式回答，每条建议独立成行，用"- "开头。控制在10条建议以内。
"""
        return prompt.strip()
    
    def _format_daily_summaries(self, daily_summaries: Dict[str, Any]) -> str:
        """
        格式化每日摘要
        
        Args:
            daily_summaries: 每日摘要数据
            
        Returns:
            str: 格式化的每日摘要
        """
        if not daily_summaries:
            return "暂无每日摘要数据"
        
        formatted = []
        for date, summary in daily_summaries.items():
            formatted.append(
                f"- {date}: 平均{summary.get('avg_pm25', 'N/A')}µg/m³, "
                f"范围{summary.get('min_pm25', 'N/A')}-{summary.get('max_pm25', 'N/A')}µg/m³, "
                f"主要等级: {summary.get('dominant_air_quality_level', 'N/A')}"
            )
        
        return "\n".join(formatted)
    
    def _parse_advice_response(self, response: str) -> List[str]:
        """
        解析建议响应为列表
        
        Args:
            response: LLM响应内容
            
        Returns:
            List[str]: 建议列表
        """
        if not response:
            return ["暂无建议"]
        
        # 按行分割并过滤空行
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # 提取以"- "开头的建议
        advice_list = []
        for line in lines:
            if line.startswith('- '):
                advice_list.append(line[2:].strip())  # 移除"- "前缀
            elif line.startswith('•'):
                advice_list.append(line[1:].strip())  # 移除"•"前缀
            elif line and not any(line.startswith(prefix) for prefix in ['作为', '请', '基于']):
                # 如果不是以特定前缀开头的说明性文字，也加入建议
                advice_list.append(line)
        
        # 如果没有找到格式化的建议，将整个响应作为一条建议
        if not advice_list:
            advice_list = [response]
        
        return advice_list[:10]  # 限制最多10条建议
    
    def _determine_overall_risk_level(self, prediction_data: Dict[str, Any]) -> str:
        """
        确定整体风险等级
        
        Args:
            prediction_data: 预测数据
            
        Returns:
            str: 整体风险等级
        """
        try:
            stats = prediction_data.get('statistical_summary', {})
            risk_analysis = prediction_data.get('risk_analysis', {})
            
            max_pm25 = stats.get('max_pm25', 0)
            mean_pm25 = stats.get('mean_pm25', 0)
            unhealthy_hours = risk_analysis.get('unhealthy_hours', 0)
            
            # 基于最高浓度确定基础风险等级
            if max_pm25 > self.health_thresholds['very_unhealthy']:
                base_risk = '严重污染'
            elif max_pm25 > self.health_thresholds['unhealthy']:
                base_risk = '重度污染'
            elif max_pm25 > self.health_thresholds['unhealthy_sensitive']:
                base_risk = '中度污染'
            elif max_pm25 > self.health_thresholds['moderate']:
                base_risk = '轻度污染'
            elif max_pm25 > self.health_thresholds['good']:
                base_risk = '良'
            else:
                base_risk = '优'
            
            # 考虑持续时间调整风险等级
            if unhealthy_hours > 24:  # 超过24小时不健康
                if base_risk in ['轻度污染', '良']:
                    base_risk = '中度污染'
            
            return base_risk
            
        except Exception as e:
            self.logger.error(f"确定风险等级时出错: {str(e)}")
            return '未知风险'
    
    def _create_fallback_report(self, prediction_data: Dict[str, Any], error_msg: str) -> AnalysisReport:
        """
        创建备用报告（当主要分析失败时）
        
        Args:
            prediction_data: 预测数据
            error_msg: 错误信息
            
        Returns:
            AnalysisReport: 备用分析报告
        """
        stats = prediction_data.get('statistical_summary', {})
        
        return AnalysisReport(
            historical_summary=f"历史数据分析暂时不可用。平均PM2.5浓度: {stats.get('mean_pm25', 'N/A')} µg/m³",
            prediction_summary=f"预测分析暂时不可用。预测最高PM2.5浓度: {stats.get('max_pm25', 'N/A')} µg/m³",
            health_warnings=[f"分析系统暂时不可用: {error_msg}"],
            government_recommendations=["建议关注官方空气质量监测信息"],
            citizen_recommendations=["建议关注空气质量变化，适当调整户外活动"],
            risk_level=self._determine_overall_risk_level(prediction_data),
            generated_at=datetime.now(),
            metadata={
                'model_used': self.model_name,
                'analysis_version': '1.0',
                'error': error_msg,
                'fallback_mode': True
            }
        )
    
    def test_connection(self) -> bool:
        """
        测试Ollama连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """
        获取可用模型列表
        
        Returns:
            List[str]: 可用模型名称列表
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            else:
                return []
        except Exception:
            return []
    
    def _check_night_pollution(self, hourly_predictions: List[Dict[str, Any]]) -> bool:
        """
        检查夜间污染情况
        
        Args:
            hourly_predictions: 小时预测数据
            
        Returns:
            bool: 是否存在夜间污染
        """
        try:
            night_pollution_count = 0
            for prediction in hourly_predictions:
                timestamp_str = prediction.get('timestamp', '')
                if timestamp_str:
                    hour = int(timestamp_str.split(' ')[1].split(':')[0])
                    # 夜间时段：22:00-06:00
                    if hour >= 22 or hour <= 6:
                        pm25 = prediction.get('pm25_prediction', 0)
                        if pm25 > self.health_thresholds['moderate']:
                            night_pollution_count += 1
            
            # 如果夜间超过3小时污染，则发出预警
            return night_pollution_count >= 3
        except Exception:
            return False
    
    def _check_morning_peak_pollution(self, hourly_predictions: List[Dict[str, Any]]) -> bool:
        """
        检查早高峰污染情况
        
        Args:
            hourly_predictions: 小时预测数据
            
        Returns:
            bool: 是否存在早高峰污染
        """
        try:
            morning_pollution_count = 0
            for prediction in hourly_predictions:
                timestamp_str = prediction.get('timestamp', '')
                if timestamp_str:
                    hour = int(timestamp_str.split(' ')[1].split(':')[0])
                    # 早高峰时段：7:00-9:00
                    if 7 <= hour <= 9:
                        pm25 = prediction.get('pm25_prediction', 0)
                        if pm25 > self.health_thresholds['unhealthy_sensitive']:
                            morning_pollution_count += 1
            
            # 如果早高峰时段有污染，则发出预警
            return morning_pollution_count >= 1
        except Exception:
            return False
    
    def _check_exercise_time_warnings(self, hourly_predictions: List[Dict[str, Any]]) -> List[str]:
        """
        检查运动时段预警
        
        Args:
            hourly_predictions: 小时预测数据
            
        Returns:
            List[str]: 运动时段预警列表
        """
        warnings = []
        try:
            # 检查常见运动时段的空气质量
            exercise_periods = {
                '早晨运动时段(6-8点)': (6, 8),
                '上午运动时段(9-11点)': (9, 11),
                '下午运动时段(16-18点)': (16, 18),
                '晚间运动时段(19-21点)': (19, 21)
            }
            
            for period_name, (start_hour, end_hour) in exercise_periods.items():
                period_pollution = False
                for prediction in hourly_predictions:
                    timestamp_str = prediction.get('timestamp', '')
                    if timestamp_str:
                        hour = int(timestamp_str.split(' ')[1].split(':')[0])
                        if start_hour <= hour <= end_hour:
                            pm25 = prediction.get('pm25_prediction', 0)
                            if pm25 > self.health_thresholds['unhealthy_sensitive']:
                                period_pollution = True
                                break
                
                if period_pollution:
                    warnings.append(f"🏃 运动预警：{period_name}空气质量较差，建议避免户外运动")
            
        except Exception:
            pass
        
        return warnings