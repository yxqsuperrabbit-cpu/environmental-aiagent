"""
预测引擎模块
实现空气质量预测功能，包括模型加载、预测执行和结果格式化
"""
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import json
import pickle

from src.model_trainer.model_trainer import LSTMModel, ImprovedLSTMModel, TrainedModel
from src.gpu_utils import get_optimal_device, optimize_for_training


@dataclass
class PredictionResult:
    """
    预测结果数据类
    根据设计文档定义的预测结果格式
    """
    timestamps: List[datetime]  # 未来24-72小时时间点
    pm25_predictions: List[float]  # PM2.5预测值
    confidence_intervals: List[Tuple[float, float]]  # 置信区间
    uncertainty_ranges: List[float]  # 不确定性范围
    metadata: Dict[str, Any]  # 预测元数据


class PredictionEngine:
    """
    预测引擎类
    负责加载训练好的模型并执行空气质量预测
    
    根据需求2.1-2.4实现：
    - 2.1: 通过标准接口加载已训练的模型
    - 2.2: 生成未来24、48和72小时的预报
    - 2.3: 显示置信区间和不确定性范围
    - 2.4: 输出格式便于LLM理解和处理
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化预测引擎
        
        Args:
            model_dir: 模型存储目录
        """
        if model_dir is None:
            from src.config import Config
            model_dir = Config.MODEL_PATH
        
        self.model_dir = Path(model_dir)
        self.loaded_model: Optional[TrainedModel] = None
        self.device = get_optimal_device()
        
        # 优化设备设置
        optimize_for_training(self.device)
        
        # 预测配置
        self.prediction_horizons = [24, 48, 72]  # 预测时间范围（小时）
        self.confidence_level = 0.95  # 置信水平
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的模型
        
        根据需求2.1：通过标准接口加载已训练的模型
        
        Args:
            model_path: 模型路径（可以是模型名称或完整路径）
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 如果是模型名称，构建完整路径
            if not Path(model_path).is_absolute():
                full_model_path = self.model_dir / model_path
            else:
                full_model_path = Path(model_path)
            
            if not full_model_path.exists():
                print(f"模型路径不存在: {full_model_path}")
                return False
            
            # 加载元数据
            metadata_file = full_model_path / 'metadata.json'
            if not metadata_file.exists():
                print(f"元数据文件不存在: {metadata_file}")
                return False
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 加载标准化器
            scaler_file = full_model_path / 'scaler.pkl'
            if not scaler_file.exists():
                print(f"标准化器文件不存在: {scaler_file}")
                return False
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # 重建模型架构
            model_type = metadata.get('model_type', 'LSTM')
            bidirectional = metadata.get('bidirectional', False)
            
            if model_type == 'ImprovedLSTM':
                model = ImprovedLSTMModel(
                    input_size=metadata['input_size'],
                    hidden_size=metadata['hidden_size'],
                    num_layers=metadata['num_layers'],
                    bidirectional=bidirectional
                ).to(self.device)
            else:
                model = LSTMModel(
                    input_size=metadata['input_size'],
                    hidden_size=metadata['hidden_size'],
                    num_layers=metadata['num_layers'],
                    bidirectional=bidirectional
                ).to(self.device)
            
            # 加载模型权重
            model_file = full_model_path / 'model.pth'
            if not model_file.exists():
                print(f"模型文件不存在: {model_file}")
                return False
            
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.eval()
            
            # 创建TrainedModel对象
            self.loaded_model = TrainedModel(model, scaler, metadata)
            
            print(f"模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
    
    def predict(self, historical_data: pd.DataFrame) -> PredictionResult:
        """
        执行空气质量预测
        
        根据需求2.2和2.3：
        - 生成未来24、48和72小时的预报
        - 显示置信区间和不确定性范围
        
        Args:
            historical_data: 前24小时的历史数据
            
        Returns:
            PredictionResult: 预测结果对象
            
        Raises:
            ValueError: 当模型未加载或数据格式不正确时
        """
        if self.loaded_model is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        # 验证输入数据格式
        self._validate_input_data(historical_data)
        
        # 数据预处理
        processed_data = self._preprocess_input_data(historical_data)
        
        # 执行多步预测
        predictions = self._execute_multi_step_prediction(processed_data)
        
        # 计算置信区间和不确定性
        confidence_intervals, uncertainty_ranges = self._calculate_uncertainty(predictions)
        
        # 生成时间戳
        timestamps = self._generate_prediction_timestamps(historical_data)
        
        # 创建预测结果对象
        result = PredictionResult(
            timestamps=timestamps,
            pm25_predictions=predictions.tolist(),
            confidence_intervals=confidence_intervals,
            uncertainty_ranges=uncertainty_ranges,
            metadata={
                'model_name': self.loaded_model.metadata.get('model_name', 'unknown'),
                'prediction_time': datetime.now().isoformat(),
                'input_data_points': len(historical_data),
                'prediction_horizons': self.prediction_horizons,
                'confidence_level': self.confidence_level
            }
        )
        
        return result
    
    def format_for_llm(self, prediction: PredictionResult) -> Dict[str, Any]:
        """
        格式化预测结果供LLM使用
        
        根据需求2.4：输出格式便于LLM理解和处理
        
        Args:
            prediction: 预测结果对象
            
        Returns:
            Dict: 格式化的预测数据
        """
        # 按时间范围组织数据
        formatted_data = {
            'prediction_summary': {
                'total_predictions': len(prediction.pm25_predictions),
                'prediction_time_range': f"{prediction.timestamps[0].strftime('%Y-%m-%d %H:%M')} 到 {prediction.timestamps[-1].strftime('%Y-%m-%d %H:%M')}",
                'model_used': prediction.metadata.get('model_name', 'unknown'),
                'generated_at': prediction.metadata.get('prediction_time', '')
            },
            'hourly_predictions': [],
            'daily_summaries': {},
            'risk_analysis': self._analyze_risk_levels(prediction.pm25_predictions),
            'statistical_summary': {
                'min_pm25': float(np.min(prediction.pm25_predictions)),
                'max_pm25': float(np.max(prediction.pm25_predictions)),
                'mean_pm25': float(np.mean(prediction.pm25_predictions)),
                'std_pm25': float(np.std(prediction.pm25_predictions))
            }
        }
        
        # 详细的小时预测数据
        for i, (timestamp, pm25, conf_interval, uncertainty) in enumerate(zip(
            prediction.timestamps, 
            prediction.pm25_predictions,
            prediction.confidence_intervals,
            prediction.uncertainty_ranges
        )):
            hourly_data = {
                'hour': i + 1,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                'pm25_prediction': round(pm25, 2),
                'confidence_interval': {
                    'lower': round(conf_interval[0], 2),
                    'upper': round(conf_interval[1], 2)
                },
                'uncertainty': round(uncertainty, 2),
                'air_quality_level': self._get_air_quality_level(pm25),
                'health_risk': self._get_health_risk_level(pm25)
            }
            formatted_data['hourly_predictions'].append(hourly_data)
        
        # 按天汇总数据
        current_date = None
        daily_predictions = []
        
        for i, timestamp in enumerate(prediction.timestamps):
            date_str = timestamp.strftime('%Y-%m-%d')
            
            if current_date != date_str:
                if daily_predictions:  # 保存前一天的数据
                    formatted_data['daily_summaries'][current_date] = {
                        'date': current_date,
                        'avg_pm25': round(np.mean(daily_predictions), 2),
                        'min_pm25': round(np.min(daily_predictions), 2),
                        'max_pm25': round(np.max(daily_predictions), 2),
                        'dominant_air_quality_level': self._get_dominant_air_quality_level(daily_predictions),
                        'hours_count': len(daily_predictions)
                    }
                
                current_date = date_str
                daily_predictions = []
            
            daily_predictions.append(prediction.pm25_predictions[i])
        
        # 处理最后一天的数据
        if daily_predictions and current_date:
            formatted_data['daily_summaries'][current_date] = {
                'date': current_date,
                'avg_pm25': round(np.mean(daily_predictions), 2),
                'min_pm25': round(np.min(daily_predictions), 2),
                'max_pm25': round(np.max(daily_predictions), 2),
                'dominant_air_quality_level': self._get_dominant_air_quality_level(daily_predictions),
                'hours_count': len(daily_predictions)
            }
        
        return formatted_data
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        验证输入数据格式
        
        Args:
            data: 输入的历史数据
            
        Raises:
            ValueError: 当数据格式不正确时
        """
        required_columns = ['timestamp', 'pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction']
        
        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        
        # 检查数据长度（至少需要24小时数据）
        sequence_length = self.loaded_model.metadata.get('sequence_length', 24)
        if len(data) < sequence_length:
            raise ValueError(f"输入数据不足，需要至少{sequence_length}小时的数据，当前只有{len(data)}小时")
        
        # 检查数据类型
        for col in ['pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"列 {col} 必须是数值类型")
        
        # 检查时间戳格式
        try:
            pd.to_datetime(data['timestamp'])
        except Exception:
            raise ValueError("timestamp列格式不正确，应为有效的日期时间格式")
    
    def _preprocess_input_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        预处理输入数据
        
        Args:
            data: 原始输入数据
            
        Returns:
            np.ndarray: 预处理后的数据
        """
        # 确保数据按时间排序
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # 生成完整的特征集
        features_df = self._generate_features(data)
        
        # 选择模型需要的特征
        feature_names = self.loaded_model.metadata.get('feature_names', [])
        if feature_names:
            # 确保所有特征都存在
            missing_features = [f for f in feature_names if f not in features_df.columns]
            if missing_features:
                raise ValueError(f"缺少特征: {missing_features}")
            features = features_df[feature_names].values
        else:
            # 如果没有特征名称信息，使用所有数值特征
            features = features_df.select_dtypes(include=[np.number]).values
        
        # 使用训练时的标准化器
        features_scaled = self.loaded_model.scaler.transform(features)
        
        # 获取序列长度
        sequence_length = self.loaded_model.metadata.get('sequence_length', 24)
        
        # 取最后sequence_length个时间步作为输入
        input_sequence = features_scaled[-sequence_length:]
        
        return input_sequence
    
    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成完整的特征集
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 包含所有特征的数据框
        """
        features_df = data.copy()
        
        # 确保timestamp是datetime类型
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # 生成PM2.5相关特征
        features_df['pm25_log'] = np.log1p(features_df['pm25'])
        
        # 生成滞后特征
        for lag in [1, 2, 3, 6, 12]:
            features_df[f'pm25_lag_{lag}'] = features_df['pm25'].shift(lag)
        
        # 生成移动平均特征
        for window in [3, 6, 12]:
            features_df[f'pm25_ma_{window}'] = features_df['pm25'].rolling(window=window, min_periods=1).mean()
        
        # 生成时间特征
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day'] = features_df['timestamp'].dt.day
        features_df['month'] = features_df['timestamp'].dt.month
        
        # 生成周期性时间特征
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day'] / 31)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day'] / 31)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # 填充NaN值（主要是滞后特征的前几行）
        features_df = features_df.bfill().ffill()
        
        return features_df
    
    def _execute_multi_step_prediction(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行多步预测
        
        Args:
            input_data: 预处理后的输入数据
            
        Returns:
            np.ndarray: 预测结果数组
        """
        model = self.loaded_model.model
        model.eval()
        
        # 最大预测步数
        max_horizon = max(self.prediction_horizons)
        predictions = []
        
        # 当前输入序列
        current_sequence = input_data.copy()
        
        with torch.no_grad():
            for step in range(max_horizon):
                # 准备输入张量
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # 执行预测
                pred = model(input_tensor)
                pred_value = pred.cpu().numpy()[0, 0]
                predictions.append(pred_value)
                
                # 更新序列用于下一步预测
                # 创建新的特征向量（假设其他特征保持最后一个值）
                last_features = current_sequence[-1].copy()
                last_features[0] = pred_value  # 更新PM2.5预测值
                
                # 滑动窗口：移除第一个时间步，添加新预测
                current_sequence = np.vstack([current_sequence[1:], last_features])
        
        # 反标准化预测结果
        predictions = np.array(predictions)
        
        # 获取特征数量
        n_features = self.loaded_model.metadata.get('input_size', 20)
        
        # 创建临时数组用于反标准化
        temp_array = np.zeros((len(predictions), n_features))
        temp_array[:, 0] = predictions  # PM2.5在第一列
        
        # 使用最后一行的其他特征值填充
        for i in range(1, n_features):
            if i < input_data.shape[1]:
                temp_array[:, i] = input_data[-1, i]
            else:
                temp_array[:, i] = 0  # 如果特征不足，用0填充
        
        # 反标准化
        predictions_original = self.loaded_model.scaler.inverse_transform(temp_array)[:, 0]
        
        # 确保PM2.5预测值不为负
        predictions_original = np.maximum(predictions_original, 0.0)
        
        return predictions_original
    
    def _calculate_uncertainty(self, predictions: np.ndarray) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        计算置信区间和不确定性范围
        
        Args:
            predictions: 预测值数组
            
        Returns:
            Tuple: (置信区间列表, 不确定性范围列表)
        """
        # 基于历史模型性能估算不确定性
        # 这里使用简化的方法，实际应用中可以使用更复杂的不确定性量化方法
        
        # 基础不确定性（基于验证集性能）
        base_uncertainty = 5.0  # µg/m³
        
        # 随时间增长的不确定性
        time_factor = np.arange(1, len(predictions) + 1) * 0.1
        
        confidence_intervals = []
        uncertainty_ranges = []
        
        for i, pred in enumerate(predictions):
            # 计算当前时间步的不确定性
            current_uncertainty = base_uncertainty * (1 + time_factor[i])
            uncertainty_ranges.append(current_uncertainty)
            
            # 计算置信区间（假设正态分布）
            z_score = 1.96  # 95%置信水平
            margin = z_score * current_uncertainty
            
            lower_bound = max(0, pred - margin)  # PM2.5不能为负
            upper_bound = pred + margin
            
            # 确保预测值在置信区间内
            if pred < lower_bound:
                lower_bound = pred
            if pred > upper_bound:
                upper_bound = pred
            
            confidence_intervals.append((lower_bound, upper_bound))
        
        return confidence_intervals, uncertainty_ranges
    
    def _generate_prediction_timestamps(self, historical_data: pd.DataFrame) -> List[datetime]:
        """
        生成预测时间戳
        
        Args:
            historical_data: 历史数据
            
        Returns:
            List[datetime]: 预测时间戳列表
        """
        # 获取最后一个时间戳
        last_timestamp = pd.to_datetime(historical_data['timestamp']).iloc[-1]
        
        # 生成未来时间戳
        timestamps = []
        max_horizon = max(self.prediction_horizons)
        
        for hour in range(1, max_horizon + 1):
            future_time = last_timestamp + timedelta(hours=hour)
            timestamps.append(future_time)
        
        return timestamps
    
    def _analyze_risk_levels(self, predictions: List[float]) -> Dict[str, Any]:
        """
        分析风险等级分布
        
        Args:
            predictions: 预测值列表
            
        Returns:
            Dict: 风险分析结果
        """
        risk_counts = {'优': 0, '良': 0, '轻度污染': 0, '中度污染': 0, '重度污染': 0, '严重污染': 0}
        
        for pm25 in predictions:
            level = self._get_air_quality_level(pm25)
            risk_counts[level] += 1
        
        total_hours = len(predictions)
        risk_percentages = {level: (count / total_hours) * 100 for level, count in risk_counts.items()}
        
        # 找出主要风险等级
        dominant_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'risk_distribution': risk_counts,
            'risk_percentages': risk_percentages,
            'dominant_risk_level': dominant_risk,
            'total_hours': total_hours,
            'unhealthy_hours': sum(count for level, count in risk_counts.items() 
                                 if level in ['轻度污染', '中度污染', '重度污染', '严重污染'])
        }
    
    def _get_air_quality_level(self, pm25: float) -> str:
        """
        根据PM2.5值获取空气质量等级
        
        Args:
            pm25: PM2.5浓度值
            
        Returns:
            str: 空气质量等级
        """
        if pm25 <= 35:
            return '优'
        elif pm25 <= 75:
            return '良'
        elif pm25 <= 115:
            return '轻度污染'
        elif pm25 <= 150:
            return '中度污染'
        elif pm25 <= 250:
            return '重度污染'
        else:
            return '严重污染'
    
    def _get_health_risk_level(self, pm25: float) -> str:
        """
        根据PM2.5值获取健康风险等级
        
        Args:
            pm25: PM2.5浓度值
            
        Returns:
            str: 健康风险等级
        """
        if pm25 <= 35:
            return '低风险'
        elif pm25 <= 75:
            return '中等风险'
        elif pm25 <= 115:
            return '敏感人群高风险'
        else:
            return '所有人群高风险'
    
    def _get_dominant_air_quality_level(self, pm25_values: List[float]) -> str:
        """
        获取主要空气质量等级
        
        Args:
            pm25_values: PM2.5值列表
            
        Returns:
            str: 主要空气质量等级
        """
        levels = [self._get_air_quality_level(pm25) for pm25 in pm25_values]
        level_counts = {}
        
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return max(level_counts.items(), key=lambda x: x[1])[0]
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        获取当前加载模型的信息
        
        Returns:
            Dict: 模型信息，如果未加载模型则返回None
        """
        if self.loaded_model is None:
            return None
        
        return {
            'model_loaded': True,
            'model_metadata': self.loaded_model.metadata,
            'device': str(self.device),
            'prediction_horizons': self.prediction_horizons,
            'confidence_level': self.confidence_level
        }
    
    def list_available_models(self) -> List[str]:
        """
        列出可用的模型
        
        Returns:
            List[str]: 可用模型名称列表
        """
        if not self.model_dir.exists():
            return []
        
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and (item / 'metadata.json').exists():
                models.append(item.name)
        
        return models