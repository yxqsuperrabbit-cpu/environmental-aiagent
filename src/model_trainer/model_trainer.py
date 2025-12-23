"""
模型训练模块
实现LSTM模型训练和数据要求文档生成功能
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle
import json

from .data_requirements_generator import DataRequirementsGenerator
from src.gpu_utils import get_optimal_device, optimize_for_training, clear_cache

class LSTMModel(nn.Module):
    """
    LSTM时间序列预测模型
    根据需求1.2，使用LSTM架构进行空气质量预测
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出大小
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出大小
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 全连接层
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            
        Returns:
            输出张量 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(dropped)
        
        return output


class ImprovedLSTMModel(nn.Module):
    """
    改进的LSTM模型，包含更复杂的架构
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 output_size: int = 1, dropout: float = 0.3, bidirectional: bool = True):
        """
        初始化改进的LSTM模型
        
        Args:
            input_size: 输入特征数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出大小
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(ImprovedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出大小
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 多层全连接网络
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
            
        Returns:
            输出张量 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 批归一化
        if last_output.size(0) > 1:  # 只有batch_size > 1时才应用批归一化
            normalized = self.batch_norm(last_output)
        else:
            normalized = last_output
        
        # 全连接层
        output = self.fc_layers(normalized)
        
        return output


class ModelTrainer:
    """
    LSTM模型训练器
    负责数据要求文档生成、模型训练、验证和保存
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化模型训练器
        
        Args:
            model_dir: 模型保存目录
        """
        if model_dir is None:
            from src.config import Config
            model_dir = Config.MODEL_PATH
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 初始化数据要求文档生成器
        self.data_requirements_generator = DataRequirementsGenerator()
        
        # 模型相关属性
        self.model = None
        self.scaler = None
        self.training_history = None
        
        # 设备选择 - 使用GPU管理器选择最优设备
        self.device = get_optimal_device()
        
        # 优化设备设置
        optimize_for_training(self.device)
        
        self.logger = self._setup_logger()
        self.logger.info(f"使用设备: {self.device}")
        
        # 模型超参数
        self.sequence_length = 24  # 使用前24小时数据预测
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
    
    def _setup_logger(self):
        """设置日志记录器"""
        import logging
        logger = logging.getLogger('model_trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_data_requirements(self) -> str:
        """
        生成数据要求文档
        
        根据需求1.1，生成包含数据格式、特征要求、质量标准的markdown文档
        
        Returns:
            str: 生成的markdown文档内容
        """
        return self.data_requirements_generator.generate_data_requirements()
    
    def get_data_format_requirements(self) -> Dict[str, Any]:
        """
        获取结构化的数据格式要求
        
        Returns:
            Dict: 数据格式要求字典
        """
        return self.data_requirements_generator.get_data_format_requirements()
    
    def validate_training_data(self, data_path: str) -> Dict[str, bool]:
        """
        验证训练数据是否符合要求
        
        Args:
            data_path: 训练数据文件路径
            
        Returns:
            Dict: 验证结果
        """
        return self.data_requirements_generator.validate_data_format(data_path)
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        数据预处理管道
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            Tuple[X, y, scaler]: 处理后的特征、标签和标准化器
        """
        # 确保数据按时间排序
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # 选择特征列
        feature_columns = ['pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction']
        features = data[feature_columns].values
        
        # 数据标准化
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 创建时间序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            # 使用前sequence_length个时间步作为输入
            X.append(features_scaled[i-self.sequence_length:i])
            # 预测下一个时间步的PM2.5值
            y.append(features_scaled[i, 0])  # PM2.5是第一列
        
        return np.array(X), np.array(y), scaler
    
    def _create_data_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            X: 特征数据
            y: 标签数据
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader: PyTorch数据加载器
        """
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 创建数据加载器
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_model(self, data_path: str) -> 'TrainedModel':
        """
        训练LSTM模型
        
        Args:
            data_path: 训练数据路径
            
        Returns:
            TrainedModel: 训练好的模型对象
        """
        # 验证数据格式
        validation_result = self.validate_training_data(data_path)
        if not all(validation_result.values()):
            raise ValueError(f"数据验证失败: {validation_result}")
        
        # 加载数据
        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # 数据预处理
        X, y, scaler = self._preprocess_data(data)
        
        # 划分训练集和验证集
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 创建数据加载器
        train_loader = self._create_data_loader(X_train, y_train, self.batch_size, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, self.batch_size, shuffle=False)
        
        # 创建模型
        input_size = X.shape[2]  # 特征数量
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # 训练历史记录
        train_losses = []
        val_losses = []
        
        # 训练循环
        self.logger.info(f"开始训练，使用设备: {self.device}")
        self.logger.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        for epoch in range(self.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # 记录损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
                # 每10个epoch清理一次GPU缓存
                if self.device.type == 'cuda':
                    clear_cache()
        
        # 保存训练历史
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': self.epochs
        }
        
        # 创建训练好的模型对象
        metadata = {
            'model_type': 'LSTM',
            'input_size': input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'device_used': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        
        trained_model = TrainedModel(model, scaler, metadata)
        
        # 保存模型和相关组件
        self.model = model
        self.scaler = scaler
        
        # 清理GPU缓存
        clear_cache()
        
        self.logger.info("模型训练完成")
        
        return trained_model
    
    def validate_model(self, model: 'TrainedModel', test_data: pd.DataFrame) -> float:
        """
        验证模型性能
        
        Args:
            model: 训练好的模型
            test_data: 测试数据
            
        Returns:
            float: 平均绝对误差(MAE)
        """
        # 数据预处理
        test_data = test_data.sort_values('timestamp').reset_index(drop=True)
        feature_columns = ['pm25', 'temperature', 'humidity', 'wind_speed', 'wind_direction']
        features = test_data[feature_columns].values
        
        # 使用训练时的标准化器
        features_scaled = model.scaler.transform(features)
        
        # 创建测试序列
        X_test, y_test = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X_test.append(features_scaled[i-self.sequence_length:i])
            y_test.append(features[i, 0])  # 原始PM2.5值用于计算MAE
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # 模型预测
        model.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), self.batch_size):
                batch_X = torch.FloatTensor(X_test[i:i+self.batch_size]).to(self.device)
                batch_pred = model.model(batch_X)
                predictions.extend(batch_pred.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        
        # 反标准化预测结果
        # 创建临时数组用于反标准化
        temp_array = np.zeros((len(predictions), len(feature_columns)))
        temp_array[:, 0] = predictions  # PM2.5在第一列
        predictions_original = model.scaler.inverse_transform(temp_array)[:, 0]
        
        # 计算MAE
        mae = mean_absolute_error(y_test, predictions_original)
        
        return mae
    
    def save_model(self, model: 'TrainedModel', model_name: str) -> bool:
        """
        保存训练好的模型
        
        Args:
            model: 训练好的模型
            model_name: 模型名称
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建模型保存目录
            model_path = self.model_dir / model_name
            model_path.mkdir(exist_ok=True)
            
            # 保存PyTorch模型
            torch.save(model.model.state_dict(), model_path / 'model.pth')
            
            # 保存标准化器
            with open(model_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(model.scaler, f)
            
            # 保存模型元数据
            metadata_with_info = {
                **model.metadata,
                'model_name': model_name,
                'saved_at': datetime.now().isoformat(),
                'model_file': 'model.pth',
                'scaler_file': 'scaler.pkl'
            }
            
            with open(model_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata_with_info, f, indent=2, ensure_ascii=False)
            
            # 保存训练历史（如果存在）
            if self.training_history:
                with open(model_path / 'training_history.json', 'w', encoding='utf-8') as f:
                    json.dump(self.training_history, f, indent=2)
            
            print(f"模型已成功保存到: {model_path}")
            return True
            
        except Exception as e:
            print(f"模型保存失败: {str(e)}")
            return False
    
    def load_model(self, model_name: str) -> Optional['TrainedModel']:
        """
        加载已保存的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            TrainedModel: 加载的模型对象，如果失败则返回None
        """
        try:
            model_path = self.model_dir / model_name
            
            if not model_path.exists():
                print(f"模型目录不存在: {model_path}")
                return None
            
            # 加载元数据
            metadata_file = model_path / 'metadata.json'
            if not metadata_file.exists():
                print(f"元数据文件不存在: {metadata_file}")
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 加载标准化器
            scaler_file = model_path / 'scaler.pkl'
            if not scaler_file.exists():
                print(f"标准化器文件不存在: {scaler_file}")
                return None
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # 重建模型架构
            model = LSTMModel(
                input_size=metadata['input_size'],
                hidden_size=metadata['hidden_size'],
                num_layers=metadata['num_layers']
            ).to(self.device)
            
            # 加载模型权重
            model_file = model_path / 'model.pth'
            if not model_file.exists():
                print(f"模型文件不存在: {model_file}")
                return None
            
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.eval()
            
            # 创建TrainedModel对象
            trained_model = TrainedModel(model, scaler, metadata)
            
            print(f"模型已成功加载: {model_name}")
            return trained_model
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None
    
    def list_saved_models(self) -> list:
        """
        列出所有已保存的模型
        
        Returns:
            list: 模型名称列表
        """
        if not self.model_dir.exists():
            return []
        
        models = []
        for item in self.model_dir.iterdir():
            if item.is_dir() and (item / 'metadata.json').exists():
                models.append(item.name)
        
        return models


class TrainedModel:
    """
    训练好的模型类
    封装模型对象和相关元数据
    """
    
    def __init__(self, model, scaler, metadata: Dict[str, Any]):
        """
        初始化训练好的模型
        
        Args:
            model: 训练好的模型对象
            scaler: 数据标准化器
            metadata: 模型元数据
        """
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.created_at = datetime.now()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        return {
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "has_scaler": self.scaler is not None,
            "model_type": self.metadata.get("model_type", "Unknown"),
            "input_size": self.metadata.get("input_size", 0),
            "hidden_size": self.metadata.get("hidden_size", 0),
            "num_layers": self.metadata.get("num_layers", 0)
        }
    
    def predict(self, input_data: np.ndarray, device: torch.device) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            input_data: 输入数据 (batch_size, seq_len, input_size)
            device: 计算设备
            
        Returns:
            np.ndarray: 预测结果
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).to(device)
            predictions = self.model(input_tensor)
            return predictions.cpu().numpy()