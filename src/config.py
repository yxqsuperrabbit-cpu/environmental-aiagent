"""
AirGuardian 配置模块
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent
    
    # 数据和模型路径
    DATA_PATH = ROOT_DIR / "data"
    MODEL_PATH = ROOT_DIR / "models"
    DOCS_PATH = ROOT_DIR / "docs"
    
    # Ollama配置
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
    
    # 应用配置
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Streamlit配置
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # 模型配置
    LSTM_HIDDEN_SIZE: int = 64
    LSTM_NUM_LAYERS: int = 2
    SEQUENCE_LENGTH: int = 24  # 24小时历史数据
    PREDICTION_HOURS: list = [24, 48, 72]  # 预测24、48、72小时
    
    # 预测精度要求
    MAX_MAE_THRESHOLD: float = 15.0  # PM2.5预测最大平均绝对误差 (µg/m³)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """确保必要的目录存在"""
        cls.DATA_PATH.mkdir(exist_ok=True)
        cls.MODEL_PATH.mkdir(exist_ok=True)
        cls.DOCS_PATH.mkdir(exist_ok=True)

# 初始化时创建必要目录
Config.ensure_directories()