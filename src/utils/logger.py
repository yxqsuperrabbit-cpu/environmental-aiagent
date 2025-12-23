"""
日志配置模块
"""
import logging
import sys
from pathlib import Path
from src.config import Config

def setup_logger(name: str = "air_guardian") -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果不是调试模式）
    if not Config.DEBUG:
        log_dir = Config.ROOT_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "air_guardian.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger