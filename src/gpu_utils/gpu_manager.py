"""
计算设备管理工具模块
提供跨平台的计算设备检测、选择和性能监控功能
支持CPU、CUDA GPU、MPS (Apple Silicon)等多种计算设备
"""
import os
import sys
import logging
import platform
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import psutil
import numpy as np

# 尝试导入可选的GPU监控库
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class DeviceInfo:
    """计算设备信息数据类"""
    device_type: str  # 'cpu', 'cuda', 'mps'
    device_id: Optional[int] = None
    name: str = ""
    memory_total: float = 0.0  # GB
    memory_used: float = 0.0   # GB
    memory_free: float = 0.0   # GB
    utilization: float = 0.0   # %
    temperature: float = 0.0   # °C
    power_draw: float = 0.0    # W
    is_available: bool = True
    capabilities: Dict[str, Any] = None


@dataclass
class SystemInfo:
    """系统信息数据类"""
    platform: str
    cpu_count: int
    cpu_usage: float
    memory_total: float  # GB
    memory_used: float   # GB
    memory_available: float  # GB
    devices: List[DeviceInfo]
    pytorch_version: str
    recommended_device: str


class DeviceManager:
    """
    计算设备管理器
    负责设备检测、选择、性能监控和优化配置
    支持CPU、CUDA GPU、MPS等多种计算设备
    """
    
    def __init__(self):
        """初始化设备管理器"""
        self.logger = self._setup_logger()
        self.platform = platform.system().lower()
        
        # 检测可用设备
        self.available_devices = self._detect_devices()
        self.current_device = None
        
        # 设置优化配置
        self._configure_optimizations()
        
        self.logger.info(f"设备管理器初始化完成，检测到 {len(self.available_devices)} 个设备")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('device_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_devices(self) -> List[DeviceInfo]:
        """检测所有可用的计算设备"""
        devices = []
        
        # 1. CPU设备（总是可用）
        cpu_info = self._get_cpu_info()
        devices.append(cpu_info)
        
        # 2. CUDA设备
        if torch.cuda.is_available():
            cuda_devices = self._get_cuda_devices()
            devices.extend(cuda_devices)
        
        # 3. MPS设备（Apple Silicon）
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_info = self._get_mps_info()
            devices.append(mps_info)
        
        return devices
    
    def _get_cpu_info(self) -> DeviceInfo:
        """获取CPU设备信息"""
        memory = psutil.virtual_memory()
        
        return DeviceInfo(
            device_type='cpu',
            name=f"{psutil.cpu_count()} Core CPU",
            memory_total=memory.total / (1024**3),
            memory_used=memory.used / (1024**3),
            memory_free=memory.available / (1024**3),
            utilization=psutil.cpu_percent(interval=0.1),
            is_available=True,
            capabilities={'cores': psutil.cpu_count(), 'threads': psutil.cpu_count(logical=True)}
        )
    
    def _get_cuda_devices(self) -> List[DeviceInfo]:
        """获取CUDA设备信息"""
        devices = []
        
        for i in range(torch.cuda.device_count()):
            try:
                device_info = self._get_single_cuda_device(i)
                devices.append(device_info)
            except Exception as e:
                self.logger.warning(f"获取CUDA设备 {i} 信息失败: {e}")
                # 创建基本设备信息
                devices.append(DeviceInfo(
                    device_type='cuda',
                    device_id=i,
                    name=f"CUDA Device {i}",
                    is_available=False
                ))
        
        return devices
    
    def _get_single_cuda_device(self, device_id: int) -> DeviceInfo:
        """获取单个CUDA设备信息"""
        name = torch.cuda.get_device_name(device_id)
        props = torch.cuda.get_device_properties(device_id)
        
        # 内存信息
        memory_total = props.total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        memory_free = memory_total - memory_reserved
        
        # 尝试获取详细信息
        utilization = 0.0
        temperature = 0.0
        power_draw = 0.0
        
        # 使用GPUtil获取利用率和温度
        if GPUTIL_AVAILABLE:
            try:
                gpu_list = GPUtil.getGPUs()
                if device_id < len(gpu_list):
                    gpu = gpu_list[device_id]
                    utilization = gpu.load * 100
                    temperature = gpu.temperature
            except Exception as e:
                self.logger.debug(f"GPUtil获取设备 {device_id} 信息失败: {e}")
        
        # 使用NVIDIA ML获取更详细信息
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
                
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except:
                    power_draw = 0.0
                    
            except Exception as e:
                self.logger.debug(f"NVIDIA ML获取设备 {device_id} 信息失败: {e}")
        
        return DeviceInfo(
            device_type='cuda',
            device_id=device_id,
            name=name,
            memory_total=memory_total,
            memory_used=memory_allocated,
            memory_free=memory_free,
            utilization=utilization,
            temperature=temperature,
            power_draw=power_draw,
            is_available=True,
            capabilities={
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count,
                'max_threads_per_block': props.max_threads_per_block,
                'warp_size': props.warp_size
            }
        )
    
    def _get_mps_info(self) -> DeviceInfo:
        """获取MPS设备信息（Apple Silicon）"""
        return DeviceInfo(
            device_type='mps',
            name="Apple Metal Performance Shaders",
            is_available=True,
            capabilities={'unified_memory': True}
        )
    
    def _configure_optimizations(self):
        """配置设备优化设置"""
        try:
            # 启用cuDNN基准模式（如果有CUDA）
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                self.logger.info("已启用cuDNN基准模式")
            
            # 设置内存分配策略
            if torch.cuda.is_available():
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
            
        except Exception as e:
            self.logger.warning(f"设备优化配置失败: {e}")
    
    def get_system_info(self) -> SystemInfo:
        """获取完整的系统信息"""
        # CPU和内存信息
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024**3)
        memory_used = memory.used / (1024**3)
        memory_available = memory.available / (1024**3)
        
        # 推荐设备
        recommended = self.select_best_device()
        
        return SystemInfo(
            platform=self.platform,
            cpu_count=cpu_count,
            cpu_usage=cpu_usage,
            memory_total=memory_total,
            memory_used=memory_used,
            memory_available=memory_available,
            devices=self.available_devices,
            pytorch_version=torch.__version__,
            recommended_device=str(recommended)
        )
    
    def select_best_device(self) -> torch.device:
        """
        选择最佳的计算设备
        
        Returns:
            torch.device: 最佳设备
        """
        # 设备优先级：CUDA GPU > MPS > CPU
        
        # 1. 优先选择可用的CUDA GPU
        cuda_devices = [d for d in self.available_devices if d.device_type == 'cuda' and d.is_available]
        if cuda_devices:
            # 选择内存最多且利用率最低的GPU
            best_gpu = None
            best_score = -1
            
            for gpu in cuda_devices:
                if gpu.memory_total > 0:
                    memory_score = gpu.memory_free / gpu.memory_total
                    utilization_score = (100 - gpu.utilization) / 100
                    score = 0.7 * memory_score + 0.3 * utilization_score
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu
            
            if best_gpu:
                device = torch.device(f'cuda:{best_gpu.device_id}')
                self.current_device = device
                self.logger.info(f"选择CUDA设备: {best_gpu.name} (GPU {best_gpu.device_id})")
                return device
        
        # 2. 如果没有CUDA，尝试MPS（Apple Silicon）
        mps_devices = [d for d in self.available_devices if d.device_type == 'mps' and d.is_available]
        if mps_devices:
            device = torch.device('mps')
            self.current_device = device
            self.logger.info("选择MPS设备 (Apple Silicon)")
            return device
        
        # 3. 默认使用CPU
        device = torch.device('cpu')
        self.current_device = device
        self.logger.info("选择CPU设备")
        return device
    
    def get_device_by_type(self, device_type: str, device_id: Optional[int] = None) -> Optional[torch.device]:
        """
        根据类型获取设备
        
        Args:
            device_type: 设备类型 ('cpu', 'cuda', 'mps')
            device_id: 设备ID（对于CUDA）
            
        Returns:
            torch.device: 设备对象，如果不可用则返回None
        """
        if device_type == 'cpu':
            return torch.device('cpu')
        
        elif device_type == 'cuda':
            if torch.cuda.is_available():
                if device_id is not None:
                    if device_id < torch.cuda.device_count():
                        return torch.device(f'cuda:{device_id}')
                else:
                    return torch.device('cuda')
        
        elif device_type == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
        
        return None
    
    def optimize_for_device(self, device: torch.device):
        """
        为特定设备优化设置
        
        Args:
            device: 目标设备
        """
        try:
            if device.type == 'cuda':
                # CUDA优化
                torch.cuda.empty_cache()
                self.logger.info(f"已为CUDA设备 {device} 优化设置")
                
            elif device.type == 'mps':
                # MPS优化
                self.logger.info("已为MPS设备优化设置")
                
            else:
                # CPU优化
                # 设置线程数
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(psutil.cpu_count())
                self.logger.info("已为CPU设备优化设置")
                
        except Exception as e:
            self.logger.warning(f"设备优化失败: {e}")
    
    def clear_cache(self):
        """清理设备缓存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA缓存已清理")
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS没有显式的缓存清理方法
                pass
                
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
    
    def get_memory_usage(self, device: torch.device) -> Dict[str, float]:
        """
        获取设备内存使用情况
        
        Args:
            device: 设备
            
        Returns:
            Dict: 内存使用信息
        """
        if device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(device.index) / (1024**3)
                reserved = torch.cuda.memory_reserved(device.index) / (1024**3)
                total = torch.cuda.get_device_properties(device.index).total_memory / (1024**3)
                
                return {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': total,
                    'free_gb': total - reserved,
                    'utilization_percent': (reserved / total) * 100 if total > 0 else 0
                }
            except Exception as e:
                self.logger.error(f"获取CUDA内存使用情况失败: {e}")
                return {}
        
        elif device.type == 'cpu':
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'utilization_percent': memory.percent
            }
        
        else:
            # MPS或其他设备
            return {'device_type': device.type}
    
    def monitor_device_usage(self, duration: int = 60, interval: int = 5) -> List[Dict[str, Any]]:
        """
        监控设备使用情况
        
        Args:
            duration: 监控持续时间（秒）
            interval: 监控间隔（秒）
            
        Returns:
            List[Dict]: 监控数据列表
        """
        monitoring_data = []
        start_time = time.time()
        
        self.logger.info(f"开始监控设备使用情况，持续{duration}秒，间隔{interval}秒")
        
        while time.time() - start_time < duration:
            try:
                timestamp = time.time()
                
                # 更新设备信息
                current_devices = self._detect_devices()
                
                data_point = {
                    'timestamp': timestamp,
                    'devices': [
                        {
                            'type': device.device_type,
                            'id': device.device_id,
                            'name': device.name,
                            'memory_used': device.memory_used,
                            'memory_total': device.memory_total,
                            'memory_utilization': (device.memory_used / device.memory_total) * 100 if device.memory_total > 0 else 0,
                            'utilization': device.utilization,
                            'temperature': device.temperature,
                            'power_draw': device.power_draw
                        }
                        for device in current_devices
                    ]
                }
                
                monitoring_data.append(data_point)
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("监控被用户中断")
                break
            except Exception as e:
                self.logger.error(f"监控过程中出错: {e}")
                break
        
        self.logger.info(f"监控完成，收集了{len(monitoring_data)}个数据点")
        return monitoring_data
    
    def print_system_summary(self):
        """打印系统摘要信息"""
        system_info = self.get_system_info()
        
        print("\n" + "="*60)
        print("系统信息摘要")
        print("="*60)
        
        # 系统信息
        print(f"操作系统: {system_info.platform.title()}")
        print(f"CPU: {system_info.cpu_count} 核心, 使用率: {system_info.cpu_usage:.1f}%")
        print(f"内存: {system_info.memory_used:.1f}GB / {system_info.memory_total:.1f}GB "
              f"({(system_info.memory_used/system_info.memory_total)*100:.1f}%)")
        
        # PyTorch信息
        print(f"PyTorch版本: {system_info.pytorch_version}")
        
        # 设备信息
        print(f"可用设备数量: {len(system_info.devices)}")
        print(f"推荐设备: {system_info.recommended_device}")
        
        print("\n设备详细信息:")
        for device in system_info.devices:
            print(f"  {device.device_type.upper()}: {device.name}")
            if device.device_id is not None:
                print(f"    设备ID: {device.device_id}")
            if device.memory_total > 0:
                print(f"    内存: {device.memory_used:.1f}GB / {device.memory_total:.1f}GB "
                      f"({(device.memory_used/device.memory_total)*100:.1f}%)")
            if device.utilization > 0:
                print(f"    利用率: {device.utilization:.1f}%")
            if device.temperature > 0:
                print(f"    温度: {device.temperature:.1f}°C")
            if device.power_draw > 0:
                print(f"    功耗: {device.power_draw:.1f}W")
            print(f"    状态: {'可用' if device.is_available else '不可用'}")
            print()
        
        print("="*60)


# 全局设备管理器实例
device_manager = DeviceManager()


def get_optimal_device() -> torch.device:
    """
    获取最优计算设备的便捷函数
    
    Returns:
        torch.device: 最优设备
    """
    return device_manager.select_best_device()


def optimize_for_training(device: torch.device):
    """
    为训练优化设备设置的便捷函数
    
    Args:
        device: 目标设备
    """
    device_manager.optimize_for_device(device)


def clear_cache():
    """清理设备缓存的便捷函数"""
    device_manager.clear_cache()


def print_device_info():
    """打印设备信息的便捷函数"""
    device_manager.print_system_summary()


def get_device_by_preference(preferences: List[str] = None) -> torch.device:
    """
    根据偏好列表获取设备
    
    Args:
        preferences: 设备偏好列表，如 ['cuda', 'mps', 'cpu']
        
    Returns:
        torch.device: 选择的设备
    """
    if preferences is None:
        preferences = ['cuda', 'mps', 'cpu']
    
    for pref in preferences:
        device = device_manager.get_device_by_type(pref)
        if device is not None:
            return device
    
    # 默认返回CPU
    return torch.device('cpu')