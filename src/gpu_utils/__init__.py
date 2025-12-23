"""
GPU工具模块
提供设备管理和优化功能
"""

from .gpu_manager import (
    get_optimal_device,
    optimize_for_training,
    clear_cache,
    print_device_info,
    device_manager
)

__all__ = [
    'get_optimal_device',
    'optimize_for_training', 
    'clear_cache',
    'print_device_info',
    'device_manager'
]