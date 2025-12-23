#!/usr/bin/env python3
"""
AirGuardian应用启动脚本
运行Streamlit Web界面，集成完整的系统功能
"""
import sys
import os
from pathlib import Path
import logging

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def check_system_requirements():
    """检查系统要求"""
    try:
        # 检查必要的模块
        from src.air_guardian_system import AirGuardianSystem
        from src.web_interface.streamlit_app import main
        
        # 初始化系统进行健康检查
        system = AirGuardianSystem()
        health_status = system.health_check()
        
        print("AirGuardian 系统启动检查")
        print("=" * 50)
        print(f"整体状态: {health_status['overall_status'].upper()}")
        
        for component, details in health_status['components'].items():
            status_icon = "[OK]" if details['status'] == 'healthy' else "[ERROR]"
            component_name = component.replace('_', ' ').title()
            print(f"{status_icon} {component_name}: {details['details']}")
        
        if health_status['errors']:
            print("\n系统警告:")
            for error in health_status['errors']:
                print(f"   • {error}")
        
        print("\n启动Streamlit Web界面...")
        return True
        
    except ImportError as e:
        print(f"[ERROR] 模块导入失败: {str(e)}")
        print("请确保所有依赖包已正确安装")
        return False
    except Exception as e:
        print(f"[ERROR] 系统检查失败: {str(e)}")
        return False

if __name__ == "__main__":
    if check_system_requirements():
        # 使用streamlit run命令启动应用
        import subprocess
        import sys
        
        print("正在启动Streamlit应用...")
        print("应用将在浏览器中打开: http://localhost:8504")
        print("按 Ctrl+C 停止应用")
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "src/web_interface/streamlit_app.py",
                "--server.port", "8504",
                "--server.address", "localhost"
            ])
        except KeyboardInterrupt:
            print("\n应用已停止")
        except Exception as e:
            print(f"启动Streamlit应用失败: {str(e)}")
            print("请手动运行: streamlit run src/web_interface/streamlit_app.py --server.port 8503")
    else:
        print("[ERROR] 系统检查失败，无法启动应用")
        print("请检查以下组件:")
        print("1. 确保所有Python依赖已安装")
        print("2. 确保Ollama服务正在运行")
        print("3. 确保deepseek-r1模型已下载")
        sys.exit(1)