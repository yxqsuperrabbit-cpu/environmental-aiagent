"""
AirGuardian系统集成模块
实现完整的数据流管道，连接模型训练、预测、LLM分析和界面模块

根据任务7.1要求：
- 连接模型训练和预测模块
- 集成LLM分析和界面模块  
- 实现完整的数据流管道
- 满足需求1.1-4.5的所有功能
"""
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import logging

# 导入系统模块
from src.model_trainer.model_trainer import ModelTrainer, TrainedModel
from src.prediction_engine.prediction_engine import PredictionEngine, PredictionResult
from src.llm_analyzer.llm_analyzer import LLMAnalyzer, AnalysisReport
from src.web_interface.document_generator import DocumentGenerator
from src.config import Config


@dataclass
class SystemStatus:
    """系统状态数据类"""
    model_trainer_ready: bool = False
    prediction_engine_ready: bool = False
    llm_analyzer_ready: bool = False
    document_generator_ready: bool = False
    current_model: Optional[str] = None
    last_prediction: Optional[datetime] = None
    last_analysis: Optional[datetime] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
    
    @property
    def is_fully_ready(self) -> bool:
        """检查系统是否完全就绪"""
        return (self.model_trainer_ready and 
                self.prediction_engine_ready and 
                self.llm_analyzer_ready and 
                self.document_generator_ready)


@dataclass
class PipelineResult:
    """完整管道执行结果"""
    success: bool
    prediction_result: Optional[PredictionResult] = None
    analysis_report: Optional[AnalysisReport] = None
    citizen_document: Optional[str] = None
    government_document: Optional[str] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AirGuardianSystem:
    """
    AirGuardian系统集成类
    
    实现完整的空气质量预测与分析系统，包括：
    - 模型训练管道
    - 预测执行管道
    - 智能分析管道
    - 文档生成管道
    - 端到端工作流程
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        初始化AirGuardian系统
        
        Args:
            model_dir: 模型存储目录
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个模块
        self.model_trainer = ModelTrainer(model_dir)
        self.prediction_engine = PredictionEngine(model_dir)
        self.llm_analyzer = LLMAnalyzer()
        self.document_generator = DocumentGenerator()
        
        # 系统状态
        self.status = SystemStatus()
        
        # 初始化系统
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """初始化系统各模块"""
        try:
            # 检查模型训练器
            self.status.model_trainer_ready = True
            self.logger.info("模型训练器初始化成功")
            
            # 检查预测引擎
            available_models = self.prediction_engine.list_available_models()
            if available_models:
                # 尝试加载最新模型
                latest_model = available_models[0]
                if self.prediction_engine.load_model(latest_model):
                    self.status.prediction_engine_ready = True
                    self.status.current_model = latest_model
                    self.logger.info(f"预测引擎初始化成功，加载模型: {latest_model}")
                else:
                    self.status.error_messages.append("预测引擎模型加载失败")
                    self.logger.warning("预测引擎模型加载失败")
            else:
                self.status.error_messages.append("未找到可用的预测模型")
                self.logger.warning("未找到可用的预测模型")
            
            # 检查LLM分析器
            if self.llm_analyzer.test_connection():
                self.status.llm_analyzer_ready = True
                self.logger.info("LLM分析器连接成功")
            else:
                self.status.error_messages.append("LLM服务连接失败")
                self.logger.warning("LLM服务连接失败")
            
            # 检查文档生成器
            self.status.document_generator_ready = True
            self.logger.info("文档生成器初始化成功")
            
        except Exception as e:
            self.status.error_messages.append(f"系统初始化失败: {str(e)}")
            self.logger.error(f"系统初始化失败: {str(e)}")
    
    def get_system_status(self) -> SystemStatus:
        """
        获取系统状态
        
        Returns:
            SystemStatus: 当前系统状态
        """
        return self.status
    
    def generate_data_requirements(self) -> str:
        """
        生成数据要求文档
        
        根据需求1.1：生成包含数据格式、特征要求、质量标准的markdown文档
        
        Returns:
            str: 数据要求文档内容
        """
        try:
            return self.model_trainer.generate_data_requirements()
        except Exception as e:
            self.logger.error(f"生成数据要求文档失败: {str(e)}")
            raise
    
    def train_model_pipeline(self, data_path: str, model_name: Optional[str] = None) -> TrainedModel:
        """
        完整的模型训练管道
        
        根据需求1.2-1.5：训练LSTM模型并保存
        
        Args:
            data_path: 训练数据路径
            model_name: 模型名称（可选，默认使用时间戳）
            
        Returns:
            TrainedModel: 训练好的模型
            
        Raises:
            Exception: 当训练失败时
        """
        try:
            self.logger.info(f"开始模型训练管道: {data_path}")
            
            # 1. 验证数据格式
            validation_result = self.model_trainer.validate_training_data(data_path)
            if not all(validation_result.values()):
                raise ValueError(f"训练数据验证失败: {validation_result}")
            
            # 2. 训练模型
            trained_model = self.model_trainer.train_model(data_path)
            self.logger.info("模型训练完成")
            
            # 3. 验证模型性能
            test_data = pd.read_csv(data_path)
            test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
            mae = self.model_trainer.validate_model(trained_model, test_data)
            
            # 检查是否满足精度要求（需求1.3）
            if mae > Config.MAX_MAE_THRESHOLD:
                self.logger.warning(f"模型精度不达标: MAE={mae:.2f} > {Config.MAX_MAE_THRESHOLD}")
            else:
                self.logger.info(f"模型精度达标: MAE={mae:.2f}")
            
            # 4. 保存模型
            if model_name is None:
                model_name = f"air_quality_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            success = self.model_trainer.save_model(trained_model, model_name)
            if not success:
                raise Exception("模型保存失败")
            
            # 5. 更新系统状态
            self.status.current_model = model_name
            if not self.status.prediction_engine_ready:
                # 尝试加载新训练的模型到预测引擎
                if self.prediction_engine.load_model(model_name):
                    self.status.prediction_engine_ready = True
            
            self.logger.info(f"模型训练管道完成: {model_name}")
            return trained_model
            
        except Exception as e:
            self.logger.error(f"模型训练管道失败: {str(e)}")
            raise
    
    def prediction_pipeline(self, historical_data: pd.DataFrame) -> PredictionResult:
        """
        完整的预测管道
        
        根据需求2.1-2.4：执行空气质量预测并格式化结果
        
        Args:
            historical_data: 历史数据（前24小时）
            
        Returns:
            PredictionResult: 预测结果
            
        Raises:
            Exception: 当预测失败时
        """
        try:
            if not self.status.prediction_engine_ready:
                raise Exception("预测引擎未就绪，请先加载模型")
            
            self.logger.info("开始预测管道")
            
            # 执行预测
            prediction_result = self.prediction_engine.predict(historical_data)
            
            # 更新状态
            self.status.last_prediction = datetime.now()
            
            self.logger.info(f"预测管道完成，生成了{len(prediction_result.pm25_predictions)}小时的预测")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"预测管道失败: {str(e)}")
            raise
    
    def analysis_pipeline(self, prediction_result: PredictionResult) -> AnalysisReport:
        """
        完整的智能分析管道
        
        根据需求3.1-3.6：分析预测结果并生成智能报告
        
        Args:
            prediction_result: 预测结果
            
        Returns:
            AnalysisReport: 分析报告
            
        Raises:
            Exception: 当分析失败时
        """
        try:
            if not self.status.llm_analyzer_ready:
                raise Exception("LLM分析器未就绪，请检查Ollama服务")
            
            self.logger.info("开始智能分析管道")
            
            # 格式化预测结果供LLM使用
            formatted_data = self.prediction_engine.format_for_llm(prediction_result)
            
            # 执行智能分析
            analysis_report = self.llm_analyzer.analyze_prediction(formatted_data)
            
            # 更新状态
            self.status.last_analysis = datetime.now()
            
            self.logger.info("智能分析管道完成")
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"智能分析管道失败: {str(e)}")
            raise
    
    def document_generation_pipeline(self, analysis_report: AnalysisReport, 
                                   prediction_result: PredictionResult) -> Tuple[str, str]:
        """
        完整的文档生成管道
        
        根据需求4.3-4.5：生成市民版和政府版文档
        
        Args:
            analysis_report: 分析报告
            prediction_result: 预测结果
            
        Returns:
            Tuple[str, str]: (市民版文档, 政府版文档)
            
        Raises:
            Exception: 当文档生成失败时
        """
        try:
            if not self.status.document_generator_ready:
                raise Exception("文档生成器未就绪")
            
            self.logger.info("开始文档生成管道")
            
            # 生成市民版文档
            citizen_document = self.document_generator.generate_citizen_document(
                analysis_report, prediction_result
            )
            
            # 生成政府版文档
            government_document = self.document_generator.generate_government_document(
                analysis_report, prediction_result
            )
            
            self.logger.info("文档生成管道完成")
            return citizen_document, government_document
            
        except Exception as e:
            self.logger.error(f"文档生成管道失败: {str(e)}")
            raise
    
    def full_pipeline(self, historical_data: pd.DataFrame, 
                     generate_documents: bool = True) -> PipelineResult:
        """
        完整的端到端管道
        
        从历史数据到最终文档的完整工作流程
        
        Args:
            historical_data: 历史数据（前24小时）
            generate_documents: 是否生成文档
            
        Returns:
            PipelineResult: 完整的管道执行结果
        """
        start_time = datetime.now()
        result = PipelineResult(success=False)
        
        try:
            self.logger.info("开始完整管道执行")
            
            # 检查系统状态
            if not self.status.is_fully_ready:
                missing_components = []
                if not self.status.prediction_engine_ready:
                    missing_components.append("预测引擎")
                if not self.status.llm_analyzer_ready:
                    missing_components.append("LLM分析器")
                if not self.status.document_generator_ready:
                    missing_components.append("文档生成器")
                
                raise Exception(f"系统未完全就绪，缺少组件: {', '.join(missing_components)}")
            
            # 1. 执行预测管道
            self.logger.info("步骤1: 执行预测")
            prediction_result = self.prediction_pipeline(historical_data)
            result.prediction_result = prediction_result
            
            # 2. 执行智能分析管道
            self.logger.info("步骤2: 执行智能分析")
            analysis_report = self.analysis_pipeline(prediction_result)
            result.analysis_report = analysis_report
            
            # 3. 生成文档（可选）
            if generate_documents:
                self.logger.info("步骤3: 生成文档")
                citizen_doc, government_doc = self.document_generation_pipeline(
                    analysis_report, prediction_result
                )
                result.citizen_document = citizen_doc
                result.government_document = government_doc
            
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # 设置元数据
            result.metadata = {
                'pipeline_version': '1.0',
                'execution_timestamp': start_time.isoformat(),
                'model_used': self.status.current_model,
                'input_data_points': len(historical_data),
                'prediction_hours': len(prediction_result.pm25_predictions),
                'documents_generated': generate_documents,
                'system_status': {
                    'model_trainer_ready': self.status.model_trainer_ready,
                    'prediction_engine_ready': self.status.prediction_engine_ready,
                    'llm_analyzer_ready': self.status.llm_analyzer_ready,
                    'document_generator_ready': self.status.document_generator_ready
                }
            }
            
            result.success = True
            self.logger.info(f"完整管道执行成功，耗时: {execution_time:.2f}秒")
            
        except Exception as e:
            result.error_message = str(e)
            result.execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"完整管道执行失败: {str(e)}")
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        系统健康检查
        
        Returns:
            Dict: 健康检查结果
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if self.status.is_fully_ready else 'degraded',
            'components': {
                'model_trainer': {
                    'status': 'healthy' if self.status.model_trainer_ready else 'unhealthy',
                    'details': 'Model trainer initialized successfully' if self.status.model_trainer_ready else 'Model trainer not ready'
                },
                'prediction_engine': {
                    'status': 'healthy' if self.status.prediction_engine_ready else 'unhealthy',
                    'details': f'Model loaded: {self.status.current_model}' if self.status.prediction_engine_ready else 'No model loaded',
                    'available_models': self.prediction_engine.list_available_models()
                },
                'llm_analyzer': {
                    'status': 'healthy' if self.status.llm_analyzer_ready else 'unhealthy',
                    'details': 'LLM service connected' if self.status.llm_analyzer_ready else 'LLM service not available',
                    'model_name': self.llm_analyzer.model_name,
                    'base_url': self.llm_analyzer.base_url
                },
                'document_generator': {
                    'status': 'healthy' if self.status.document_generator_ready else 'unhealthy',
                    'details': 'Document generator ready' if self.status.document_generator_ready else 'Document generator not ready'
                }
            },
            'errors': self.status.error_messages,
            'last_prediction': self.status.last_prediction.isoformat() if self.status.last_prediction else None,
            'last_analysis': self.status.last_analysis.isoformat() if self.status.last_analysis else None
        }
        
        return health_status
    
    def reload_model(self, model_name: str) -> bool:
        """
        重新加载指定模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 加载是否成功
        """
        try:
            success = self.prediction_engine.load_model(model_name)
            if success:
                self.status.prediction_engine_ready = True
                self.status.current_model = model_name
                self.logger.info(f"模型重新加载成功: {model_name}")
            else:
                self.logger.error(f"模型重新加载失败: {model_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"模型重新加载异常: {str(e)}")
            return False
    
    def reset_system(self) -> None:
        """重置系统状态"""
        self.status = SystemStatus()
        self._initialize_system()
        self.logger.info("系统已重置")
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.prediction_engine.list_available_models()
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        return self.prediction_engine.get_model_info()