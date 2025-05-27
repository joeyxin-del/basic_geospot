import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from src.utils import get_logger, ConfigManager, Visualizer, Evaluator

logger = get_logger('experiment')

class ExperimentManager:
    """
    实验管理工具类。
    用于管理训练实验，记录实验参数和结果，支持结果比较和分析。
    """
    def __init__(
        self,
        save_dir: str = "experiments",
        config_manager: Optional[ConfigManager] = None,
        visualizer: Optional[Visualizer] = None,
        evaluator: Optional[Evaluator] = None
    ):
        """
        初始化实验管理工具。
        
        Args:
            save_dir: 实验数据保存目录
            config_manager: 配置管理工具实例
            visualizer: 可视化工具实例
            evaluator: 评估工具实例
        """
        self.save_dir = save_dir
        self.config_manager = config_manager or ConfigManager()
        self.visualizer = visualizer or Visualizer(save_dir=os.path.join(save_dir, "plots"))
        self.evaluator = evaluator or Evaluator(save_dir=os.path.join(save_dir, "evaluations"))
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        
        # 初始化实验记录
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.current_experiment: Optional[str] = None
        
    def create_experiment(
        self,
        name: Optional[str] = None,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建新的实验。
        
        Args:
            name: 实验名称，如果为None则自动生成
            description: 实验描述
            config: 实验配置参数
            
        Returns:
            实验ID
        """
        # 生成实验ID
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{timestamp}"
        else:
            experiment_id = name
            
        # 检查实验ID是否已存在
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
            
        # 创建实验记录
        experiment = {
            "id": experiment_id,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "metrics": {},
            "results": {},
            "checkpoints": [],
            "status": "created"
        }
        
        # 保存实验配置
        if config:
            config_path = os.path.join(self.save_dir, "configs", f"{experiment_id}.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        # 更新实验记录
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
        
    def load_experiment(self, experiment_id: str) -> None:
        """
        加载已存在的实验。
        
        Args:
            experiment_id: 实验ID
        """
        if experiment_id not in self.experiments:
            # 尝试从文件加载
            config_path = os.path.join(self.save_dir, "configs", f"{experiment_id}.json")
            results_path = os.path.join(self.save_dir, "results", f"{experiment_id}.json")
            
            if not os.path.exists(config_path):
                raise ValueError(f"Experiment {experiment_id} not found")
                
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 加载结果
            results = {}
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
            # 创建实验记录
            experiment = {
                "id": experiment_id,
                "description": results.get("description", ""),
                "created_at": results.get("created_at", datetime.now().isoformat()),
                "config": config,
                "metrics": results.get("metrics", {}),
                "results": results.get("results", {}),
                "checkpoints": results.get("checkpoints", []),
                "status": results.get("status", "loaded")
            }
            
            self.experiments[experiment_id] = experiment
            
        self.current_experiment = experiment_id
        logger.info(f"Loaded experiment: {experiment_id}")
        
    def update_metrics(
        self,
        metrics: Dict[str, float],
        experiment_id: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> None:
        """
        更新实验指标。
        
        Args:
            metrics: 指标字典
            experiment_id: 实验ID，如果为None则使用当前实验
            epoch: 训练轮次，如果为None则不记录轮次信息
        """
        experiment_id = experiment_id or self.current_experiment
        if experiment_id is None:
            raise ValueError("No experiment selected")
            
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # 更新指标
        if epoch is not None:
            if "epoch_metrics" not in self.experiments[experiment_id]["metrics"]:
                self.experiments[experiment_id]["metrics"]["epoch_metrics"] = {}
            self.experiments[experiment_id]["metrics"]["epoch_metrics"][str(epoch)] = metrics
        else:
            self.experiments[experiment_id]["metrics"].update(metrics)
            
        # 保存更新
        self._save_experiment(experiment_id)
        
    def add_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Optional[Dict[str, float]] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        添加模型检查点。
        
        Args:
            checkpoint_path: 检查点文件路径
            metrics: 检查点对应的指标
            experiment_id: 实验ID，如果为None则使用当前实验
        """
        experiment_id = experiment_id or self.current_experiment
        if experiment_id is None:
            raise ValueError("No experiment selected")
            
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # 添加检查点记录
        checkpoint = {
            "path": checkpoint_path,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        
        self.experiments[experiment_id]["checkpoints"].append(checkpoint)
        self._save_experiment(experiment_id)
        
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        plot: bool = True
    ) -> pd.DataFrame:
        """
        比较多个实验的结果。
        
        Args:
            experiment_ids: 要比较的实验ID列表
            metrics: 要比较的指标列表，如果为None则比较所有指标
            plot: 是否绘制比较图表
            
        Returns:
            比较结果DataFrame
        """
        # 验证实验ID
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                raise ValueError(f"Experiment {exp_id} not found")
                
        # 准备比较数据
        comparison_data = []
        for exp_id in experiment_ids:
            exp = self.experiments[exp_id]
            data = {
                "experiment_id": exp_id,
                "description": exp["description"],
                "created_at": exp["created_at"],
                "status": exp["status"]
            }
            
            # 添加指标数据
            if metrics:
                for metric in metrics:
                    if metric in exp["metrics"]:
                        data[metric] = exp["metrics"][metric]
            else:
                data.update(exp["metrics"])
                
            comparison_data.append(data)
            
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 绘制比较图表
        if plot and len(experiment_ids) > 1:
            self._plot_comparison(df, metrics)
            
        return df
        
    def _plot_comparison(self, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> None:
        """
        绘制实验比较图表。
        
        Args:
            df: 比较数据DataFrame
            metrics: 要绘制的指标列表
        """
        if metrics is None:
            # 选择数值型列作为指标
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = [m for m in metrics if m not in ["created_at"]]
            
        # 绘制每个指标的条形图
        for metric in metrics:
            if metric in df.columns:
                plt.figure(figsize=(10, 6))
                plt.bar(df["experiment_id"], df[metric])
                plt.title(f"Comparison of {metric}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                save_path = os.path.join(self.save_dir, "plots", f"comparison_{metric}.png")
                plt.savefig(save_path)
                plt.close()
                
    def _save_experiment(self, experiment_id: str) -> None:
        """
        保存实验数据。
        
        Args:
            experiment_id: 实验ID
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # 保存结果
        results_path = os.path.join(self.save_dir, "results", f"{experiment_id}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiments[experiment_id], f, indent=2, ensure_ascii=False)
            
    def get_experiment_summary(self, experiment_id: Optional[str] = None) -> str:
        """
        获取实验摘要。
        
        Args:
            experiment_id: 实验ID，如果为None则使用当前实验
            
        Returns:
            实验摘要字符串
        """
        experiment_id = experiment_id or self.current_experiment
        if experiment_id is None:
            raise ValueError("No experiment selected")
            
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp = self.experiments[experiment_id]
        
        summary = [f"Experiment Summary: {experiment_id}"]
        summary.append("-" * 40)
        summary.append(f"Description: {exp['description']}")
        summary.append(f"Created at: {exp['created_at']}")
        summary.append(f"Status: {exp['status']}")
        summary.append("\nMetrics:")
        
        # 添加指标信息
        for metric, value in exp["metrics"].items():
            if isinstance(value, dict):
                # 处理epoch指标
                if metric == "epoch_metrics":
                    latest_epoch = max(map(int, value.keys()))
                    summary.append(f"Latest epoch ({latest_epoch}):")
                    for m, v in value[str(latest_epoch)].items():
                        summary.append(f"  {m}: {v:.4f}")
            else:
                summary.append(f"{metric}: {value:.4f}")
                
        # 添加检查点信息
        if exp["checkpoints"]:
            summary.append("\nCheckpoints:")
            for cp in exp["checkpoints"]:
                summary.append(f"- {os.path.basename(cp['path'])} ({cp['created_at']})")
                
        return "\n".join(summary)


# 导出
__all__ = ['ExperimentManager'] 