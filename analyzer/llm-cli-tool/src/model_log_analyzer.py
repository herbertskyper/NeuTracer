"""
AI模型日志分析器
专门用于分析AI训练和推理日志，识别异常指标和问题
"""

import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LogMetric:
    """日志指标数据结构"""

    name: str
    value: float
    timestamp: str
    line_number: int
    context: str


@dataclass
class LogAnomaly:
    """日志异常数据结构"""

    category: str  # memory, gpu, training, inference, model, config
    severity: str  # critical, high, medium, low
    description: str
    line_number: int
    metric_value: Optional[float]
    threshold: Optional[float]
    suggestion: str
    context: str


class ModelLogAnalyzer:
    """AI模型日志分析器"""

    def __init__(self):
        self.metrics_patterns = self._init_metrics_patterns()
        self.anomaly_patterns = self._init_anomaly_patterns()
        self.thresholds = self._init_thresholds()

    def _init_metrics_patterns(self) -> Dict[str, List[str]]:
        """初始化指标提取模式"""
        return {
            "loss": [
                r"loss[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                r"train_loss[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                r"validation_loss[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            ],
            "accuracy": [
                r"accuracy[:=]\s*([0-9]*\.?[0-9]+)",
                r"acc[:=]\s*([0-9]*\.?[0-9]+)",
                r"train_acc[:=]\s*([0-9]*\.?[0-9]+)",
                r"val_acc[:=]\s*([0-9]*\.?[0-9]+)",
            ],
            "learning_rate": [
                r"learning_rate[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                r"lr[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            ],
            "gpu_memory": [
                r"GPU.*memory.*([0-9]+\.?[0-9]*)\s*GB",
                r"memory.*usage.*([0-9]+\.?[0-9]*)\s*GB",
                r"allocated.*([0-9]+\.?[0-9]*)\s*[GM]B",
                r"memory usage=([0-9]+\.?[0-9]*)GB",
            ],
            "gpu_utilization": [
                r"GPU.*utilization.*([0-9]+)",
                r"GPU.*usage.*([0-9]+)",
                r"utilization=([0-9]+)%",
            ],
            "throughput": [
                r"throughput[:=]\s*([0-9]*\.?[0-9]+).*tokens?/s",
                r"tokens/second[:=]\s*([0-9]*\.?[0-9]+)",
                r"samples/second[:=]\s*([0-9]*\.?[0-9]+)",
            ],
            "latency": [
                r"latency[:=]\s*([0-9]*\.?[0-9]+).*ms",
                r"inference.*time[:=]\s*([0-9]*\.?[0-9]+).*seconds?",
                r"processing.*time[:=]\s*([0-9]*\.?[0-9]+)",
            ],
            "gradient_norm": [
                r"gradient.*norm[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                r"grad_norm[:=]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            ],
            "epoch": [r"epoch[:=]\s*([0-9]+)", r"Epoch\s+([0-9]+)"],
            "step": [r"step[:=]\s*([0-9]+)", r"Step\s+([0-9]+)"],
        }

    def _init_anomaly_patterns(self) -> Dict[str, List[str]]:
        """初始化异常检测模式"""
        return {
            "memory_issues": [
                r"out of memory|OOM|OutOfMemoryError",
                r"CUDA out of memory",
                r"memory allocation failed",
                r"insufficient memory",
                r"memory leak",
            ],
            "gpu_errors": [
                r"CUDA error|CUDA runtime error",
                r"CUDNN_STATUS_EXECUTION_FAILED",
                r"GPU device not found",
                r"NVIDIA-SMI has failed",
                r"device-side assert triggered",
            ],
            "training_instability": [
                r"loss.*nan|loss.*inf",
                r"gradient.*nan|gradient.*inf",
                r"gradient.*explosion",
                r"learning rate.*too.*high",
                r"training.*unstable",
                r"diverged|diverging",
            ],
            "convergence_issues": [
                r"no improvement for \d+ epochs",
                r"early stopping",
                r"plateau.*detected",
                r"learning.*stagnated",
            ],
            "data_issues": [
                r"empty.*batch|batch.*size.*0",
                r"data.*corruption|corrupt.*data",
                r"missing.*files|file.*not.*found",
                r"invalid.*data|data.*invalid",
            ],
            "model_issues": [
                r"failed.*load.*model|model.*loading.*failed",
                r"checkpoint.*not.*found|missing.*checkpoint",
                r"model.*format.*error|invalid.*model",
                r"version.*mismatch|incompatible.*version",
            ],
            "performance_issues": [
                r"timeout|time.*out",
                r"slow.*training|training.*slow",
                r"low.*throughput|throughput.*low",
                r"high.*latency|latency.*high",
            ],
        }

    def _init_thresholds(self) -> Dict[str, Dict[str, float]]:
        """初始化异常阈值"""
        return {
            "loss": {"min": 0.0, "max": 100.0, "nan_check": True},
            "accuracy": {"min": 0.0, "max": 1.0},
            "learning_rate": {"min": 1e-8, "max": 1.0},
            "gpu_memory": {"max": 95.0},  # 内存使用率超过95%
            "gpu_utilization": {"min": 10.0, "max": 100.0},  # 利用率过低或过高
            "throughput": {"min": 1.0},  # 吞吐量过低
            "latency": {"max": 10000.0},  # 延迟超过10秒
            "gradient_norm": {"max": 100.0, "nan_check": True},  # 梯度爆炸
        }

    def analyze_log_file(self, file_path: str) -> Dict[str, Any]:
        """分析日志文件"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            metrics = self._extract_metrics(lines)
            anomalies = self._detect_anomalies(lines, metrics)
            summary = self._generate_summary(metrics, anomalies, len(lines))

            return {
                "file_path": file_path,
                "total_lines": len(lines),
                "analysis_time": datetime.now().isoformat(),
                "summary": summary,
                "metrics": self._serialize_metrics(metrics),
                "anomalies": self._serialize_anomalies(anomalies),
                "recommendations": self._generate_recommendations(anomalies, metrics),
            }

        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}

    def _extract_metrics(self, lines: List[str]) -> Dict[str, List[LogMetric]]:
        """提取日志中的指标"""
        metrics: Dict[str, List[LogMetric]] = {
            key: [] for key in self.metrics_patterns.keys()
        }

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 提取时间戳
            timestamp = self._extract_timestamp(line)

            # 提取各种指标
            for metric_name, patterns in self.metrics_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        try:
                            value = float(match.group(1))
                            metric = LogMetric(
                                name=metric_name,
                                value=value,
                                timestamp=timestamp,
                                line_number=line_num,
                                context=line,
                            )
                            metrics[metric_name].append(metric)
                        except (ValueError, IndexError):
                            continue

        return metrics

    def _extract_timestamp(self, line: str) -> str:
        """提取时间戳"""
        # 假设时间戳格式为 ISO 8601
        match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?", line)
        if match:
            return match.group(0)

        return datetime.now().isoformat()  # 默认返回当前时间

    def _detect_anomalies(
        self, lines: List[str], metrics: Dict[str, List[LogMetric]]
    ) -> List[LogAnomaly]:
        """检测日志中的异常"""
        anomalies = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 检测已知的异常模式
            for anomaly_category, patterns in self.anomaly_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        anomaly = LogAnomaly(
                            category=anomaly_category,
                            severity=self._determine_severity(anomaly_category),
                            description=f"检测到{anomaly_category}异常",
                            line_number=line_num,
                            metric_value=None,
                            threshold=None,
                            suggestion=self._generate_suggestion(anomaly_category),
                            context=line,
                        )
                        anomalies.append(anomaly)

        # 基于指标阈值检测异常
        for metric_name, metric_values in metrics.items():
            if metric_name not in self.thresholds:
                continue

            for metric in metric_values:
                thresholds = self.thresholds[metric_name]

                # 检查 NaN 和 Inf
                if "nan_check" in thresholds and thresholds["nan_check"]:
                    if (
                        metric.value != metric.value
                        or metric.value == float("inf")
                        or metric.value == float("-inf")
                    ):
                        anomaly = LogAnomaly(
                            category=f"{metric_name}_issues",
                            severity="critical",
                            description=f"{metric_name} 值异常",
                            line_number=metric.line_number,
                            metric_value=metric.value,
                            threshold=None,
                            suggestion="检查数据源和预处理步骤",
                            context=metric.context,
                        )
                        anomalies.append(anomaly)
                        continue

                # 检查上下限
                if "min" in thresholds and metric.value < thresholds["min"]:
                    anomaly = LogAnomaly(
                        category=f"{metric_name}_issues",
                        severity="high",
                        description=f"{metric_name} 值低于阈值",
                        line_number=metric.line_number,
                        metric_value=metric.value,
                        threshold=thresholds["min"],
                        suggestion=f"增加{metric_name}，当前值: {metric.value}",
                        context=metric.context,
                    )
                    anomalies.append(anomaly)

                if "max" in thresholds and metric.value > thresholds["max"]:
                    anomaly = LogAnomaly(
                        category=f"{metric_name}_issues",
                        severity="high",
                        description=f"{metric_name} 值超过阈值",
                        line_number=metric.line_number,
                        metric_value=metric.value,
                        threshold=thresholds["max"],
                        suggestion=f"减少{metric_name}，当前值: {metric.value}",
                        context=metric.context,
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _determine_severity(self, anomaly_category: str) -> str:
        """确定异常的严重性"""
        if "critical" in anomaly_category:
            return "critical"
        elif "high" in anomaly_category:
            return "high"
        elif "medium" in anomaly_category:
            return "medium"
        else:
            return "low"

    def _generate_summary(
        self,
        metrics: Dict[str, List[LogMetric]],
        anomalies: List[LogAnomaly],
        total_lines: int,
    ) -> Dict[str, Any]:
        """生成分析摘要"""
        summary = {}

        # 指标摘要
        for metric_name, metric_values in metrics.items():
            if not metric_values:
                continue

            summary[metric_name] = {
                "count": len(metric_values),
                "min": min(m.value for m in metric_values),
                "max": max(m.value for m in metric_values),
                "avg": sum(m.value for m in metric_values) / len(metric_values),
                "last": metric_values[-1].value,
                "trend": self._detect_trend(metric_values),
            }

        # 异常摘要
        summary["anomalies"] = {
            "total": len(anomalies),
            "by_category": self._count_anomalies_by_category(anomalies),
            "by_severity": self._count_anomalies_by_severity(anomalies),
        }

        return summary

    def _detect_trend(self, metric_values: List[LogMetric]) -> str:
        """检测指标趋势"""
        if len(metric_values) < 2:
            return "stable"

        # 简单线性回归判断趋势
        x = list(range(len(metric_values)))
        y = [m.value for m in metric_values]

        # 计算斜率
        slope = (len(x) * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (
            len(x) * sum(xi**2 for xi in x) - sum(x) ** 2
        )

        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"

    def _count_anomalies_by_category(
        self, anomalies: List[LogAnomaly]
    ) -> Dict[str, int]:
        """按类别统计异常"""
        category_counts = {}

        for anomaly in anomalies:
            if anomaly.category not in category_counts:
                category_counts[anomaly.category] = 0
            category_counts[anomaly.category] += 1

        return category_counts

    def _count_anomalies_by_severity(
        self, anomalies: List[LogAnomaly]
    ) -> Dict[str, int]:
        """按严重性统计异常"""
        severity_counts = {}

        for anomaly in anomalies:
            if anomaly.severity not in severity_counts:
                severity_counts[anomaly.severity] = 0
            severity_counts[anomaly.severity] += 1

        return severity_counts

    def _serialize_metrics(
        self, metrics: Dict[str, List[LogMetric]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """序列化指标数据"""
        serialized = {
            key: [m.__dict__ for m in value] for key, value in metrics.items()
        }
        return serialized

    def _serialize_anomalies(self, anomalies: List[LogAnomaly]) -> List[Dict[str, Any]]:
        """序列化异常数据"""
        serialized = [anomaly.__dict__ for anomaly in anomalies]
        return serialized

    def _generate_recommendations(
        self, anomalies: List[LogAnomaly], metrics: Dict[str, List[LogMetric]]
    ) -> List[str]:
        """生成针对性的建议"""
        recommendations = []

        for anomaly in anomalies:
            if anomaly.category in ["memory_issues", "gpu_errors"]:
                recommendations.append("检查硬件资源，确保足够的内存和GPU可用")

            if anomaly.category in ["training_instability", "convergence_issues"]:
                recommendations.append("调整学习率或其他超参数，尝试恢复训练稳定性")

            if anomaly.category in ["data_issues"]:
                recommendations.append("检查数据集，确保数据完整性和正确性")

            if anomaly.category in ["model_issues"]:
                recommendations.append("检查模型文件和路径，确保模型可正确加载")

            if anomaly.category in ["performance_issues"]:
                recommendations.append("优化模型或数据管道，尝试提高性能")

        # 基于指标的建议
        for metric_name, metric_values in metrics.items():
            if metric_name not in self.thresholds:
                continue

            for metric in metric_values:
                thresholds = self.thresholds[metric_name]

                if "min" in thresholds and metric.value < thresholds["min"]:
                    recommendations.append(
                        f"增加{metric_name}，当前值过低: {metric.value}"
                    )

                if "max" in thresholds and metric.value > thresholds["max"]:
                    recommendations.append(
                        f"减少{metric_name}，当前值过高: {metric.value}"
                    )

        return list(set(recommendations))  # 去重返回建议列表

    def _generate_suggestion(self, anomaly_category: str) -> str:
        """为特定异常类别生成建议"""
        suggestions = {
            "memory_issues": "释放内存、减少批次大小或增加系统内存",
            "gpu_errors": "检查GPU驱动、CUDA版本或重启GPU设备",
            "training_instability": "降低学习率、检查梯度裁剪或调整优化器",
            "convergence_issues": "调整学习率调度、增加训练轮数或改进模型架构",
            "data_issues": "检查数据完整性、清理损坏数据或修复数据管道",
            "model_issues": "验证模型文件、检查版本兼容性或重新下载模型",
            "performance_issues": "优化代码、增加并行度或升级硬件配置",
        }
        return suggestions.get(anomaly_category, "检查相关配置和日志详情")

    def create_system_prompt_for_log_analysis(
        self, analysis_result: Dict[str, Any], user_question: Optional[str] = None
    ) -> str:
        """创建用于LLM分析的系统提示"""
        summary = analysis_result.get("summary", {})
        anomalies = analysis_result.get("anomalies", [])

        # 构建系统提示
        system_prompt = f"""你是一位专业的AI模型训练和推理专家，具有丰富的调试和优化经验。请基于以下AI模型日志分析结果，提供专业的诊断和建议。

## 📊 日志分析摘要

**文件信息:**
- 文件路径: {analysis_result.get('file_path', 'N/A')}
- 总行数: {analysis_result.get('total_lines', 0)}
- 分析时间: {analysis_result.get('analysis_time', 'N/A')}

**异常统计:**
- 总异常数: {summary.get('anomalies', {}).get('total', 0)}

"""

        # 添加严重程度统计
        if "by_severity" in summary.get("anomalies", {}):
            system_prompt += "**按严重程度分布:**\n"
            for severity, count in summary["anomalies"]["by_severity"].items():
                system_prompt += f"- {severity}: {count}个\n"
            system_prompt += "\n"

        # 添加类别统计
        if "by_category" in summary.get("anomalies", {}):
            system_prompt += "**按类别分布:**\n"
            for category, count in summary["anomalies"]["by_category"].items():
                system_prompt += f"- {category}: {count}个\n"
            system_prompt += "\n"

        # 添加关键指标
        system_prompt += "## 📈 关键指标分析\n\n"
        metrics_added = 0
        for metric_name, metric_data in summary.items():
            if (
                metric_name == "anomalies"
                or not isinstance(metric_data, dict)
                or "count" not in metric_data
            ):
                continue
            if metric_data.get("count", 0) > 0:
                system_prompt += f"**{metric_name}:**\n"
                system_prompt += f"- 数据点数量: {metric_data['count']}\n"
                system_prompt += (
                    f"- 数值范围: {metric_data['min']:.4f} - {metric_data['max']:.4f}\n"
                )
                system_prompt += f"- 平均值: {metric_data['avg']:.4f}\n"
                system_prompt += f"- 最新值: {metric_data['last']:.4f}\n"
                system_prompt += f"- 趋势: {metric_data.get('trend', 'unknown')}\n\n"
                metrics_added += 1
                if metrics_added >= 8:  # 限制指标数量
                    break

        # 添加详细异常信息
        if anomalies:
            system_prompt += "## ⚠️ 检测到的异常详情\n\n"

            # 按严重程度排序异常
            critical_anomalies = [
                a for a in anomalies if a.get("severity") == "critical"
            ]
            high_anomalies = [a for a in anomalies if a.get("severity") == "high"]
            medium_anomalies = [a for a in anomalies if a.get("severity") == "medium"]

            # 显示关键异常
            if critical_anomalies:
                system_prompt += "**🔴 严重异常:**\n"
                for i, anomaly in enumerate(critical_anomalies[:5], 1):  # 最多5个
                    system_prompt += f"{i}. {anomaly.get('description', 'N/A')} (行 {anomaly.get('line_number', 'N/A')})\n"
                    system_prompt += f"   类别: {anomaly.get('category', 'N/A')}\n"
                    if anomaly.get("metric_value") is not None:
                        system_prompt += f"   指标值: {anomaly.get('metric_value')}\n"
                    if anomaly.get("threshold") is not None:
                        system_prompt += f"   阈值: {anomaly.get('threshold')}\n"
                    system_prompt += (
                        f"   上下文: {anomaly.get('context', 'N/A')[:100]}...\n\n"
                    )

            # 显示高级异常
            if high_anomalies:
                system_prompt += "**🟠 高级异常:**\n"
                for i, anomaly in enumerate(high_anomalies[:3], 1):  # 最多3个
                    system_prompt += f"{i}. {anomaly.get('description', 'N/A')} (行 {anomaly.get('line_number', 'N/A')})\n"
                    system_prompt += f"   类别: {anomaly.get('category', 'N/A')}\n"
                    if anomaly.get("metric_value") is not None:
                        system_prompt += f"   指标值: {anomaly.get('metric_value')}\n"
                    system_prompt += (
                        f"   上下文: {anomaly.get('context', 'N/A')[:100]}...\n\n"
                    )

            # 显示中等异常（简化版）
            if medium_anomalies:
                system_prompt += f"**🟡 中等异常:** {len(medium_anomalies)}个\n"
                for anomaly in medium_anomalies[:2]:
                    system_prompt += f"- {anomaly.get('description', 'N/A')} (行 {anomaly.get('line_number', 'N/A')})\n"
                system_prompt += "\n"

        # 添加用户特定问题
        if user_question:
            system_prompt += (
                f"## 🎯 用户关注的具体问题\n\n**问题:** {user_question}\n\n"
            )

        # 添加分析指导
        system_prompt += """## 📋 请提供以下分析

请作为AI模型专家，基于上述日志分析结果，提供：

1. **🔍 问题诊断**: 
   - 识别最关键的问题和风险
   - 分析问题的根本原因
   - 评估问题的严重程度和影响

2. **🛠️ 解决方案**: 
   - 提供具体的修复建议
   - 推荐最佳实践
   - 给出优先级排序

3. **📊 性能优化建议**: 
   - 基于指标趋势的优化建议
   - 资源配置优化
   - 训练/推理效率提升

4. **🚨 预防措施**: 
   - 如何避免类似问题
   - 监控和预警建议
   - 配置优化建议

请用专业、清晰、易懂的语言回答，并提供可执行的具体步骤。如果日志中没有明显问题，请分析整体健康状况并提供优化建议。
"""

        return system_prompt

    def analyze_with_llm_assistance(
        self,
        file_path: str,
        user_question: Optional[str] = None,
        llm_model: str = "deepseek-chat",
    ) -> Dict[str, Any]:
        """使用LLM辅助进行智能日志分析"""
        # 首先进行基础分析
        basic_analysis = self.analyze_log_file(file_path)

        if "error" in basic_analysis:
            return basic_analysis

        # 创建系统提示
        system_prompt = self.create_system_prompt_for_log_analysis(
            basic_analysis, user_question
        )

        # 构建用户提示
        user_prompt = "请分析这个AI模型日志并提供专业的诊断和建议。"
        if user_question:
            user_prompt += f"\n\n特别关注这个问题: {user_question}"

        # 返回完整分析结果
        return {
            **basic_analysis,
            "llm_analysis": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": llm_model,
                "ready_for_llm": True,
            },
        }
