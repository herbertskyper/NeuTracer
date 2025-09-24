#!/usr/bin/env python3
"""
AI模型日志分析功能演示脚本
展示如何使用LLM系统提示辅助诊断AI训练和推理日志
"""

import sys
import os

sys.path.append("src")

from model_log_analyzer import ModelLogAnalyzer


def demo_basic_analysis():
    """演示基础日志分析功能"""
    print("=" * 60)
    print("🔍 基础AI模型日志分析演示")
    print("=" * 60)

    analyzer = ModelLogAnalyzer()

    # 分析训练日志
    print("\n📚 分析训练日志...")
    training_result = analyzer.analyze_log_file("example_training_log.txt")

    if "error" not in training_result:
        summary = training_result.get("summary", {})
        anomalies = training_result.get("anomalies", [])

        print(f"✅ 检测到 {len(anomalies)} 个异常")
        print(
            f"📊 提取了 {sum(len(metrics) for metrics in training_result.get('metrics', {}).values())} 个指标"
        )

        # 显示关键指标
        for metric_name, metric_data in summary.items():
            if metric_name != "anomalies" and isinstance(metric_data, dict):
                if metric_data.get("count", 0) > 0:
                    print(
                        f"   • {metric_name}: {metric_data['count']}个数据点, 趋势: {metric_data.get('trend', 'unknown')}"
                    )


def demo_llm_system_prompt():
    """演示LLM系统提示生成功能"""
    print("\n" + "=" * 60)
    print("🤖 LLM系统提示生成演示")
    print("=" * 60)

    analyzer = ModelLogAnalyzer()

    # 使用LLM辅助分析
    result = analyzer.analyze_with_llm_assistance(
        "example_training_log.txt", "训练过程中GPU内存使用有什么问题？"
    )

    if "llm_analysis" in result:
        llm_data = result["llm_analysis"]
        system_prompt = llm_data.get("system_prompt", "")

        print("📝 生成的系统提示预览 (前800字符):")
        print("-" * 40)
        print(
            system_prompt[:800] + "..." if len(system_prompt) > 800 else system_prompt
        )

        print(f"\n📏 完整系统提示长度: {len(system_prompt)} 字符")
        print(f"🎯 目标模型: {llm_data.get('model', 'deepseek-chat')}")
        print(f"✅ 准备就绪: {llm_data.get('ready_for_llm', False)}")


def demo_anomaly_detection():
    """演示异常检测功能"""
    print("\n" + "=" * 60)
    print("⚠️ 异常检测功能演示")
    print("=" * 60)

    analyzer = ModelLogAnalyzer()

    # 分析推理日志
    inference_result = analyzer.analyze_log_file("example_inference_log.txt")

    if "error" not in inference_result:
        anomalies = inference_result.get("anomalies", [])

        print(f"🔍 在推理日志中检测到 {len(anomalies)} 个异常:")

        # 按严重程度分组显示
        severity_groups = {}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "unknown")
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(anomaly)

        severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

        for severity, anomaly_list in severity_groups.items():
            icon = severity_icons.get(severity, "⚪")
            print(f"\n{icon} {severity.upper()} 级异常 ({len(anomaly_list)}个):")
            for i, anomaly in enumerate(anomaly_list[:3], 1):  # 最多显示3个
                print(
                    f"   {i}. {anomaly.get('description', 'N/A')} (行 {anomaly.get('line_number', 'N/A')})"
                )
                print(f"      类别: {anomaly.get('category', 'N/A')}")
                if anomaly.get("metric_value") is not None:
                    print(f"      值: {anomaly.get('metric_value')}")


def demo_metrics_extraction():
    """演示指标提取功能"""
    print("\n" + "=" * 60)
    print("📊 指标提取功能演示")
    print("=" * 60)

    analyzer = ModelLogAnalyzer()

    # 分析训练日志的指标
    result = analyzer.analyze_log_file("example_training_log.txt")

    if "error" not in result:
        metrics = result.get("metrics", {})

        print("📈 提取的关键指标类型:")

        for metric_name, metric_list in metrics.items():
            if metric_list:  # 只显示有数据的指标
                first_metric = metric_list[0]
                last_metric = metric_list[-1]

                print(f"\n• {metric_name}:")
                print(f"  数据点数量: {len(metric_list)}")
                print(
                    f"  首次记录: {first_metric.get('value')} (行 {first_metric.get('line_number')})"
                )
                print(
                    f"  最新记录: {last_metric.get('value')} (行 {last_metric.get('line_number')})"
                )

                # 显示值域
                values = [m.get("value", 0) for m in metric_list]
                print(f"  值域: {min(values):.4f} - {max(values):.4f}")


def main():
    """主演示函数"""
    print("🚀 AI模型日志分析工具演示")
    print("支持使用LLM系统提示进行智能诊断")

    # 检查示例文件是否存在
    required_files = ["example_training_log.txt", "example_inference_log.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"\n❌ 缺少示例文件: {', '.join(missing_files)}")
        print("请确保示例日志文件存在于当前目录")
        return

    try:
        # 运行各种演示
        demo_basic_analysis()
        demo_anomaly_detection()
        demo_metrics_extraction()
        demo_llm_system_prompt()

        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)
        print("\n💡 使用提示:")
        print("1. 基础分析: python -m src.cli --analyze-model-log your_log.txt")
        print(
            "2. LLM辅助分析: python -m src.cli --analyze-model-log your_log.txt --use-llm-analysis"
        )
        print(
            "3. 特定问题分析: python -m src.cli --analyze-model-log your_log.txt --model-log-question '你的问题'"
        )
        print(
            "4. 完整LLM分析: python -m src.cli --analyze-model-log your_log.txt --use-llm-analysis --model-log-question '详细问题'"
        )

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
