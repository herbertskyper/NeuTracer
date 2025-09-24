#!/usr/bin/env python3
"""
AI模型日志分析功能测试脚本
"""

import os
import sys

sys.path.append("src")

from model_log_analyzer import ModelLogAnalyzer


def test_basic_analysis():
    """测试基础分析功能"""
    print("🧪 测试基础日志分析...")

    analyzer = ModelLogAnalyzer()
    result = analyzer.analyze_log_file("example_training_log.txt")

    assert "error" not in result, f"分析失败: {result.get('error')}"
    assert "summary" in result, "缺少摘要"
    assert "anomalies" in result, "缺少异常检测"
    assert "metrics" in result, "缺少指标提取"

    anomalies = result["anomalies"]
    summary = result["summary"]

    print(f"✅ 检测到 {len(anomalies)} 个异常")
    print(f"✅ 异常统计: {summary.get('anomalies', {})}")

    return True


def test_llm_prompt_generation():
    """测试LLM系统提示生成"""
    print("\n🧪 测试LLM系统提示生成...")

    analyzer = ModelLogAnalyzer()
    result = analyzer.analyze_with_llm_assistance(
        "example_training_log.txt", "GPU内存使用是否正常？"
    )

    assert "error" not in result, f"分析失败: {result.get('error')}"
    assert "llm_analysis" in result, "缺少LLM分析数据"

    llm_data = result["llm_analysis"]
    assert "system_prompt" in llm_data, "缺少系统提示"
    assert "ready_for_llm" in llm_data, "LLM准备状态缺失"
    assert llm_data["ready_for_llm"], "LLM未准备就绪"

    system_prompt = llm_data["system_prompt"]
    print(f"✅ 生成系统提示长度: {len(system_prompt)} 字符")
    print(f"✅ LLM准备状态: {llm_data['ready_for_llm']}")

    return True


def test_anomaly_detection():
    """测试异常检测功能"""
    print("\n🧪 测试异常检测功能...")

    analyzer = ModelLogAnalyzer()
    result = analyzer.analyze_log_file("example_inference_log.txt")

    assert "error" not in result, f"分析失败: {result.get('error')}"

    anomalies = result["anomalies"]
    by_severity = result["summary"]["anomalies"]["by_severity"]
    by_category = result["summary"]["anomalies"]["by_category"]

    print(f"✅ 检测到 {len(anomalies)} 个异常")
    print(f"✅ 按严重程度: {by_severity}")
    print(f"✅ 按类别: {by_category}")

    # 验证异常数据结构
    if anomalies:
        first_anomaly = anomalies[0]
        required_fields = [
            "category",
            "severity",
            "description",
            "line_number",
            "suggestion",
            "context",
        ]
        for field in required_fields:
            assert field in first_anomaly, f"异常数据缺少字段: {field}"

    return True


def test_metrics_extraction():
    """测试指标提取功能"""
    print("\n🧪 测试指标提取功能...")

    analyzer = ModelLogAnalyzer()
    result = analyzer.analyze_log_file("example_training_log.txt")

    assert "error" not in result, f"分析失败: {result.get('error')}"

    metrics = result["metrics"]
    summary = result["summary"]

    # 验证指标提取
    expected_metrics = ["loss", "accuracy", "learning_rate", "gpu_memory"]
    extracted_metrics = []

    for metric_name in expected_metrics:
        if metric_name in metrics and metrics[metric_name]:
            extracted_metrics.append(metric_name)
            print(f"✅ 提取到 {metric_name}: {len(metrics[metric_name])} 个数据点")

    assert len(extracted_metrics) > 0, "未提取到任何预期指标"

    # 验证趋势分析
    for metric_name in extracted_metrics:
        if metric_name in summary:
            trend = summary[metric_name].get("trend", "unknown")
            print(f"✅ {metric_name} 趋势: {trend}")

    return True


def main():
    """主测试函数"""
    print("🚀 开始AI模型日志分析功能测试")
    print("=" * 50)

    # 检查测试文件
    test_files = ["example_training_log.txt", "example_inference_log.txt"]
    for file in test_files:
        if not os.path.exists(file):
            print(f"❌ 测试文件缺失: {file}")
            return False

    tests = [
        test_basic_analysis,
        test_anomaly_detection,
        test_metrics_extraction,
        test_llm_prompt_generation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 通过")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} 异常: {str(e)}")

    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("🎉 所有测试通过！AI模型日志分析功能正常工作")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
