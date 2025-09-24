import argparse
import sys
import os
from .api import send_request
from .config import DEFAULT_MODEL, set_api_key
from .settings import save_api_key, get_saved_api_key
from .reasoning_logger import ReasoningLogger, prepare_file_analysis_prompt
from .model_log_analyzer import ModelLogAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Command-line tool for interacting with the LLM API."
    )
    parser.add_argument(
        "content", type=str, nargs="?", help="The content input for the model."
    )
    parser.add_argument(
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The model to use for the request. (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming response (default is to stream tokens)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="SiliconFlow API key. Can also be set via SILICONFLOW_API_KEY environment variable.",
    )
    parser.add_argument(
        "--save-key",
        action="store_true",
        help="Save the API key to use in future sessions.",
    )

    # New reasoning and file analysis arguments
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a file to analyze. If provided, the AI will analyze this file.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable detailed reasoning process logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to store reasoning logs (default: ~/.llm-cli-tool/reasoning_logs)",
    )
    parser.add_argument(
        "--analyze-logs",
        type=str,
        metavar="SESSION_ID",
        help="Analyze reasoning logs for a specific session ID.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all available reasoning sessions.",
    )

    # AI模型日志分析参数
    parser.add_argument(
        "--analyze-model-log",
        type=str,
        metavar="LOG_FILE",
        help="智能分析AI训练/推理日志文件，识别异常和性能问题",
    )
    parser.add_argument(
        "--model-log-question",
        type=str,
        help="针对AI模型日志的具体分析问题或关注点",
    )
    parser.add_argument(
        "--use-llm-analysis",
        action="store_true",
        help="使用LLM智能辅助分析日志（需要API密钥）",
    )

    args = parser.parse_args()

    # Initialize reasoning logger if needed
    logger = None
    if (
        args.reasoning
        or args.analyze_logs
        or args.list_sessions
        or args.analyze_model_log
    ):
        logger = ReasoningLogger(args.log_dir)

    # Handle AI model log analysis
    if args.analyze_model_log:
        handle_model_log_analysis(args, logger)
        return

    # Handle log analysis and session listing
    if args.analyze_logs:
        analysis = logger.analyze_session(args.analyze_logs)
        print(f"Session Analysis for {args.analyze_logs}:")
        print("-" * 50)
        for key, value in analysis.items():
            print(f"{key}: {value}")
        return

    if args.list_sessions:
        sessions = logger.list_sessions()
        if sessions:
            print("Available reasoning sessions:")
            print("-" * 30)
            for session in sessions:
                print(f"  {session}")
        else:
            print("No reasoning sessions found.")
        return

    # Handle file analysis
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)

        # Prepare file analysis prompt
        file_prompt = prepare_file_analysis_prompt(args.file, args.content)
        args.content = file_prompt

        if logger:
            logger.start_new_session(
                {
                    "type": "file_analysis",
                    "file_path": args.file,
                    "original_question": args.content,
                }
            )
            logger.log_reasoning_step("file_read", f"Reading file: {args.file}")
    elif not args.content:
        print("Error: Content is required unless using --file option.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Process API key
    saved_key = get_saved_api_key()
    api_key = (
        args.api_key
        if args.api_key is not None
        else (
            os.environ.get("SILICONFLOW_API_KEY")
            if os.environ.get("SILICONFLOW_API_KEY")
            else saved_key
        )
    )

    if not api_key:
        print(
            "Error: API key not provided. Use --api-key option, set SILICONFLOW_API_KEY environment variable, or save a key with --save-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Save the API key if requested and different from the saved one
    if args.save_key and args.api_key and args.api_key != saved_key:
        if save_api_key(args.api_key):
            print("API key saved successfully.")
        else:
            print("Warning: Failed to save API key.", file=sys.stderr)

    # Set the API key for this session
    set_api_key(api_key)

    stream = not args.no_stream

    if logger:
        logger.log_reasoning_step(
            "input", args.content, {"model": args.m, "stream": stream}
        )

    if stream:
        handle_streaming_response(args.content, args.m, logger)
    else:
        handle_normal_response(args.content, args.m, logger)

    if logger:
        logger.end_session("Reasoning session completed successfully")


def handle_streaming_response(content, model, logger=None):
    """Handle streaming responses, printing tokens as they arrive."""
    sys.stdout.write("Response: ")
    sys.stdout.flush()

    if logger:
        logger.log_reasoning_step("streaming_start", "Starting streaming response")

    full_response = ""
    for chunk in send_request(content, model, stream=True):
        if "error" in chunk:
            if logger:
                logger.log_reasoning_step("error", f"Chunk error: {chunk['error']}")
            print(f"\nChunkError: {chunk['error']}", file=sys.stderr)
            sys.exit(1)

        if "content" in chunk:
            if chunk["content"] is not None:
                sys.stdout.write(chunk["content"])
                sys.stdout.flush()
                full_response += chunk["content"]

    # Add a newline at the end
    sys.stdout.write("\n")
    sys.stdout.flush()

    if logger:
        logger.log_reasoning_step("response_complete", full_response)


def handle_normal_response(content, model, logger=None):
    """Handle non-streaming responses."""
    if logger:
        logger.log_reasoning_step("request_start", "Sending non-streaming request")

    response = send_request(content, model, stream=False)

    if "error" in response:
        if logger:
            logger.log_reasoning_step("error", f"Response error: {response['error']}")
        print(f"Error: {response['error']}", file=sys.stderr)
        sys.exit(1)

    if "choices" in response and response["choices"]:
        for i, choice in enumerate(response["choices"]):
            if "message" in choice and "content" in choice["message"]:
                response_content = choice["message"]["content"]
                print(f"Response {i+1}:\n{response_content}")
                if logger:
                    logger.log_reasoning_step(
                        "response_complete", response_content, {"choice_index": i}
                    )
    else:
        print("No valid response received.")
        if logger:
            logger.log_reasoning_step("error", "No valid response received")


def handle_model_log_analysis(args, logger=None):
    """处理AI模型日志分析"""
    if not os.path.exists(args.analyze_model_log):
        print(f"错误: 日志文件 '{args.analyze_model_log}' 不存在.", file=sys.stderr)
        sys.exit(1)

    print(f"正在分析AI模型日志文件: {args.analyze_model_log}")
    print("-" * 60)

    # 创建模型日志分析器
    analyzer = ModelLogAnalyzer()

    # 检查是否使用LLM辅助分析
    if args.use_llm_analysis:
        # 使用LLM辅助分析
        analysis_result = analyzer.analyze_with_llm_assistance(
            args.analyze_model_log, args.model_log_question
        )

        if "error" in analysis_result:
            print(f"分析失败: {analysis_result['error']}", file=sys.stderr)
            sys.exit(1)

        # 显示基础分析结果
        print_model_log_analysis(analysis_result, args.model_log_question)

        # 检查API密钥和进行LLM分析
        saved_key = get_saved_api_key()
        api_key = os.environ.get("SILICONFLOW_API_KEY", saved_key)

        if not api_key:
            print("\n" + "=" * 60)
            print("🤖 LLM智能分析")
            print("=" * 60)
            print("❌ 未找到API密钥，无法进行LLM智能分析")
            print("请设置SILICONFLOW_API_KEY环境变量或使用--save-key保存密钥")
            print("\n📋 系统提示预览:")
            print("-" * 40)
            llm_data = analysis_result.get("llm_analysis", {})
            system_prompt = llm_data.get("system_prompt", "")
            # 显示系统提示的前500字符
            print(
                system_prompt[:500] + "..."
                if len(system_prompt) > 500
                else system_prompt
            )
        else:
            # 设置API密钥并进行LLM分析
            set_api_key(api_key)

            print("\n" + "=" * 60)
            print("🤖 LLM智能分析中...")
            print("=" * 60)

            llm_data = analysis_result.get("llm_analysis", {})
            system_prompt = llm_data.get("system_prompt", "")

            # 构建完整的分析提示
            full_prompt = (
                f"请基于以下AI模型日志分析数据提供专业诊断:\n\n{system_prompt}"
            )

            # 进行LLM分析
            try:
                print("📤 正在发送分析请求到LLM...")
                response = send_request(full_prompt, args.m, stream=False)

                if "error" in response:
                    print(f"❌ LLM分析失败: {response['error']}", file=sys.stderr)
                else:
                    print("✅ LLM分析完成:")
                    print("-" * 40)
                    if "choices" in response and response["choices"]:
                        llm_response = response["choices"][0]["message"]["content"]
                        print(llm_response)

                        # 保存LLM分析结果
                        analysis_result["llm_analysis"]["response"] = llm_response
                    else:
                        print("❌ 未收到有效的LLM响应")

            except Exception as e:
                print(f"❌ LLM分析过程中发生错误: {str(e)}", file=sys.stderr)
    else:
        # 基础分析
        analysis_result = analyzer.analyze_log_file(args.analyze_model_log)

        if "error" in analysis_result:
            print(f"分析失败: {analysis_result['error']}", file=sys.stderr)
            sys.exit(1)

        # 显示分析结果
        print_model_log_analysis(analysis_result, args.model_log_question)

    # 如果启用了推理日志记录
    if logger:
        logger.start_new_session(
            {
                "type": "model_log_analysis",
                "log_file": args.analyze_model_log,
                "question": args.model_log_question,
                "use_llm": args.use_llm_analysis,
            }
        )
        logger.log_reasoning_step("analysis_complete", analysis_result)
        logger.end_session("Model log analysis completed")


def print_model_log_analysis(analysis_result, specific_question=None):
    """打印模型日志分析结果"""
    summary = analysis_result.get("summary", {})
    anomalies = analysis_result.get("anomalies", [])
    recommendations = analysis_result.get("recommendations", [])

    # 基本信息
    print(f"📄 文件: {analysis_result.get('file_path', 'N/A')}")
    print(f"📊 总行数: {analysis_result.get('total_lines', 0)}")
    print(f"⏰ 分析时间: {analysis_result.get('analysis_time', 'N/A')}")
    print()

    # 异常统计
    anomaly_summary = summary.get("anomalies", {})
    print("🚨 异常统计:")
    print(f"  总计: {anomaly_summary.get('total', 0)}")

    if "by_severity" in anomaly_summary:
        print("  按严重程度:")
        for severity, count in anomaly_summary["by_severity"].items():
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                severity, "⚪"
            )
            print(f"    {emoji} {severity}: {count}")

    if "by_category" in anomaly_summary:
        print("  按类别:")
        for category, count in anomaly_summary["by_category"].items():
            print(f"    • {category}: {count}")
    print()

    # 指标摘要
    print("📈 关键指标摘要:")
    metrics_displayed = 0
    for metric_name, metric_data in summary.items():
        if metric_name == "anomalies" or not isinstance(metric_data, dict):
            continue
        if "count" in metric_data and metric_data["count"] > 0:
            print(f"  {metric_name}:")
            print(f"    数量: {metric_data['count']}")
            print(f"    范围: {metric_data['min']:.4f} - {metric_data['max']:.4f}")
            print(f"    平均: {metric_data['avg']:.4f}")
            print(f"    最新: {metric_data['last']:.4f}")
            print(f"    趋势: {metric_data.get('trend', 'unknown')}")
            metrics_displayed += 1
            if metrics_displayed >= 5:  # 只显示前5个指标
                break

    if metrics_displayed == 0:
        print("  未检测到关键指标")
    print()

    # 重要异常详情
    if anomalies:
        print("⚠️  重要异常详情:")
        critical_anomalies = [a for a in anomalies if a.get("severity") == "critical"]
        high_anomalies = [a for a in anomalies if a.get("severity") == "high"]

        # 显示关键异常
        for anomaly in critical_anomalies[:3]:  # 最多显示3个关键异常
            print(f"  🔴 严重: {anomaly.get('description', 'N/A')}")
            print(f"     行号: {anomaly.get('line_number', 'N/A')}")
            print(f"     建议: {anomaly.get('suggestion', 'N/A')}")
            print()

        # 显示高级异常
        for anomaly in high_anomalies[:2]:  # 最多显示2个高级异常
            print(f"  🟠 高级: {anomaly.get('description', 'N/A')}")
            print(f"     行号: {anomaly.get('line_number', 'N/A')}")
            print(f"     建议: {anomaly.get('suggestion', 'N/A')}")
            print()

    # 推荐操作
    if recommendations:
        print("💡 推荐操作:")
        for i, rec in enumerate(recommendations[:5], 1):  # 最多显示5个建议
            print(f"  {i}. {rec}")
        print()

    # 如果有特定问题，提供针对性分析
    if specific_question:
        print(f"🎯 针对问题 '{specific_question}' 的分析:")
        print("  (基于检测到的异常和指标)")

        # 根据问题关键词匹配相关异常和指标
        question_lower = specific_question.lower()
        relevant_anomalies = []
        relevant_metrics = []

        # 匹配异常
        for anomaly in anomalies:
            category = anomaly.get("category", "").lower()
            if any(
                keyword in question_lower
                for keyword in ["性能", "延迟", "latency", "throughput", "吞吐"]
            ) and any(
                keyword in category
                for keyword in ["performance", "gpu", "throughput", "latency"]
            ):
                relevant_anomalies.append(anomaly)
            elif any(
                keyword in question_lower
                for keyword in ["训练", "training", "问题", "error", "错误"]
            ) and any(
                keyword in category
                for keyword in ["training", "memory", "gpu", "convergence"]
            ):
                relevant_anomalies.append(anomaly)
            elif any(
                keyword in question_lower for keyword in ["内存", "memory", "gpu"]
            ) and any(keyword in category for keyword in ["memory", "gpu"]):
                relevant_anomalies.append(anomaly)

        # 匹配指标趋势
        for metric_name, metric_data in summary.items():
            if metric_name == "anomalies" or not isinstance(metric_data, dict):
                continue
            if any(
                keyword in question_lower for keyword in ["性能", "吞吐", "throughput"]
            ) and metric_name in ["throughput", "latency"]:
                relevant_metrics.append((metric_name, metric_data))
            elif any(
                keyword in question_lower for keyword in ["训练", "training"]
            ) and metric_name in ["loss", "accuracy", "learning_rate"]:
                relevant_metrics.append((metric_name, metric_data))

        if relevant_anomalies:
            print("  相关异常:")
            for anomaly in relevant_anomalies[:3]:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(anomaly.get("severity", "low"), "⚪")
                print(
                    f"    {severity_emoji} {anomaly.get('description', 'N/A')} (行 {anomaly.get('line_number', 'N/A')})"
                )

        if relevant_metrics:
            print("  相关指标趋势:")
            for metric_name, metric_data in relevant_metrics[:3]:
                trend = metric_data.get("trend", "stable")
                trend_emoji = {
                    "increasing": "📈",
                    "decreasing": "📉",
                    "stable": "➡️",
                }.get(trend, "➡️")
                print(
                    f"    {trend_emoji} {metric_name}: {trend} (当前: {metric_data.get('last', 'N/A')})"
                )

        if not relevant_anomalies and not relevant_metrics:
            print("  未发现与该问题直接相关的异常或关键指标变化")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
