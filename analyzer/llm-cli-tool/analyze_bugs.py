#!/usr/bin/env python3
"""
AI推理日志Bug分析工具
自动检测推理过程中的问题和bug
"""

import json
import re
from typing import Dict, List, Any
from src.reasoning_logger import ReasoningLogger


class ReasoningBugAnalyzer:
    """分析推理日志中的潜在问题和bug"""

    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def analyze_session_for_bugs(self, session_id: str) -> Dict[str, Any]:
        """
        分析会话中的潜在bug和问题

        Args:
            session_id: 会话ID

        Returns:
            分析结果字典
        """
        logs = self.logger.get_session_logs(session_id)
        analysis = self.logger.analyze_session(session_id)

        if not logs:
            return {"error": "No logs found"}

        issues = {
            "errors": [],
            "warnings": [],
            "performance_issues": [],
            "logical_inconsistencies": [],
            "incomplete_responses": [],
            "summary": {},
        }

        # 检查错误
        issues["errors"] = self._detect_errors(logs)

        # 检查警告
        issues["warnings"] = self._detect_warnings(logs)

        # 检查性能问题
        issues["performance_issues"] = self._detect_performance_issues(logs, analysis)

        # 检查逻辑不一致
        issues["logical_inconsistencies"] = self._detect_logical_issues(logs)

        # 检查不完整响应
        issues["incomplete_responses"] = self._detect_incomplete_responses(logs)

        # 生成摘要
        issues["summary"] = self._generate_summary(issues)

        return issues

    def _detect_errors(self, logs: List[Dict]) -> List[Dict]:
        """检测明确的错误"""
        errors = []

        for i, log in enumerate(logs):
            # 检查错误step_type
            if log.get("step_type") == "error":
                errors.append(
                    {
                        "type": "explicit_error",
                        "step": i + 1,
                        "content": log.get("content", ""),
                        "severity": "high",
                    }
                )

            # 检查内容中的错误关键词
            content = log.get("content", "").lower()
            error_keywords = [
                "error",
                "failed",
                "exception",
                "traceback",
                "错误",
                "失败",
                "异常",
            ]

            for keyword in error_keywords:
                if keyword in content:
                    errors.append(
                        {
                            "type": "error_keyword",
                            "step": i + 1,
                            "keyword": keyword,
                            "content": log.get("content", "")[:200] + "...",
                            "severity": "medium",
                        }
                    )
                    break

        return errors

    def _detect_warnings(self, logs: List[Dict]) -> List[Dict]:
        """检测警告"""
        warnings = []

        for i, log in enumerate(logs):
            content = log.get("content", "").lower()

            # 检查API相关问题
            if "api" in content and any(
                word in content for word in ["limit", "quota", "rate", "限制", "配额"]
            ):
                warnings.append(
                    {
                        "type": "api_limit_warning",
                        "step": i + 1,
                        "content": log.get("content", "")[:200] + "...",
                        "severity": "medium",
                    }
                )

            # 检查超时问题
            if any(word in content for word in ["timeout", "超时", "time out"]):
                warnings.append(
                    {
                        "type": "timeout_warning",
                        "step": i + 1,
                        "content": log.get("content", "")[:200] + "...",
                        "severity": "low",
                    }
                )

        return warnings

    def _detect_performance_issues(
        self, logs: List[Dict], analysis: Dict
    ) -> List[Dict]:
        """检测性能问题"""
        issues = []

        # 检查响应时间
        duration = analysis.get("duration", 0)
        if duration > 60:  # 超过60秒
            issues.append(
                {
                    "type": "slow_response",
                    "duration": duration,
                    "severity": "medium",
                    "description": f"响应时间过长: {duration:.2f}秒",
                }
            )

        # 检查步骤数量
        total_steps = analysis.get("total_steps", 0)
        if total_steps > 20:
            issues.append(
                {
                    "type": "too_many_steps",
                    "steps": total_steps,
                    "severity": "low",
                    "description": f"推理步骤过多: {total_steps}步",
                }
            )

        return issues

    def _detect_logical_issues(self, logs: List[Dict]) -> List[Dict]:
        """检测逻辑不一致问题"""
        issues = []

        # 检查是否有前后矛盾的陈述
        contents = [
            log.get("content", "")
            for log in logs
            if log.get("step_type") == "response_complete"
        ]

        for i, content in enumerate(contents):
            # 简单的矛盾检测（可以扩展）
            if "不是" in content and "是" in content:
                issues.append(
                    {
                        "type": "contradiction",
                        "step": i + 1,
                        "content": content[:200] + "...",
                        "severity": "medium",
                        "description": "可能存在前后矛盾的陈述",
                    }
                )

        return issues

    def _detect_incomplete_responses(self, logs: List[Dict]) -> List[Dict]:
        """检测不完整的响应"""
        issues = []

        # 检查是否有未完成的streaming
        streaming_start_count = sum(
            1 for log in logs if log.get("step_type") == "streaming_start"
        )
        response_complete_count = sum(
            1 for log in logs if log.get("step_type") == "response_complete"
        )

        if streaming_start_count > response_complete_count:
            issues.append(
                {
                    "type": "incomplete_streaming",
                    "severity": "high",
                    "description": f"有{streaming_start_count - response_complete_count}个未完成的流式响应",
                }
            )

        # 检查响应是否过短
        for i, log in enumerate(logs):
            if log.get("step_type") == "response_complete":
                content = log.get("content", "")
                if len(content.strip()) < 50:  # 响应太短
                    issues.append(
                        {
                            "type": "short_response",
                            "step": i + 1,
                            "length": len(content),
                            "severity": "low",
                            "description": f"响应内容过短: 仅{len(content)}字符",
                        }
                    )

        return issues

    def _generate_summary(self, issues: Dict) -> Dict:
        """生成问题摘要"""
        total_issues = (
            len(issues["errors"])
            + len(issues["warnings"])
            + len(issues["performance_issues"])
            + len(issues["logical_inconsistencies"])
            + len(issues["incomplete_responses"])
        )

        high_severity = sum(
            1
            for category in issues.values()
            if isinstance(category, list)
            for issue in category
            if isinstance(issue, dict) and issue.get("severity") == "high"
        )

        return {
            "total_issues": total_issues,
            "high_severity_count": high_severity,
            "errors_count": len(issues["errors"]),
            "warnings_count": len(issues["warnings"]),
            "performance_issues_count": len(issues["performance_issues"]),
            "recommendation": self._get_recommendation(issues),
        }

    def _get_recommendation(self, issues: Dict) -> str:
        """根据发现的问题提供建议"""
        total_issues = (
            len(issues["errors"])
            + len(issues["warnings"])
            + len(issues["performance_issues"])
            + len(issues["logical_inconsistencies"])
            + len(issues["incomplete_responses"])
        )

        high_severity = sum(
            1
            for category in issues.values()
            if isinstance(category, list)
            for issue in category
            if isinstance(issue, dict) and issue.get("severity") == "high"
        )

        if high_severity > 0:
            return "发现严重问题，需要立即处理"
        elif total_issues > 5:
            return "发现多个问题，建议优化"
        elif total_issues > 0:
            return "发现少量问题，可以忽略或稍后处理"
        else:
            return "未发现明显问题"


def main():
    """主函数 - 演示如何使用bug分析器"""
    import argparse

    parser = argparse.ArgumentParser(description="分析推理日志中的bug和问题")
    parser.add_argument("session_id", help="要分析的会话ID")
    parser.add_argument("--log-dir", help="日志目录")
    parser.add_argument("--export", help="导出分析结果到文件")

    args = parser.parse_args()

    logger = ReasoningLogger(args.log_dir)
    analyzer = ReasoningBugAnalyzer(logger)

    print(f"🔍 分析会话: {args.session_id}")
    print("=" * 50)

    # 执行分析
    results = analyzer.analyze_session_for_bugs(args.session_id)

    if "error" in results:
        print(f"❌ 错误: {results['error']}")
        return

    # 显示结果
    summary = results["summary"]
    print(f"📊 问题摘要:")
    print(f"  总问题数: {summary['total_issues']}")
    print(f"  严重问题: {summary['high_severity_count']}")
    print(f"  错误: {summary['errors_count']}")
    print(f"  警告: {summary['warnings_count']}")
    print(f"  性能问题: {summary['performance_issues_count']}")
    print(f"  建议: {summary['recommendation']}")
    print()

    # 显示详细问题
    for category, issues in results.items():
        if category == "summary" or not isinstance(issues, list) or not issues:
            continue

        print(f"🔍 {category.replace('_', ' ').title()}:")
        for issue in issues:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                issue.get("severity", "low"), "⚪"
            )
            print(f"  {severity_icon} {issue.get('type', 'Unknown')}")
            if "description" in issue:
                print(f"     {issue['description']}")
            elif "content" in issue:
                print(f"     {issue['content'][:100]}...")
        print()

    # 导出结果
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 分析结果已导出到: {args.export}")


if __name__ == "__main__":
    main()
