"""
MCP 工具池维护脚本 —— 定期清理过期工具 + LLM 评估功能冗余

用法:
    # 模拟运行（只查看，不删除）
    python tools/manage_tools.py

    # 实际执行清理
    python tools/manage_tools.py --execute

    # 自定义过期天数和使用次数阈值
    python tools/manage_tools.py --days 14 --min-usage 2

    # 仅过期清理，跳过 LLM 评估
    python tools/manage_tools.py --no-llm
"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from tools.mcp_pool import MCPToolPool


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MCP 工具池维护")
    parser.add_argument("--execute", action="store_true", help="实际执行删除（默认模拟）")
    parser.add_argument("--days", type=int, default=7, help="过期天数阈值（默认7天）")
    parser.add_argument("--min-usage", type=int, default=1, help="最少使用次数阈值（默认1，即从未使用过的工具才可能被清理）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 冗余评估")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI 兼容 API Key")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="LLM 模型名")
    args = parser.parse_args()

    dry_run = not args.execute
    pool = MCPToolPool(
        pool_file=os.path.join(root_dir, "tools", "mcp_tools.json"),
        code_dir=os.path.join(root_dir, "tools", "tool_add", "tool_direct"),
    )

    # 创建 LLM 客户端（可选）
    client = None
    if not args.no_llm and args.api_key:
        from openai import OpenAI
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 运行维护
    report = pool.maintenance(
        client=client,
        model=args.model,
        days_unused=args.days,
        min_usage=args.min_usage,
        dry_run=dry_run,
    )
    print(report)

    if dry_run:
        print("\n💡 使用 --execute 参数执行实际删除")
        print("💡 使用 --api-key YOUR_KEY 启用 LLM 冗余评估")


if __name__ == "__main__":
    main()
