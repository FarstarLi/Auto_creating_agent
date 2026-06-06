"""四状态机定义 —— 思考 → 执行 ⇄ 反思 → 结束"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass, field


class LoopMode(Enum):
    THINK = "think"         # 思考：分析任务、制定计划
    EXECUTE = "execute"     # 执行：调用工具完成任务
    REFLECT = "reflect"     # 反思：出错/失败时分析原因、寻找解决方案
    END = "end"             # 结束：任务完成，输出最终答案


@dataclass
class LoopState:
    """四状态机运行时状态"""
    mode: LoopMode = LoopMode.THINK
    iteration: int = 0
    max_iterations: int = 10
    task: str = ""

    # 状态数据
    thought: str = ""          # THINK 产出：分析结果
    plan: str = ""             # THINK 产出：执行计划
    result: str = ""           # EXECUTE 产出：工具执行结果
    reflection: str = ""       # REFLECT 产出：原因分析+解决方案
    final_answer: str = ""     # END 产出：最终答案

    # 错误追踪
    last_error: str = ""
    error_count: int = 0
    max_retries: int = 3

    # 多轮内循环控制（THINK/REFLECT 阶段可进行多轮内部迭代）
    max_think_rounds: int = 3      # THINK 阶段最多内循环轮数
    max_reflect_rounds: int = 3    # REFLECT 阶段最多内循环轮数
    think_rounds: int = 0          # 当前 THINK 轮次（每轮 THINK 重置）
    reflect_rounds: int = 0        # 当前 REFLECT 轮次（每轮 REFLECT 重置）

    # 多轮思考/反思标志
    already_have_data: bool = False  # THINK 产出：LLM 判断数据已获取
    can_retry: bool = False          # REFLECT 产出：LLM 确认有可执行方案

    # 记录
    history: List[Dict] = field(default_factory=list)
    created_tools: List[Dict] = field(default_factory=list)
