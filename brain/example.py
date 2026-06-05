"""
Brain 使用示例
"""

from config import config
from brain import create_agent, ModelConfig, ModelProvider

# 1. 配置（从 config.json 读取）
configs = [
    ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    ),
]

# 2. 创建智能体（四状态机 + MCP 池 + 三层记忆）
brain = create_agent(
    model_configs=configs,
    system_prompt=config.brain_system_prompt,
    max_iterations=config.brain_max_iterations,
    openai_client=config.create_llm_client(),
)

# 3. 注册自定义工具
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

brain.tools.register("calculate", "计算数学表达式", calculate)

# 4. 执行任务
result = brain.run("计算 1+2+3+...+100 的结果")
print(result)

# 5. 查看历史
for i, step in enumerate(brain.get_history()):
    print(f"步骤 {i+1}: [{step.get('mode')}]")
