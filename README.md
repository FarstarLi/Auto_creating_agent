# 自主工具智能体 v2.4

LLM 驱动的自主工具智能体 —— 不止是调用工具，更能**自主创建工具**来完成任务。

当 MCP 工具池中缺少所需工具时，智能体会自动生成 Python 代码，经过安全检查后持久化并立即调用。配合四状态机和三层记忆系统，实现对复杂任务的自主规划、执行、纠错与知识积累。

## 核心特性

- **🧠 四状态机** — THINK → EXECUTE ⇄ REFLECT → END，支持多轮内循环与断点续跑
- **🔧 自主工具创建** — LLM 发现工具缺失时自动生成 Python 代码，安全校验后立即可用
- **🧳 三层记忆** — 工作记忆（Token 预算）→ 短期记忆（事实提取）→ 长期记忆（语义检索 + 遗忘曲线）
- **🛡️ 安全沙箱** — 代码安全检查、工作空间隔离、危险命令/敏感文件拦截
- **🔌 多模型适配** — 适配器模式支持 OpenAI / DeepSeek / Ollama 等兼容接口
- **💾 断点续跑** — 任务超限自动保存断点，下次启动继续执行

## 项目结构

```
auto_coding/
├── main.py                        入口 + REPL 交互 + / 命令系统
├── config.py                      配置加载器（属性式访问）
├── config.json                    运行时配置（API Key、模型、参数）
├── config.example.json            配置模板
├── system_prompt.txt              智能体系统提示词
├── limitation.txt                 安全限制规则（禁用命令/文件）
├── requirements.txt               Python 依赖
│
├── brain/                         大脑模块（四状态机核心）
│   ├── brain.py                   AgentBrain — 状态机调度 + 工具注册 + LLM 调用
│   ├── adapters.py                多模型适配器（OpenAI / Ollama）
│   └── state.py                   状态数据结构（LoopMode / LoopState）
│
├── memory/                        三层记忆模块
│   ├── manager.py                 MemoryManager — 记忆编排（流转/摘要/巩固/检索）
│   ├── models.py                  数据模型（Message / MemoryItem / LongTermMemory）
│   └── embeddings.py              Embedding 引擎（TF-IDF 本地 + OpenAI 远程）
│
├── tools/                         工具池模块
│   ├── mcp_pool.py                MCPToolPool — 工具注册/执行/安全/清理
│   ├── mcp_tools.json             工具元数据注册表
│   ├── manage_tools.py            工具管理 CLI
│   └── tool_add/tool_direct/      内置工具实现（7个 .py 文件）
│
├── workspace/                     工作空间（文件操作隔离区）
└── test/                          测试
    ├── test_state_machine.py      四状态机流程测试（Mock LLM）
    ├── test_memory.py             三层记忆功能测试
    ├── test_brain.py              MCP 工具池基础测试
    ├── test_stress.py             压力测试（100+ 轮交互）
    └── test_llm_create_tool.py    LLM 自主创建工具集成测试
```

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI 兼容 API Key（推荐 [DeepSeek](https://platform.deepseek.com)，性价比高）

### 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd auto_coding

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp config.example.json config.json
# 编辑 config.json，填入 API Key：
#   "api_key": "sk-xxxxxxxxxxxxxxxx"
#   默认使用 DeepSeek，换其他 API 改 base_url 即可
```

### 运行

```bash
python main.py
```

交互示例：

```
User：帮我查豆瓣电影Top10
      → THINK: 无爬虫工具 → plan=create_tool
      → EXECUTE: 创建爬虫 → 调用 → 获取数据
      → END: 数据已保存到 workspace/

User：求100的阶乘
      → THINK: 池中已有 factorial → plan=直接调用
      → EXECUTE: factorial(100) → 得到结果
```

## 架构详解

### 四状态机

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────┐
│  THINK  │────▶│ EXECUTE │────▶│ REFLECT │────▶│ END │
│ 分析规划 │     │ 执行工具 │◀────│ 错误恢复 │     │ 输出 │
└─────────┘     └─────────┘     └─────────┘     └─────┘
      ▲               │               │
      └───────────────┴───────────────┘
              重新规划
```

| 状态 | 功能 | 关键行为 |
|------|------|----------|
| **THINK** | 分析任务，判断工具可用性 | 多轮内循环（最多3轮）；防幻觉护栏：未实质执行工具则 `done` 强制不通过 |
| **EXECUTE** | 调用/创建工具 | 首轮隐藏 `run_command` 防止直接调用命令；重复调用检测 → 自动切换方案 |
| **REFLECT** | 出错后分析原因、寻找方案 | "工具不存在" → 程序化跳过 LLM 直接创建；多轮反思最多 3 轮 |
| **END** | 生成友好回复 | 自动清理 XML 残留；保存对话记忆 |

### 三层记忆

```
工作记忆 (L1)              短期记忆 (L2)              长期记忆 (L3)
┌──────────────┐  Token溢出  ┌──────────────┐  巩固迁移  ┌──────────────┐
│ 当前对话消息  │──LLM摘要──▶│ 摘要 + 事实   │──向量化──▶│ 语义记忆     │
│ Token预算管理 │            │ 最多200项     │           │ 最多1000项   │
│ 上限4000t     │            │ 衰减遗忘      │           │ 混合检索     │
└──────────────┘            └──────────────┘           └──────────────┘
                                    │                         │
                                    └──── 反思合并 ◀──────────┘
```

- **L1 工作记忆**：当前对话的全部消息，Token 超 60% 预算自动触发摘要压缩
- **L2 短期记忆**：LLM 摘要 + 关键词提取的事实片段，带 `decay_score()` 衰减评分
- **L3 长期记忆**：跨会话持久化，支持 TF-IDF + OpenAI Embedding 混合检索

### 工具池

7 个内置核心工具 + LLM 自主创建：

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入/追加文件（w/a 模式） |
| `create_file` | 创建文件（含 overwrite 选项） |
| `delete_file` | 删除文件 |
| `create_directory` | 创建目录（递归） |
| `run_command` | 执行系统命令（受限制，仅 pip/版本检查） |
| `delete_tool` | 删除指定工具 |

工具创建流程：LLM 生成代码 → 语法编译检查 → 安全检查（拦截 os/subprocess/eval）→ 中文标点修正 → 写入 .py → 注册到 JSON → 立即可用

## 配置说明

`config.json` 完整配置项：

```json
{
  "llm": {
    "api_key": "sk-xxx",              // API Key（必填）
    "base_url": "https://api.deepseek.com",  // API 地址
    "model": "deepseek-chat",         // 模型名称
    "max_tokens": 4096,               // 单次最大 Token
    "temperature": 0.7,               // 生成温度
    "timeout": 120                    // 请求超时（秒）
  },
  "brain": {
    "max_iterations": 100,            // 状态机最大迭代轮次
    "max_retries": 3,                 // 最大重试次数
    "system_prompt_file": "./system_prompt.txt"  // 系统提示词文件
  },
  "memory": {
    "working_token_budget": 4000,     // 工作记忆 Token 预算
    "short_term_max_items": 200,      // 短期记忆上限
    "long_term_max_items": 1000,      // 长期记忆上限
    "working_memory_file": "./memory/archives/conversation_memory.json",
    "long_term_memory_file": "./memory/archives/long_term_archive.json"
  },
  "tools": {
    "pool_file": "./tools/mcp_tools.json",      // 工具注册表
    "code_dir": "./tools/tool_add/tool_direct", // 工具代码目录
    "workspace_dir": "./workspace"              // 工作空间（文件操作隔离区）
  }
}
```

## 内置命令

REPL 中可用 `/` 命令：

| 命令 | 功能 |
|------|------|
| `/help` `/` | 显示命令列表 |
| `/model [name]` | 查看/切换模型 |
| `/config` | 查看当前配置（API Key 脱敏） |
| `/tools` | 查看工具池（含过期标记） |
| `/memory` | 查看三层记忆统计 |
| `/system [text]` | 查看/修改系统提示词 |
| `/temp  [value]` | 查看/调整 temperature |
| `/clear` | 清空对话记忆 |
| `/save` | 手动保存记忆到磁盘 |
| `/continue` | 续跑被终止的上一个任务 |
| `/exit` `/quit` | 退出并自动保存 |

退出时自动保存长期记忆和工具池状态。

## API 调用

可作为库嵌入其他项目：

```python
from brain.brain import AgentBrain
from brain.adapters import ModelConfig, ModelProvider

brain = AgentBrain(
    model_configs=[
        ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="deepseek-chat",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
        ),
    ],
)

# REPL 模式（保留上下文，适合多轮对话）
brain.process_turn("帮我写一个排序函数")

# 断点续跑
if brain.has_breakpoint():
    brain.resume()
```

## 安全机制

- **代码安全检查**：拦截 `os`/`subprocess`/`shutil`/`eval`/`exec` 等危险调用
- **工作空间隔离**：所有文件操作限制在 `workspace/` 目录内，防止目录穿越
- **命令拦截**：`limitation.txt` 定义禁止的命令模式（`rm -rf /`、`format *` 等）
- **敏感文件保护**：禁止读取 `config.json`、`.env`、密钥文件等

自定义安全规则：编辑 `limitation.txt`，在 `[commands]` 或 `[files]` 段添加规则，支持 `*` 通配符。

## 测试

```bash
# MCP 工具池基础功能
python test/test_brain.py

# 四状态机流程（Mock LLM，无需 API）
python test/test_state_machine.py

# 三层记忆功能
python test/test_memory.py

# 压力测试（100+ 轮交互）
python test/test_stress.py

# LLM 自主创建工具集成测试（需 API Key）
python test/test_llm_create_tool.py
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM 接口 | OpenAI 兼容 API（DeepSeek / OpenAI / SiliconFlow / Ollama） |
| 数据模型 | Pydantic v2 |
| Embedding | OpenAI text-embedding-3-small / 本地 TF-IDF |
| HTTP 客户端 | requests |
| 测试框架 | pytest + Mock Adapter |

## License

MIT
