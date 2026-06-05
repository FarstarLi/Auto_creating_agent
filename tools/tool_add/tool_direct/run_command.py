import subprocess
import platform

_IS_WINDOWS = platform.system() == "Windows"

def run_command(command: str) -> str:
    """执行系统命令（谨慎使用）"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr] {stderr}")
        if result.returncode != 0 and not stdout and not stderr:
            parts.append(f"命令退出码: {result.returncode}")

        if parts:
            return "\n".join(parts)

        # 无输出时给出提示
        hint = ""
        if _IS_WINDOWS:
            if "python3" in command:
                hint = " (提示: Windows 上 Python 命令是 'python' 不是 'python3')"
            elif "$((" in command or "${" in command:
                hint = " (提示: Windows 使用 cmd.exe，不支持 bash 语法，请用 python 代替)"
        return f"命令执行完成但无输出{hint}"
    except subprocess.TimeoutExpired:
        return "命令执行超时(15秒)"
    except Exception as e:
        return f"执行命令出错：{e}"
