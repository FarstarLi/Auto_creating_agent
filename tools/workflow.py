"""
工作流引擎 — 支持将复杂任务拆分为多步工作流，后台异步执行。

用法:
    from tools.workflow import WorkflowEngine
    engine = WorkflowEngine(workflow_dir="./workflows")

    # 创建
    wf = engine.create("获取热搜", "curl请求 → 解析HTML → 输出top10",
        steps=[{"tool":"run_command","args":{"command":"curl..."}},
               {"tool":"create_file","args":{"path":"result.txt","content":"..."}}])

    # 执行（后台线程）
    engine.run_async(wf.id, pool)

    # 查询
    status = engine.get(wf.id)
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class WorkflowEngine:
    """工作流管理器"""

    def __init__(self, workflow_dir: str = "./workflows"):
        self.workflow_dir = workflow_dir
        os.makedirs(workflow_dir, exist_ok=True)
        self._running: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ==================== CRUD ====================

    def create(
        self,
        name: str,
        description: str,
        steps: List[Dict],
        parent_task: str = "",
    ) -> Dict:
        """
        创建一个新工作流。

        Args:
            name: 工作流名称
            description: 用途描述
            steps: [{tool, args}, ...] 执行步骤
            parent_task: 关联的父任务

        Returns:
            工作流对象
        """
        wf_id = f"wf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(os.listdir(self.workflow_dir))}"
        wf = {
            "id": wf_id,
            "name": name,
            "description": description,
            "parent_task": parent_task,
            "status": "pending",
            "steps": steps,
            "current_step": 0,
            "results": [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self._save(wf)
        return wf

    def get(self, wf_id: str) -> Optional[Dict]:
        """获取工作流状态"""
        path = os.path.join(self.workflow_dir, f"{wf_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_workflows(self, status: Optional[str] = None) -> List[Dict]:
        """列出所有工作流"""
        workflows = []
        for f in os.listdir(self.workflow_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.workflow_dir, f), "r", encoding="utf-8") as fp:
                        wf = json.load(fp)
                        if status is None or wf.get("status") == status:
                            workflows.append(wf)
                except Exception:
                    pass
        return sorted(workflows, key=lambda w: w.get("created_at", ""), reverse=True)

    def delete(self, wf_id: str) -> bool:
        """删除工作流"""
        path = os.path.join(self.workflow_dir, f"{wf_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ==================== 执行 ====================

    def run_async(self, wf_id: str, pool: Any):
        """
        在后台线程中异步执行工作流。

        Args:
            wf_id: 工作流 ID
            pool: MCPToolPool 实例（用于执行每个步骤的工具调用）
        """
        with self._lock:
            if wf_id in self._running:
                return
            t = threading.Thread(target=self._run, args=(wf_id, pool), daemon=True)
            self._running[wf_id] = t
            t.start()

    def _run(self, wf_id: str, pool: Any):
        """同步执行工作流所有步骤"""
        wf = self.get(wf_id)
        if not wf:
            return

        wf["status"] = "running"
        self._save(wf)

        try:
            for i, step in enumerate(wf["steps"]):
                wf["current_step"] = i + 1
                self._save(wf)

                tool_name = step.get("tool", "")
                args = step.get("args", {})
                try:
                    result = pool.execute(tool_name, args)
                    wf["results"].append({
                        "step": i + 1,
                        "tool": tool_name,
                        "success": "成功" in result or not any(
                            kw in result for kw in ["失败", "错误", "Error"]),
                        "result": result[:500],
                    })
                except Exception as e:
                    wf["results"].append({
                        "step": i + 1,
                        "tool": tool_name,
                        "success": False,
                        "result": f"异常: {str(e)}",
                    })
                    wf["status"] = "failed"
                    self._save(wf)
                    return

                time.sleep(0.1)  # 避免过快

            wf["status"] = "completed"
            wf["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            wf["status"] = "failed"
            wf["results"].append({"step": -1, "tool": "", "success": False, "result": f"工作流异常: {e}"})
        finally:
            self._save(wf)
            with self._lock:
                self._running.pop(wf_id, None)

    def run_sync(self, wf_id: str, pool: Any) -> str:
        """
        同步执行并等待完成（阻塞）。

        Returns:
            最终结果摘要
        """
        self._run(wf_id, pool)
        wf = self.get(wf_id)
        if not wf:
            return "工作流不存在"
        return self._format_result(wf)

    # ==================== 工具方法 ====================

    def _save(self, wf: Dict):
        path = os.path.join(self.workflow_dir, f"{wf['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)

    def _format_result(self, wf: Dict) -> str:
        """格式化工作流结果为可读文本"""
        lines = [
            f"工作流: {wf['name']}",
            f"状态: {wf['status']}",
            f"步骤: {wf['current_step']}/{len(wf['steps'])}",
        ]
        for r in wf.get("results", []):
            status = "✅" if r.get("success") else "❌"
            lines.append(f"  {status} 步骤{r['step']}: {r['tool']} → {r['result'][:100]}")
        return "\n".join(lines)

    def summary(self) -> str:
        """工作流引擎概览"""
        all_wf = self.list_workflows()
        pending = sum(1 for w in all_wf if w["status"] == "pending")
        running = sum(1 for w in all_wf if w["status"] == "running")
        completed = sum(1 for w in all_wf if w["status"] == "completed")
        failed = sum(1 for w in all_wf if w["status"] == "failed")
        return (
            f"📋 工作流引擎: {len(all_wf)} 个\n"
            f"  待执行: {pending} | 运行中: {running} | 完成: {completed} | 失败: {failed}"
        )
