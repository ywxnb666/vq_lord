import os
import glob
import subprocess
import threading
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel
import json

app = FastAPI(title="VQ-LoRD Control Backend")

class TaskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def start(self, task_name: str, pid: int):
        with self.lock:
            self.tasks[task_name] = {
                "status": "running",
                "pid": pid,
                "start_time": time.time(),
                "end_time": None,
                "return_code": None,
            }

    def finish(self, task_name: str, return_code: int):
        with self.lock:
            if task_name in self.tasks:
                self.tasks[task_name]["status"] = "success" if return_code == 0 else "failed"
                self.tasks[task_name]["end_time"] = time.time()
                self.tasks[task_name]["return_code"] = return_code

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self.tasks)

task_state = TaskState()

# ---------------------------------------------------------
# 核心执行引擎
# ---------------------------------------------------------
DEFAULT_LOG_SUBDIR = "logs"

def run_bash_script(script_filename: str, env_payload: dict):
    """
    接收前端发来的参数字典，转化为环境变量后执行目标 Shell 脚本。同时将 stdout/stderr 捕获到本地日志文件。
    """
    task_name = script_filename.replace(".sh", "")
    root_dir = env_payload.get("ROOT_DIR", "/root/workspace/vq_lord")
    script_path = os.path.join(root_dir, "scripts2", script_filename)

    if not os.path.exists(script_path):
        print(f"[Error] 脚本未找到: {script_path}")
        return

    log_dir = os.path.join(root_dir, DEFAULT_LOG_SUBDIR, "apis")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name}_{int(time.time())}.log")

    # 1. 继承当前系统环境，防止破坏底层 PATH
    cmd_env = os.environ.copy()
    
    # 2. 将前端传来的 payload 注入并强制转为字符串
    for key, value in env_payload.items():
        if isinstance(value, bool):
            cmd_env[key] = "1" if value else "0"
        else:
            cmd_env[key] = str(value)

    # 3. 启动子进程 (无需挂起等待，直接跑在后台)
    # 标准输出已由脚本中的 common.sh 处理重定向到 .log 文件，故此处可忽略
    try:
        with open(log_file, "w", encoding="utf-8") as flog:
            flog.write(f"=== Task: {script_filename} ===\n")
            flog.write(f"=== Log:  {log_file} ===\n")
            flog.write(f"=== Time: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            flog.flush()

            process = subprocess.Popen(
                ["bash", script_path],
                env=cmd_env,
                stdout=flog,
                stderr=subprocess.STDOUT,  # 合并 stderr 到同一个文件
            )
            task_state.start(task_name, process.pid)
            print(f"[Info] 已启动 {script_filename}, PID={process.pid}, LOG={log_file}")

            # 阻塞等待完成（此函数本身跑在后台线程中，不会卡住 FastAPI）
            return_code = process.wait()
            task_state.finish(task_name, return_code)

            flog.write(f"\n=== Process exited with code {return_code} ===\n")
            print(f"[Info] {script_filename} 完成, 返回码={return_code}")

    except Exception as e:
        task_state.finish(task_name, -1)
        print(f"[Error] 执行抛出异常: {e}")


# ---------------------------------------------------------
# 路由接口
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "VQ-LoRD Backend is running"}

# ---- 任务状态查询 ----
@app.get("/api/tasks")
def get_task_status():
    return task_state.get_all()

# ---- Tab 1: 教师数据采集 ----
@app.post("/api/task/collect")
def task_collect(payload: Dict[str, Any] = Body(...)):
    # 使用线程而非 BackgroundTasks，以便追踪 wait() 返回码
    t = threading.Thread(
        target=run_bash_script,
        args=("teacher_model_data_collect.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Teacher data collection task spawned."}

# ---- Tab 1: 分桶预处理 ----
@app.post("/api/task/preprocess")
def task_preprocess(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("data_preprocess.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Data preprocessing task spawned."}

# ---- Tab 2: Stage1 VQ 码本训练 ----
@app.post("/api/task/stage1")
def task_stage1(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("run_stage1.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Stage1 VQ codebook training task spawned."}

# ---- Tab 3: Stage2 视觉蒸馏 ----
@app.post("/api/task/stage2")
def task_stage2(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("run_stage2.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Stage2 vision distillation task spawned."}

# ---- Tab 4: Stage3 LoRD 偏好对齐 ----
@app.post("/api/task/stage3")
def task_stage3(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("run_stage3.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Stage3 LoRD preference alignment task spawned."}

# ---- 路径探测：列出 ckpt 目录下匹配前缀的子目录 ----
@app.get("/api/list_ckpt_dirs")
def list_ckpt_dirs(root_dir: str = "/root/workspace/vq_lord",
                   prefix: str = "stage1_vq"):
    """列出 vq_lord_ckpts/ 下匹配指定前缀的目录，用于前端路径选择。"""
    ckpt_base = os.path.join(root_dir, "vq_lord_ckpts")
    if not os.path.isdir(ckpt_base):
        return {"dirs": [], "message": f"目录不存在: {ckpt_base}"}

    matched = []
    for name in sorted(os.listdir(ckpt_base)):
        full = os.path.join(ckpt_base, name)
        if os.path.isdir(full) and name.startswith(prefix):
            # 附带关键文件检查
            has_codebook = os.path.isfile(os.path.join(full, "vq_codebook.pt"))
            has_adapter = os.path.isfile(os.path.join(full, "adapter_config.json"))
            has_projector = os.path.isfile(os.path.join(full, "projector.pt"))
            matched.append({
                "name": name,
                "path": full,
                "has_codebook": has_codebook,
                "has_adapter": has_adapter,
                "has_projector": has_projector,
            })
    return {"dirs": matched}

# ---- Tab 5: Stage2 评测 ----
@app.post("/api/task/eval_stage2")
def task_eval_stage2(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("test_vq_lord_stage2.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Stage2 evaluation task spawned."}

# ---- Tab 5: Stage3 评测 ----
@app.post("/api/task/eval_stage3")
def task_eval_stage3(payload: Dict[str, Any] = Body(...)):
    t = threading.Thread(
        target=run_bash_script,
        args=("test_vq_lord_stage3.sh", payload),
        daemon=True,
    )
    t.start()
    return {"status": "ok", "message": "Stage3 evaluation task spawned."}

# ---------------------------------------------------------
# 评测结果读取接口
# ---------------------------------------------------------
@app.get("/api/eval_result")
def get_eval_result(root_dir: str = "/root/workspace/vq_lord",
                    prefix: str = "stage3"):
    """
    在 test_results/ 目录下查找匹配前缀的最新 JSON 结果文件，
    解析并返回 accuracy / format_rate / n 三个指标。
    
    结果 JSON 结构预期（由 sciqa_process2.py 生成）:
    {
        "accuracy": 0.85,
        "format_rate": 0.99,
        "n": 2017,
        ...
    }
    """
    result_dir = os.path.join(root_dir, "test_results")
    if not os.path.isdir(result_dir):
        return {"found": False, "message": f"结果目录不存在: {result_dir}"}

    # 查找匹配前缀的 JSON 文件
    candidates = []
    for name in os.listdir(result_dir):
        if name.startswith(prefix) and name.endswith(".json"):
            full_path = os.path.join(result_dir, name)
            if os.path.isfile(full_path):
                candidates.append(full_path)

    if not candidates:
        return {"found": False, "message": f"未找到 {prefix}*.json 结果文件"}

    # 取最新的
    latest = max(candidates, key=os.path.getmtime)

    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容多种可能的 key 名称
        accuracy = (
            data.get("accuracy")
            or data.get("acc")
            or data.get("ACCURACY")
            or 0.0
        )
        format_rate = (
            data.get("format_rate")
            or data.get("FORMAT_RATE")
            or data.get("format_ratio")
            or 0.0
        )
        n = (
            data.get("n")
            or data.get("N")
            or data.get("total")
            or data.get("num_samples")
            or 0
        )

        return {
            "found": True,
            "result_file": os.path.basename(latest),
            "result_path": latest,
            "accuracy": float(accuracy),
            "format_rate": float(format_rate),
            "n": int(n),
            "raw": data,  # 原始 JSON 完整返回，前端可选展示
        }
    except Exception as e:
        return {"found": False, "message": f"解析结果文件失败: {str(e)}"}


# ---- 列出所有评测结果文件（可选：支持历史对比） ----
@app.get("/api/eval_results_list")
def list_eval_results(root_dir: str = "/root/workspace/vq_lord"):
    """列出 test_results/ 目录下所有 JSON 结果文件的摘要信息。"""
    result_dir = os.path.join(root_dir, "test_results")
    if not os.path.isdir(result_dir):
        return {"results": [], "message": f"目录不存在: {result_dir}"}

    results = []
    for name in sorted(os.listdir(result_dir)):
        if name.endswith(".json"):
            full_path = os.path.join(result_dir, name)
            if not os.path.isfile(full_path):
                continue
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "file": name,
                    "accuracy": float(data.get("accuracy", data.get("acc", 0))),
                    "format_rate": float(data.get("format_rate", data.get("format_ratio", 0))),
                    "n": int(data.get("n", data.get("total", 0))),
                    "mtime": os.path.getmtime(full_path),
                })
            except Exception:
                results.append({"file": name, "error": "parse_failed"})

    return {"results": results}


# ---------------------------------------------------------
# 日志监控接口
# ---------------------------------------------------------
@app.get("/api/logs/latest")
def get_latest_log(root_dir: str = "/root/workspace/vq_lord",
                   subdir: str = "logs/apis",
                   tail: int = 80):
    log_dir = os.path.join(root_dir, subdir)
    if not os.path.exists(log_dir):
        return {"log_text": f"日志目录不存在: {log_dir}"}

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        return {"log_text": "当前没有找到任何日志文件。"}

    latest_file = max(log_files, key=os.path.getmtime)

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tail_lines = lines[-tail:]
        return {
            "log_file": os.path.basename(latest_file),
            "log_text": "".join(tail_lines),
        }
    except Exception as e:
        return {"log_text": f"读取日志出错: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    # host设为0.0.0.0以便外部/前端可以访问，端口默认8000
    uvicorn.run(app, host="0.0.0.0", port=8000)