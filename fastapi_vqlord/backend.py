import csv
import glob
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Query


app = FastAPI(title="VQ-LoRD Risk Evaluation Console")


ROOT_DEFAULT = "/home/ywx/Desktop/vq_lord_parallel/vq_lord"
REASON_JUDGE_DEFAULT = "/home/ywx/Desktop/vq_lord_parallel/reason_judge"
VLA_MARK_DEFAULT = "/home/ywx/Desktop/vq_lord_parallel/VLA-mark"


PIPELINE_STEPS: Dict[str, Dict[str, Any]] = {
    "full_pipeline": {
        "label": "Full sequential risk-evaluation pipeline",
        "script": "teacher_collect -> teacher_eval -> stage1_train -> stage2_train -> student_eval -> reason_judge -> risk_report",
        "seconds": 126,
        "outputs": ["sequential simulated pipeline"],
    },
    "teacher_collect": {
        "label": "Teacher API data collection",
        "script": "scripts2/teacher_model_data_collect.sh",
        "seconds": 18,
        "outputs": ["vq_lord_data/*teacher*.json"],
    },
    "teacher_eval": {
        "label": "Teacher full risk baseline",
        "script": "scripts2/run_full_eval_pipeline_teacher.sh",
        "seconds": 22,
        "outputs": [
            "vq_lord_test_results/**/mm_eval_suite_report_teacher_full.json",
            "vq_lord_test_results/**/scienceqa_control_suite_teacher_full.json",
        ],
    },
    "stage1_train": {
        "label": "Stage1 student distillation",
        "script": "scripts2/run_stage1.sh",
        "seconds": 26,
        "outputs": ["vq_lord_ckpts/stage1/**"],
    },
    "stage2_train": {
        "label": "Stage2 student distillation",
        "script": "scripts2/run_stage2.sh",
        "seconds": 30,
        "outputs": ["vq_lord_ckpts/stage2/**"],
    },
    "student_eval": {
        "label": "Student full risk evaluation",
        "script": "scripts2/run_full_eval_pipeline_fast.sh",
        "seconds": 24,
        "outputs": [
            "vq_lord_test_results/mm_eval_suite_report_full_fast.json",
            "vq_lord_test_results/scienceqa_control_suite_full_fast.json",
        ],
    },
    "reason_judge": {
        "label": "Reasoning judge",
        "script": "../reason_judge/run_judge.sh",
        "seconds": 20,
        "outputs": ["../reason_judge/outputs/*/summary.tsv"],
    },
    "risk_report": {
        "label": "Risk report aggregation",
        "script": "dashboard aggregation",
        "seconds": 12,
        "outputs": ["dashboard synthetic summary"],
    },
    "watermark_detect_vqlord": {
        "label": "VLA-Mark detection on VQ-LoRD outputs",
        "script": "../VLA-mark/detect_vq_lord_result_watermark.py",
        "seconds": 18,
        "outputs": ["../VLA-mark/outputs/**/watermark_detect_*.json"],
    },
    "watermark_detect_cache": {
        "label": "VLA-Mark detection on teacher cache",
        "script": "../VLA-mark/detect_cache_watermark.py",
        "seconds": 14,
        "outputs": ["../VLA-mark/outputs/**/watermark_detect_*.json"],
    },
}

PIPELINE_SEQUENCE = [
    "teacher_collect",
    "teacher_eval",
    "stage1_train",
    "stage2_train",
    "student_eval",
    "reason_judge",
    "risk_report",
]


@dataclass
class SimTask:
    task_id: str
    step: str
    label: str
    script: str
    payload: Dict[str, Any]
    started_at: float
    duration_sec: float
    status: str = "running"
    progress: float = 0.0
    log: List[str] = field(default_factory=list)
    ended_at: Optional[float] = None


class TaskStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tasks: Dict[str, SimTask] = {}

    def start(self, step: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = PIPELINE_STEPS[step]
        now = time.time()
        task_id = f"{step}_{int(now)}"
        if step == "full_pipeline":
            per_step_duration = float(payload.get("SIM_DURATION_SEC") or 8)
            duration = per_step_duration * len(PIPELINE_SEQUENCE)
        else:
            duration = float(payload.get("SIM_DURATION_SEC") or spec["seconds"])
        task = SimTask(
            task_id=task_id,
            step=step,
            label=spec["label"],
            script=spec["script"],
            payload=payload,
            started_at=now,
            duration_sec=max(3.0, duration),
            log=[
                f"[simulate] queued {spec['label']}",
                f"[simulate] script: {spec['script']}",
                "[simulate] no training/evaluation process is launched on this laptop backend",
            ],
        )
        with self.lock:
            self.tasks[task_id] = task
        return serialize_task(task)

    def update_all(self) -> None:
        now = time.time()
        with self.lock:
            for task in self.tasks.values():
                if task.status != "running":
                    continue
                elapsed = now - task.started_at
                pct = min(1.0, elapsed / task.duration_sec)
                task.progress = pct
                task.log = self._build_log(task, pct)
                if pct >= 1.0:
                    task.status = "success"
                    task.ended_at = now
                    task.log.append("[simulate] completed successfully")

    @staticmethod
    def _build_log(task: SimTask, pct: float) -> List[str]:
        if task.step == "full_pipeline":
            sub_tasks = build_sub_tasks(task)
            lines = [
                f"[simulate] task={task.task_id}",
                "[simulate] mode=sequential full pipeline",
                f"[simulate] progress={int(pct * 100)}%",
            ]
            for item in sub_tasks:
                lines.append(
                    f"[simulate] {item['status']:>7} {item['step']} "
                    f"{int(item['progress'] * 100)}% :: {item['script']}"
                )
            return lines

        milestones = [
            (0.05, "validated paths and environment payload"),
            (0.18, "resolved teacher/model/checkpoint inputs"),
            (0.35, "prepared dataset and cached artifacts"),
            (0.52, "running scoring/training placeholder loop"),
            (0.72, "materializing expected report locations"),
            (0.90, "aggregating dashboard metrics"),
        ]
        lines = [
            f"[simulate] task={task.task_id}",
            f"[simulate] step={task.step}",
            f"[simulate] script={task.script}",
        ]
        for threshold, message in milestones:
            if pct >= threshold:
                lines.append(f"[simulate] {message}")
        lines.append(f"[simulate] progress={int(pct * 100)}%")
        return lines

    def all(self) -> List[Dict[str, Any]]:
        self.update_all()
        with self.lock:
            return [serialize_task(task) for task in sorted(self.tasks.values(), key=lambda t: t.started_at, reverse=True)]

    def latest_log(self) -> Dict[str, Any]:
        self.update_all()
        with self.lock:
            if not self.tasks:
                return {"log_text": "No simulated tasks yet.", "log_file": "simulation.log"}
            task = max(self.tasks.values(), key=lambda t: t.started_at)
            return {"log_file": f"{task.task_id}.log", "log_text": "\n".join(task.log)}


def serialize_task(task: SimTask) -> Dict[str, Any]:
    payload = {
        "task_id": task.task_id,
        "step": task.step,
        "label": task.label,
        "script": task.script,
        "status": task.status,
        "progress": round(task.progress, 4),
        "started_at": task.started_at,
        "ended_at": task.ended_at,
        "duration_sec": task.duration_sec,
        "payload": task.payload,
        "log": task.log,
    }
    if task.step == "full_pipeline":
        payload["sub_tasks"] = build_sub_tasks(task)
    return payload


def build_sub_tasks(task: SimTask) -> List[Dict[str, Any]]:
    elapsed = max(0.0, (task.ended_at or time.time()) - task.started_at)
    per_step = max(0.001, task.duration_sec / len(PIPELINE_SEQUENCE))
    rows: List[Dict[str, Any]] = []
    for idx, step_key in enumerate(PIPELINE_SEQUENCE):
        spec = PIPELINE_STEPS[step_key]
        local_elapsed = elapsed - (idx * per_step)
        if local_elapsed <= 0:
            status = "pending"
            progress = 0.0
        elif local_elapsed >= per_step:
            status = "success"
            progress = 1.0
        else:
            status = "running"
            progress = local_elapsed / per_step
        rows.append(
            {
                "step": step_key,
                "label": spec["label"],
                "script": spec["script"],
                "status": status,
                "progress": round(progress, 4),
            }
        )
    return rows


task_store = TaskStore()


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def latest_file(patterns: List[str], root_dir: str) -> Optional[str]:
    matches: List[str] = []
    for pattern in patterns:
        full_pattern = pattern if os.path.isabs(pattern) else os.path.join(root_dir, pattern)
        matches.extend(glob.glob(full_pattern, recursive=True))
    files = [path for path in matches if os.path.isfile(path)]
    return max(files, key=os.path.getmtime) if files else None


def metric_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_eval_report(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"found": False}
    payload = read_json(path)
    if not payload:
        return {"found": False, "path": path}

    benchmark = payload.get("benchmark_summary", {})
    control = payload.get("control_summary", {})
    control_summary = control.get("control_summary", {}) if isinstance(control, dict) else {}
    baseline = control.get("baseline_accuracy") if isinstance(control, dict) else None

    controls = []
    for name, row in control_summary.items():
        controls.append(
            {
                "control": name,
                "accuracy": metric_number(row.get("accuracy")),
                "delta_vs_baseline": row.get("delta_vs_baseline"),
            }
        )

    return {
        "found": True,
        "path": path,
        "file": os.path.basename(path),
        "overall_accuracy": metric_number(benchmark.get("overall_accuracy")),
        "overall_correct": int(metric_number(benchmark.get("overall_correct"), 0)),
        "overall_total": int(metric_number(benchmark.get("overall_total"), 0)),
        "baseline_accuracy": metric_number(baseline),
        "controls": controls,
        "raw": payload,
    }


def parse_legacy_result(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"found": False}
    payload = read_json(path)
    if not payload:
        return {"found": False, "path": path}
    metrics = payload.get("metrics", payload)
    return {
        "found": True,
        "path": path,
        "file": os.path.basename(path),
        "accuracy": metric_number(metrics.get("accuracy", metrics.get("acc"))),
        "total": int(metric_number(metrics.get("total", metrics.get("n", 0)), 0)),
        "correct": int(metric_number(metrics.get("correct", 0), 0)),
        "format_rate": metric_number(metrics.get("format_rate", metrics.get("format_valid_rate"))),
    }


def parse_reason_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"found": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            return {"found": False, "path": path}
        row = rows[0]
        return {
            "found": True,
            "path": path,
            "file": os.path.basename(path),
            "n": int(metric_number(row.get("n"), 0)),
            "stage1_reason_score": metric_number(row.get("stage2_reason_score")),
            "stage2_reason_score": metric_number(row.get("stage3_reason_score")),
            "delta_reason_score": metric_number(row.get("delta_reason_score")),
            "stage1_win_rate": metric_number(row.get("stage2_win_rate")),
            "stage2_win_rate": metric_number(row.get("stage3_win_rate")),
            "tie_rate": metric_number(row.get("tie_rate")),
            "dimensions": row,
        }
    except Exception as exc:
        return {"found": False, "path": path, "message": str(exc)}


def parse_watermark_result(path: Optional[str], max_records: int = 200) -> Dict[str, Any]:
    if not path:
        return {"found": False}
    payload = read_json(path)
    if not payload:
        return {"found": False, "path": path, "message": "cannot read JSON object"}

    metrics = payload.get("metrics", {})
    records = payload.get("records", [])
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(records, list):
        records = []

    result: Dict[str, Any] = {
        "found": True,
        "path": path,
        "file": os.path.basename(path),
        "config": payload.get("config", {}),
        "num_records": len(records),
        "records": records[: max(0, int(max_records))],
        "raw_metrics": metrics,
    }

    if "vla_mark" in metrics or "random_only" in metrics:
        vla_mark = metrics.get("vla_mark", {}) if isinstance(metrics.get("vla_mark"), dict) else {}
        random_only = metrics.get("random_only", {}) if isinstance(metrics.get("random_only"), dict) else {}
        result.update(
            {
                "kind": "paired_generation_evaluation",
                "summary": {
                    "vla_roc_auc": metric_number(vla_mark.get("roc_auc")),
                    "vla_f1": metric_number(vla_mark.get("f1")),
                    "vla_accuracy": metric_number(vla_mark.get("accuracy")),
                    "random_roc_auc": metric_number(random_only.get("roc_auc")),
                    "random_f1": metric_number(random_only.get("f1")),
                    "random_accuracy": metric_number(random_only.get("accuracy")),
                },
            }
        )
        return result

    result.update(
        {
            "kind": "output_or_cache_detection",
            "summary": {
                "num_scored": int(metric_number(payload.get("num_scored", len(records)), 0)),
                "mean_z": metric_number(metrics.get("mean_z")),
                "median_z": metric_number(metrics.get("median_z")),
                "min_z": metric_number(metrics.get("min_z")),
                "max_z": metric_number(metrics.get("max_z")),
                "threshold_4_rate": metric_number(metrics.get("threshold_4_rate")),
            },
        }
    )
    return result


def list_checkpoint_dirs(root_dir: str, prefixes: List[str]) -> List[Dict[str, Any]]:
    ckpt_base = os.path.join(root_dir, "vq_lord_ckpts")
    if not os.path.isdir(ckpt_base):
        return []
    out = []
    for name in sorted(os.listdir(ckpt_base)):
        full = os.path.join(ckpt_base, name)
        if not os.path.isdir(full):
            continue
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        out.append(
            {
                "name": name,
                "path": full,
                "has_adapter": os.path.isfile(os.path.join(full, "adapter_config.json")),
                "has_projector": os.path.isfile(os.path.join(full, "projector.pt")),
                "has_codebook": os.path.isfile(os.path.join(full, "vq_codebook.pt")),
                "mtime": os.path.getmtime(full),
            }
        )
    return out


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "VQ-LoRD simulated backend is running"}


@app.get("/api/pipeline/spec")
def pipeline_spec() -> Dict[str, Any]:
    return {"steps": PIPELINE_STEPS, "simulated": True}


@app.post("/api/task/{step}")
def start_pipeline_step(step: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    if step not in PIPELINE_STEPS:
        return {"status": "error", "message": f"unknown pipeline step: {step}"}
    task = task_store.start(step, payload)
    return {"status": "ok", "task": task}


@app.get("/api/tasks")
def get_tasks() -> Dict[str, Any]:
    return {"tasks": task_store.all()}


@app.get("/api/logs/latest")
def get_latest_log() -> Dict[str, Any]:
    return task_store.latest_log()


@app.get("/api/checkpoints")
def checkpoints(
    root_dir: str = ROOT_DEFAULT,
    prefixes: List[str] = Query(default=["stage1", "stage2", "stage2_sub", "stage2_lord"]),
) -> Dict[str, Any]:
    return {"dirs": list_checkpoint_dirs(root_dir, prefixes)}


@app.get("/api/results/summary")
def results_summary(
    root_dir: str = ROOT_DEFAULT,
    reason_judge_dir: str = REASON_JUDGE_DEFAULT,
) -> Dict[str, Any]:
    teacher_report = latest_file(
        [
            "vq_lord_test_results/**/mm_eval_suite_report_teacher_full.json",
            "vq_lord_test_results/mm_eval_suite_report_teacher_full.json",
        ],
        root_dir,
    )
    student_report = latest_file(
        [
            "vq_lord_test_results/mm_eval_suite_report_full_fast.json",
            "vq_lord_test_results/**/mm_eval_suite_report_full_fast.json",
        ],
        root_dir,
    )
    stage1 = latest_file(["vq_lord_test_results/stage1*.json"], root_dir)
    stage2 = latest_file(["vq_lord_test_results/stage2*.json"], root_dir)
    reason = latest_file(["outputs/*/summary.tsv"], reason_judge_dir)

    teacher = parse_eval_report(teacher_report)
    student = parse_eval_report(student_report)
    reason_summary = parse_reason_summary(reason)
    stage1_result = parse_legacy_result(stage1)
    stage2_result = parse_legacy_result(stage2)

    teacher_acc = teacher.get("baseline_accuracy") or teacher.get("overall_accuracy") or 0.0
    student_acc = stage2_result.get("accuracy") or student.get("baseline_accuracy") or student.get("overall_accuracy") or 0.0
    acc_retention = (student_acc / teacher_acc) if teacher_acc else 0.0

    visual_delta = 0.0
    controls = student.get("controls") or teacher.get("controls") or []
    visual_controls = [row for row in controls if row.get("control") in {"text_only_blank", "random_image_swap", "image_blur", "image_downsample"}]
    if visual_controls:
        visual_delta = sum(abs(metric_number(row.get("delta_vs_baseline"))) for row in visual_controls) / len(visual_controls)

    reason_delta = reason_summary.get("delta_reason_score", 0.0) if reason_summary.get("found") else 0.0
    theft_risk = min(100.0, max(0.0, 100.0 * (0.65 * acc_retention + 0.25 * max(0.0, 1.0 - visual_delta) + 0.10 * max(0.0, reason_delta + 1.0) / 2.0)))
    defense_score = round(100.0 - theft_risk, 2)

    return {
        "teacher": teacher,
        "student": student,
        "stage1": stage1_result,
        "stage2": stage2_result,
        "reason": reason_summary,
        "derived": {
            "teacher_acc": round(teacher_acc, 6),
            "student_acc": round(student_acc, 6),
            "acc_retention": round(acc_retention, 6),
            "visual_dependency_delta": round(visual_delta, 6),
            "reason_delta": round(reason_delta, 6),
            "theft_risk": round(theft_risk, 2),
            "defense_score": defense_score,
            "simulated_backend": True,
        },
    }


@app.get("/api/watermark/result")
def watermark_result(
    path: str = "",
    vla_mark_dir: str = VLA_MARK_DEFAULT,
    max_records: int = 200,
) -> Dict[str, Any]:
    result_path = path.strip()
    if not result_path:
        result_path = latest_file(
            [
                "outputs/**/watermark_detect*.json",
                "outputs/**/*vlamark*result*.json",
                "*vlamark*result*.json",
                "local_smoke_result.json",
            ],
            vla_mark_dir,
        ) or ""
    result = parse_watermark_result(result_path, max_records=max_records)
    if not result.get("found") and not path.strip():
        result["message"] = "no VLA-Mark result JSON found"
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8011)
