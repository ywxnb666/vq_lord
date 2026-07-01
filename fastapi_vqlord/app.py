import time
from typing import Any, Dict, List

import requests
import streamlit as st


st.set_page_config(
    page_title="VQ-LoRD 实验平台",
    layout="wide",
    initial_sidebar_state="expanded",
)


PIPELINE = [
    {
        "key": "teacher_collect",
        "title": "教师四字段数据采集",
        "script": "scripts2/teacher_model_data_collect.sh",
        "desc": "调用教师 API，为训练集样本采集 observed_facts / context / reasoning / answer 四字段缓存。",
    },
    {
        "key": "teacher_eval",
        "title": "教师风险基线评测",
        "script": "scripts2/run_full_eval_pipeline_teacher.sh",
        "desc": "对教师 API 运行 special benchmarks 与 ScienceQA controls，得到学生可逼近的上界。",
    },
    {
        "key": "stage1_train",
        "title": "Stage1 蒸馏训练",
        "script": "scripts2/run_stage1.sh",
        "desc": "学生使用教师四字段数据进行第一阶段蒸馏。",
    },
    {
        "key": "stage2_train",
        "title": "Stage2 蒸馏训练",
        "script": "scripts2/run_stage2.sh",
        "desc": "基于 Stage1 学生继续进行偏好/对齐式蒸馏。",
    },
    {
        "key": "student_eval",
        "title": "学生完整评测",
        "script": "scripts2/run_full_eval_pipeline_fast.sh",
        "desc": "对学生运行 accuracy、special benchmarks 与视觉依赖 controls。",
    },
    {
        "key": "reason_judge",
        "title": "思维链质量评估",
        "script": "../reason_judge/run_judge.sh",
        "desc": "调用 reason_judge，对 Stage1/Stage2 输出进行成对 reasoning 质量评估。",
    },
    {
        "key": "risk_report",
        "title": "风险报告汇总",
        "script": "dashboard aggregation",
        "desc": "汇总 acc retention、视觉依赖变化、reason delta，形成防蒸馏能力读数。",
    },
]


def api_url(path: str) -> str:
    return f"{st.session_state.base_url}{path}"


def safe_get(path: str, **kwargs) -> Dict[str, Any]:
    try:
        resp = requests.get(api_url(path), timeout=8, **kwargs)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def safe_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = requests.post(api_url(path), json=payload, timeout=8)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def metric_value(data: Dict[str, Any], *keys: str, default: Any = "-") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def compact_path(path: str, root: str) -> str:
    if not path:
        return "-"
    return path.replace(root.rstrip("/") + "/", "")


def build_payload() -> Dict[str, Any]:
    return {
        "ROOT_DIR": st.session_state.root_dir,
        "VLA_MARK_DIR": st.session_state.vla_mark_dir,
        "PYTHON_BIN": st.session_state.python_bin,
        "MODEL_PATH": st.session_state.model_path,
        "DATASET_NAME": st.session_state.dataset_name,
        "DATASET_PATH": st.session_state.dataset_path,
        "SCIENCEQA_PATH": st.session_state.dataset_path,
        "CUDA_VISIBLE_DEVICES": st.session_state.cuda_devices,
        "TEACHER_API_KEY_SET": bool(st.session_state.teacher_api_key),
        "TEACHER_API_BASE": st.session_state.teacher_api_base,
        "VICTIM_MODEL": st.session_state.victim_model,
        "STAGE1_CKPT_PATH": st.session_state.stage1_ckpt,
        "STAGE2_FINAL_ADAPTER_PATH": st.session_state.stage2_adapter,
        "RESULT_DIR": st.session_state.result_dir,
        "REASON_JUDGE_DIR": st.session_state.reason_judge_dir,
        "JUDGE_MODEL": st.session_state.judge_model,
        "JUDGE_SAMPLE_NUM": st.session_state.judge_sample_num,
        "SIM_DURATION_SEC": st.session_state.sim_duration,
    }


def submit_step(step_key: str, extra_payload: Dict[str, Any] | None = None) -> None:
    payload = build_payload()
    if extra_payload:
        payload.update(extra_payload)
    result = safe_post(f"/api/task/{step_key}", payload)
    if result.get("error"):
        st.error(result["error"])
    else:
        st.success("✅ 任务已提交到仿真后端")


def submit_full_pipeline() -> None:
    payload = build_payload()
    result = safe_post(
        "/api/task/full_pipeline",
        {
            **payload,
            "PIPELINE_STEP": "full_pipeline",
            "PIPELINE_SEQUENCE": [step["key"] for step in PIPELINE],
        },
    )
    if result.get("error"):
        st.error(result["error"])
    else:
        st.success("✅ 完整 pipeline 已提交为一个串行仿真任务")


def render_step_launcher(step: Dict[str, str], expanded: bool = False) -> None:
    with st.expander(step["title"], expanded=expanded):
        st.info(step["desc"])
        st.code(step["script"], language="bash")
        if st.button(f"🚀 启动：{step['title']}", key=f"btn_{step['key']}", use_container_width=True):
            submit_step(step["key"], {"PIPELINE_STEP": step["key"], "SCRIPT": step["script"]})


def render_watermark_result(result: Dict[str, Any]) -> None:
    if result.get("error"):
        st.error(result["error"])
        return
    if not result.get("found"):
        st.info(result.get("message", "未发现可读取的 VLA-Mark 检测结果。"))
        return

    st.caption(result.get("path", ""))
    summary_data = result.get("summary", {})
    if result.get("kind") == "paired_generation_evaluation":
        cols = st.columns(6)
        labels = [
            ("VLA AUC", "vla_roc_auc"),
            ("VLA F1", "vla_f1"),
            ("VLA ACC", "vla_accuracy"),
            ("Random AUC", "random_roc_auc"),
            ("Random F1", "random_f1"),
            ("Random ACC", "random_accuracy"),
        ]
        for col, (label, key) in zip(cols, labels):
            with col:
                st.metric(label, fmt_num(summary_data.get(key), 3))
    else:
        cols = st.columns(6)
        labels = [
            ("Scored N", "num_scored"),
            ("Mean z", "mean_z"),
            ("Median z", "median_z"),
            ("Min z", "min_z"),
            ("Max z", "max_z"),
            ("z>4 Rate", "threshold_4_rate"),
        ]
        for col, (label, key) in zip(cols, labels):
            with col:
                value = fmt_pct(summary_data.get(key)) if key == "threshold_4_rate" else summary_data.get(key)
                st.metric(label, value if key == "num_scored" else fmt_num(value, 3) if key != "threshold_4_rate" else value)

    records = result.get("records", [])
    if records:
        st.dataframe(records, use_container_width=True, hide_index=True)
    with st.expander("水印结果元信息", expanded=False):
        st.json(
            {
                "kind": result.get("kind"),
                "file": result.get("file"),
                "config": result.get("config", {}),
                "raw_metrics": result.get("raw_metrics", {}),
                "num_records_loaded": result.get("num_records", 0),
            }
        )


if "base_url" not in st.session_state:
    st.session_state.base_url = "http://127.0.0.1:8011"
    st.session_state.root_dir = "/home/ywx/Desktop/vq_lord_parallel/vq_lord"
    st.session_state.reason_judge_dir = "/home/ywx/Desktop/vq_lord_parallel/reason_judge"
    st.session_state.vla_mark_dir = "/home/ywx/Desktop/vq_lord_parallel/VLA-mark"
    st.session_state.python_bin = "/home/ywx/anaconda3/envs/align/bin/python"
    st.session_state.model_path = "/root/autodl-tmp/models/llama3-llava-next-8b-hf"
    st.session_state.watermark_model_path = "/mnt/shared-storage-gpfs2/evoagi-share-gpfs2/xhsong/models/Qwen3-VL-32B-Instruct"
    st.session_state.dataset_name = "scienceqa"
    st.session_state.dataset_path = "/home/ywx/Desktop/vq_lord_parallel/reason_judge/datasets/ScienceQA"
    st.session_state.cuda_devices = "0"
    st.session_state.teacher_api_key = ""
    st.session_state.teacher_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    st.session_state.victim_model = "qwen3.5-flash-2026-02-23"
    st.session_state.stage1_ckpt = "/home/ywx/Desktop/vq_lord_parallel/vq_lord/vq_lord_ckpts/stage1/stage1_vision_epoch1"
    st.session_state.stage2_adapter = "/home/ywx/Desktop/vq_lord_parallel/vq_lord/vq_lord_ckpts/stage2/stage2_sub1_period7"
    st.session_state.result_dir = "/home/ywx/Desktop/vq_lord_parallel/vq_lord/vq_lord_test_results"
    st.session_state.watermark_result_path = "/home/ywx/Desktop/vq_lord_parallel/VLA-mark/outputs/scienceqa_qwen3vl_1000_delta3_skipbad/watermark_detect_200.json"
    st.session_state.judge_model = "gpt-5.5"
    st.session_state.judge_sample_num = 500
    st.session_state.sim_duration = 18


st.title("VQ-LoRD 多模态蒸馏实验平台")
st.caption("当前后端为仿真模式：前端按照 2-stage 模型窃取风险评测 pipeline 组织，但不会在笔记本上真实启动训练脚本。")


with st.sidebar:
    st.header("🔗 连接设置")
    server_ip = st.text_input("服务器IP地址", value="127.0.0.1")
    server_port = st.text_input("服务器端口", value="8011")
    st.session_state.base_url = f"http://{server_ip}:{server_port}"

    if st.button("检查连接状态"):
        status = safe_get("/")
        if status.get("error"):
            st.error(f"❌ 连接失败: {status['error']}")
        else:
            st.success("✅ 连接成功")

    st.divider()
    st.header("📁 全局路径")
    st.session_state.root_dir = st.text_input("ROOT_DIR", value=st.session_state.root_dir)
    st.session_state.reason_judge_dir = st.text_input("REASON_JUDGE_DIR", value=st.session_state.reason_judge_dir)
    st.session_state.vla_mark_dir = st.text_input("VLA_MARK_DIR", value=st.session_state.vla_mark_dir)
    st.session_state.python_bin = st.text_input("PYTHON_BIN", value=st.session_state.python_bin)
    st.session_state.model_path = st.text_input("MODEL_PATH (学生模型)", value=st.session_state.model_path)
    st.session_state.dataset_name = st.selectbox("DATASET_NAME", ["scienceqa", "aokvqa", "iconqa"], index=["scienceqa", "aokvqa", "iconqa"].index(st.session_state.dataset_name))
    st.session_state.dataset_path = st.text_input("DATASET_PATH", value=st.session_state.dataset_path)
    st.session_state.cuda_devices = st.text_input("CUDA_VISIBLE_DEVICES", value=st.session_state.cuda_devices)

    st.divider()
    st.header("🤖 教师模型连接")
    st.session_state.teacher_api_key = st.text_input("TEACHER_API_KEY", type="password", value=st.session_state.teacher_api_key)
    st.session_state.teacher_api_base = st.text_input("TEACHER_API_BASE", value=st.session_state.teacher_api_base)
    st.session_state.victim_model = st.text_input("VICTIM_MODEL", value=st.session_state.victim_model)

    st.divider()
    st.header("🧪 仿真控制")
    st.session_state.sim_duration = st.slider("单任务仿真秒数", 3, 60, int(st.session_state.sim_duration))
    auto_refresh = st.checkbox("任务运行时自动刷新", value=True)


summary = safe_get(
    "/api/results/summary",
    params={"root_dir": st.session_state.root_dir, "reason_judge_dir": st.session_state.reason_judge_dir},
)
derived = summary.get("derived", {}) if not summary.get("error") else {}

metric_cols = st.columns(5)
with metric_cols[0]:
    st.metric("防蒸馏能力", fmt_num(derived.get("defense_score"), 1))
with metric_cols[1]:
    st.metric("窃取风险", fmt_num(derived.get("theft_risk"), 1))
with metric_cols[2]:
    st.metric("Teacher Acc", fmt_pct(derived.get("teacher_acc")))
with metric_cols[3]:
    st.metric("Student Acc", fmt_pct(derived.get("student_acc")))
with metric_cols[4]:
    st.metric("Acc Retention", fmt_pct(derived.get("acc_retention")))

if summary.get("error"):
    st.warning(f"后端暂不可用：{summary['error']}")
else:
    st.info("Pipeline 已切换为：教师 API 数据 → Stage1/Stage2 学生蒸馏 → acc/视觉依赖/思维链评估 → 风险分数。")

run_all_col, refresh_col = st.columns([3, 1])
with run_all_col:
    if st.button("🚀 一键跑完整 Pipeline（仿真）", use_container_width=True, key="btn_run_full_pipeline_top"):
        submit_full_pipeline()
with refresh_col:
    if st.button("刷新大盘", use_container_width=True, key="btn_refresh_top"):
        st.rerun()


tab_info, tab_watermark, tab_data, tab_train, tab_eval, tab_risk = st.tabs([
    "任务状态和日志信息",
    "水印检测",
    "1. 数据与教师基线",
    "2. Stage1/Stage2 蒸馏",
    "3. 完整评测",
    "4. 风险大盘",
])


with tab_watermark:
    st.header("水印检测")
    st.info("当前水印检测任务仍为仿真提交，不会在笔记本上真实加载 VLA-Mark 模型；已有检测结果 JSON 会被真实读取并展示。")

    st.subheader("已有检测结果")
    rcol1, rcol2 = st.columns([4, 1])
    with rcol1:
        st.session_state.watermark_result_path = st.text_input(
            "VLA-Mark 结果 JSON",
            value=st.session_state.watermark_result_path,
            help="可读取 watermark_detect_*.json，或 main.py 生成的 *_vlamark_result_*.json。",
        )
    with rcol2:
        watermark_max_records = st.number_input("显示记录数", 10, 1000, 200, key="wm_result_max_records")

    if st.button("读取水印检测结果", use_container_width=True, key="btn_read_watermark_result"):
        st.session_state.last_watermark_refresh = time.time()

    watermark_result = safe_get(
        "/api/watermark/result",
        params={
            "path": st.session_state.watermark_result_path,
            "vla_mark_dir": st.session_state.vla_mark_dir,
            "max_records": int(watermark_max_records),
        },
    )
    render_watermark_result(watermark_result)

    st.divider()
    st.subheader("检测 VQ-LoRD 输出水印（仿真）")
    st.caption("对应 `VLA-mark/detect_vq_lord_result_watermark.py`，用于对学生/评测结果 JSON 中的模型输出计算水印 z-score。")
    vq1, vq2 = st.columns(2)
    with vq1:
        wm_vqlord_result_path = st.text_input("RESULT_PATH", value=f"{st.session_state.result_dir}/stage2_readable_eval.json", key="wm_vqlord_result_path")
        wm_scienceqa_path = st.text_input("SCIENCEQA_PATH", value=st.session_state.dataset_path, key="wm_scienceqa_path")
        wm_vqlord_output_path = st.text_input(
            "OUTPUT_PATH",
            value=f"{st.session_state.vla_mark_dir}/outputs/vq_lord_watermark_detect.json",
            key="wm_vqlord_output_path",
        )
    with vq2:
        st.session_state.watermark_model_path = st.text_input("WATERMARK_MODEL_PATH", value=st.session_state.watermark_model_path, key="wm_model_path_vqlord")
        wm_model_name = st.selectbox("MODEL_NAME", ["qwen3-vl", "qwen2-vl", "llava-next", "llava"], index=0, key="wm_vqlord_model_name")
        wm_split = st.selectbox("SPLIT", ["test", "validation", "train"], index=0, key="wm_vqlord_split")

    vq3, vq4, vq5, vq6 = st.columns(4)
    with vq3:
        wm_vqlord_sample_size = st.number_input("SAMPLE_SIZE (0=按比例)", 0, 100000, 200, key="wm_vqlord_sample_size")
    with vq4:
        wm_vqlord_sample_fraction = st.slider("SAMPLE_FRACTION", 0.01, 1.0, 0.2, key="wm_vqlord_sample_fraction")
    with vq5:
        wm_vqlord_delta = st.slider("DELTA", 0.0, 8.0, 3.0, key="wm_vqlord_delta")
    with vq6:
        wm_vqlord_seed = st.number_input("SEED", 0, 99999999, 20240306, key="wm_vqlord_seed")

    if st.button("启动 VQ-LoRD 输出水印检测（仿真）", use_container_width=True, key="btn_wm_vqlord"):
        submit_step(
            "watermark_detect_vqlord",
            {
                "SCRIPT": "../VLA-mark/detect_vq_lord_result_watermark.py",
                "RESULT_PATH": wm_vqlord_result_path,
                "SCIENCEQA_PATH": wm_scienceqa_path,
                "OUTPUT_PATH": wm_vqlord_output_path,
                "MODEL_PATH": st.session_state.watermark_model_path,
                "MODEL_NAME": wm_model_name,
                "SPLIT": wm_split,
                "SAMPLE_SIZE": wm_vqlord_sample_size,
                "SAMPLE_FRACTION": wm_vqlord_sample_fraction,
                "SEED": wm_vqlord_seed,
                "DELTA": wm_vqlord_delta,
                "FAST_UNIFORM_ENTROPY": 1,
            },
        )

    st.divider()
    st.subheader("检测教师缓存水印（仿真）")
    st.caption("对应 `VLA-mark/detect_cache_watermark.py`，用于检测教师缓存 `raw_teacher_response` 的水印 z-score。")
    c1, c2 = st.columns(2)
    with c1:
        wm_cache_path = st.text_input(
            "CACHE_PATH",
            value=f"{st.session_state.vla_mark_dir}/outputs/scienceqa_qwen3vl_1000_delta3_skipbad/scienceqa_qwen3vl_vlamark_teacher_cache_merged.json",
            key="wm_cache_path",
        )
        wm_image_dir = st.text_input(
            "IMAGE_DIR",
            value=f"{st.session_state.vla_mark_dir}/outputs/scienceqa_qwen3vl_1000_delta3_skipbad/images",
            key="wm_image_dir",
        )
        wm_cache_output_path = st.text_input(
            "OUTPUT_PATH",
            value=f"{st.session_state.vla_mark_dir}/outputs/cache_watermark_detect.json",
            key="wm_cache_output_path",
        )
    with c2:
        wm_cache_model_path = st.text_input("WATERMARK_MODEL_PATH", value=st.session_state.watermark_model_path, key="wm_model_path_cache")
        wm_cache_model_name = st.selectbox("MODEL_NAME", ["qwen3-vl", "qwen2-vl", "llava-next", "llava"], index=0, key="wm_cache_model_name")
        wm_cache_sample_size = st.number_input("SAMPLE_SIZE", 1, 100000, 200, key="wm_cache_sample_size")

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        wm_cache_delta = st.slider("DELTA", 0.0, 8.0, 3.0, key="wm_cache_delta")
    with c4:
        wm_cache_split_x = st.number_input("SPLIT_X", 2, 16, 2, key="wm_cache_split_x")
    with c5:
        wm_cache_seed = st.number_input("SEED", 0, 99999999, 20240306, key="wm_cache_seed")
    with c6:
        wm_cache_fast_entropy = st.checkbox("FAST_UNIFORM_ENTROPY", value=True, key="wm_cache_fast_entropy")

    if st.button("启动教师缓存水印检测（仿真）", use_container_width=True, key="btn_wm_cache"):
        submit_step(
            "watermark_detect_cache",
            {
                "SCRIPT": "../VLA-mark/detect_cache_watermark.py",
                "CACHE_PATH": wm_cache_path,
                "IMAGE_DIR": wm_image_dir,
                "OUTPUT_PATH": wm_cache_output_path,
                "MODEL_PATH": wm_cache_model_path,
                "MODEL_NAME": wm_cache_model_name,
                "SAMPLE_SIZE": wm_cache_sample_size,
                "SEED": wm_cache_seed,
                "DELTA": wm_cache_delta,
                "SPLIT_X": wm_cache_split_x,
                "FAST_UNIFORM_ENTROPY": int(wm_cache_fast_entropy),
            },
        )


with tab_data:
    st.header("1. 数据与教师基线")

    st.subheader("1A. 教师四字段标注采集")
    st.info("用于构造学生蒸馏数据。后端当前只模拟进度，不会真实调用教师 API。")

    col1, col2 = st.columns(2)
    with col1:
        teacher_split = st.selectbox("SCIENCEQA_SPLIT", ["train", "validation", "test"], index=0, key="teacher_split")
        train_num = st.number_input("TRAIN_NUM (0=全量)", 0, 100000, 0, key="teacher_train_num")
        max_samples = st.number_input("MAX_SAMPLES (0=全量)", 0, 100000, 0, key="teacher_max_samples")
        seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306, key="teacher_seed")
    with col2:
        teacher_lang = st.selectbox("TEACHER_LANG", ["en", "zh"], index=0)
        enable_thinking = st.checkbox("TEACHER_ENABLE_THINKING", value=False)
        strict_distill = st.checkbox("STRICT_TEACHER_DISTILL", value=True)
        workers = st.number_input("NUM_WORKERS", 1, 64, 4)

    if st.button("🚀 开始采集教师数据", use_container_width=True):
        submit_step(
            "teacher_collect",
            {
                "SCIENCEQA_SPLIT": teacher_split,
                "TRAIN_NUM": train_num,
                "MAX_SAMPLES": max_samples,
                "SCIENCEQA_SEED": seed,
                "TEACHER_LANG": teacher_lang,
                "TEACHER_ENABLE_THINKING": int(enable_thinking),
                "STRICT_TEACHER_DISTILL": int(strict_distill),
                "NUM_WORKERS": workers,
            },
        )

    st.divider()
    st.subheader("1B. 教师风险基线评测")
    st.info("对应 `run_full_eval_pipeline_teacher.sh`：special benchmarks + ScienceQA controls + report aggregation。")

    c1, c2 = st.columns(2)
    with c1:
        teacher_max_new_tokens = st.number_input("MAX_NEW_TOKENS", 16, 2048, 64, key="teacher_eval_tokens")
        teacher_max_concurrency = st.number_input("MAX_CONCURRENCY", 1, 64, 4, key="teacher_concurrency")
    with c2:
        parallel_controls = st.checkbox("PARALLEL_CONTROLS", value=True)
        teacher_result_dir = st.text_input(
            "TEACHER_RESULT_DIR",
            value=f"{st.session_state.result_dir}/teacher_compare",
        )

    if st.button("🚀 启动教师完整评测", use_container_width=True):
        submit_step(
            "teacher_eval",
            {
                "MAX_NEW_TOKENS": teacher_max_new_tokens,
                "MAX_CONCURRENCY": teacher_max_concurrency,
                "PARALLEL_CONTROLS": int(parallel_controls),
                "RESULT_DIR": teacher_result_dir,
            },
        )


with tab_train:
    st.header("2. Stage1/Stage2 蒸馏")
    st.warning("当前流程按你的新要求忽略 Stage0；若真实脚本仍需要 codebook/path，请在下方路径中显式指定。")

    with st.expander("📂 路径配置", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.stage1_ckpt = st.text_input("STAGE1_CKPT_PATH", value=st.session_state.stage1_ckpt)
        with c2:
            st.session_state.stage2_adapter = st.text_input("STAGE2_FINAL_ADAPTER_PATH", value=st.session_state.stage2_adapter)

    st.subheader("Stage1 训练")
    s1c1, s1c2, s1c3 = st.columns(3)
    with s1c1:
        s1_epochs = st.number_input("STAGE1_EPOCHS", 1, 200, 3)
        s1_batch = st.number_input("STAGE1_BATCH_SIZE", 1, 128, 8)
    with s1c2:
        s1_lr = st.text_input("STAGE1_LR", "3e-5")
        s1_max_length = st.number_input("STAGE1_MAX_LENGTH", 128, 4096, 1536)
    with s1c3:
        s1_reason_weight = st.slider("STAGE1_FIELD_WEIGHT_REASONING", 0.0, 20.0, 2.0)
        s1_answer_weight = st.slider("STAGE1_FIELD_WEIGHT_ANSWER", 0.0, 20.0, 12.0)

    if st.button("🚀 启动 Stage1 蒸馏", use_container_width=True):
        submit_step(
            "stage1_train",
            {
                "EPOCHS": s1_epochs,
                "BATCH_SIZE": s1_batch,
                "LR": s1_lr,
                "MAX_LENGTH": s1_max_length,
                "STAGE1_FIELD_WEIGHT_REASONING": s1_reason_weight,
                "STAGE1_FIELD_WEIGHT_ANSWER": s1_answer_weight,
            },
        )

    st.divider()
    st.subheader("Stage2 训练")
    s2c1, s2c2, s2c3 = st.columns(3)
    with s2c1:
        s2_epochs = st.number_input("STAGE2_EPOCHS", 1, 200, 50)
        period_num = st.number_input("PERIOD_NUM", 1, 200, 50)
    with s2c2:
        s2_lr = st.text_input("STAGE2_LR", "2e-5")
        tau1 = st.text_input("TAU1", "0.02")
    with s2c3:
        wrong_image = st.checkbox("STAGE2_WRONG_IMAGE_ENABLE", value=True)
        pair_correctness = st.checkbox("STAGE2_PAIR_USE_ANSWER_CORRECTNESS", value=False)

    if st.button("🔥 启动 Stage2 蒸馏", use_container_width=True):
        submit_step(
            "stage2_train",
            {
                "EPOCHS": s2_epochs,
                "PERIOD_NUM": period_num,
                "LR": s2_lr,
                "TAU1": tau1,
                "STAGE2_WRONG_IMAGE_ENABLE": int(wrong_image),
                "STAGE2_PAIR_USE_ANSWER_CORRECTNESS": int(pair_correctness),
            },
        )


with tab_eval:
    st.header("3. 完整评测")

    st.subheader("3A. 学生完整风险评测")
    st.info("对应 `run_full_eval_pipeline_fast.sh`：special benchmarks + ScienceQA controls + report aggregation。")
    ec1, ec2 = st.columns(2)
    with ec1:
        adapter_path = st.text_input("ADAPTER_PATH", value=st.session_state.stage2_adapter)
        vq_codebook_path = st.text_input("VQ_CODEBOOK_PATH", value=f"{st.session_state.stage2_adapter}/vq_codebook.pt")
    with ec2:
        eval_result_dir = st.text_input("RESULT_DIR", value=st.session_state.result_dir)
        eval_max_new_tokens = st.number_input("EVAL_MAX_NEW_TOKENS", 16, 2048, 512)

    if st.button("🚀 启动学生完整评测", use_container_width=True):
        submit_step(
            "student_eval",
            {
                "ADAPTER_PATH": adapter_path,
                "VQ_CODEBOOK_PATH": vq_codebook_path,
                "RESULT_DIR": eval_result_dir,
                "MAX_NEW_TOKENS": eval_max_new_tokens,
            },
        )

    st.divider()
    st.subheader("3B. 思维链评估")
    st.info("使用 `/home/ywx/Desktop/vq_lord_parallel/reason_judge/run_judge.sh`。脚本内部的 Stage2/Stage3 命名在这里映射为 Stage1/Stage2 学生输出。")
    jc1, jc2 = st.columns(2)
    with jc1:
        stage1_json = st.text_input("Stage1 readable eval JSON", "")
        stage2_json = st.text_input("Stage2 readable eval JSON", "")
        teacher_json = st.text_input("Teacher JSON (可选)", "")
    with jc2:
        st.session_state.judge_model = st.text_input("JUDGE_MODEL", value=st.session_state.judge_model)
        st.session_state.judge_sample_num = st.number_input("SAMPLE_NUM", 1, 10000, int(st.session_state.judge_sample_num))
        require_valid = st.checkbox("REQUIRE_VALID_FORMAT", value=True)

    if st.button("🚀 启动思维链评估", use_container_width=True):
        submit_step(
            "reason_judge",
            {
                "STAGE2": stage1_json,
                "STAGE3": stage2_json,
                "TEACHER": teacher_json,
                "SAMPLE_NUM": st.session_state.judge_sample_num,
                "REQUIRE_VALID_FORMAT": int(require_valid),
            },
        )


with tab_risk:
    st.header("4. 风险大盘")

    if st.button("刷新结果摘要", key="refresh_summary"):
        st.rerun()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Stage1 Acc", fmt_pct(metric_value(summary, "stage1", "accuracy", default=0)))
    with r2:
        st.metric("Stage2 Acc", fmt_pct(metric_value(summary, "stage2", "accuracy", default=0)))
    with r3:
        st.metric("视觉依赖 Delta", fmt_num(derived.get("visual_dependency_delta"), 3))
    with r4:
        st.metric("Reason Delta", fmt_num(derived.get("reason_delta"), 3))

    st.divider()
    st.subheader("Teacher / Student Controls")
    controls = metric_value(summary, "student", "controls", default=[]) or metric_value(summary, "teacher", "controls", default=[])
    if controls:
        st.dataframe(controls, use_container_width=True, hide_index=True)
    else:
        st.info("未发现 full eval control report。真实脚本完成后应生成 `mm_eval_suite_report_*.json`。")

    st.subheader("Reason Judge")
    reason = summary.get("reason", {}) if isinstance(summary, dict) else {}
    if reason.get("found"):
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            st.metric("Judged N", reason.get("n", 0))
        with rc2:
            st.metric("Stage1 Reason", fmt_num(reason.get("stage1_reason_score"), 3))
        with rc3:
            st.metric("Stage2 Reason", fmt_num(reason.get("stage2_reason_score"), 3))
        with rc4:
            st.metric("Delta", fmt_num(reason.get("delta_reason_score"), 3))
        with rc5:
            st.metric("Stage2 Win", fmt_pct(reason.get("stage2_win_rate")))
        st.caption(compact_path(reason.get("path", ""), st.session_state.reason_judge_dir))
    else:
        st.info("未发现 `reason_judge/outputs/*/summary.tsv`。")

    with st.expander("原始 summary JSON", expanded=False):
        st.json(summary)


with tab_info:
    st.subheader("📋 任务运行状态")
    if st.button("刷新任务状态", key="btn_task_status"):
        st.session_state.last_task_refresh = time.time()

    tasks_payload = safe_get("/api/tasks")
    tasks = tasks_payload.get("tasks", []) if not tasks_payload.get("error") else []
    if tasks_payload.get("error"):
        st.error(tasks_payload["error"])
    elif not tasks:
        st.info("暂无任务记录")
    else:
        running_tasks = [task for task in tasks if task.get("status") == "running"]
        current_task = running_tasks[0] if running_tasks else tasks[0]

        status = current_task.get("status", "unknown")
        icon = {"running": "🔄", "success": "✅", "failed": "❌"}.get(status, "❓")
        progress = float(current_task.get("progress", 0.0))

        c_task, c_status, c_progress = st.columns([3, 1, 1])
        with c_task:
            st.write(f"**当前任务**：{current_task.get('label', current_task.get('step'))}")
            st.caption(current_task.get("script", ""))
        with c_status:
            st.write("**状态**")
            st.write(f"{icon} {status}")
        with c_progress:
            st.write("**进度**")
            st.write(f"{progress * 100:.0f}%")

        st.progress(progress)

        sub_tasks = current_task.get("sub_tasks") or []
        if sub_tasks:
            st.write("**串行阶段明细**")
            stage_rows = []
            for sub_task in sub_tasks:
                stage_rows.append(
                    {
                        "阶段": sub_task.get("label"),
                        "状态": sub_task.get("status"),
                        "进度": f"{float(sub_task.get('progress', 0.0)) * 100:.0f}%",
                        "脚本": sub_task.get("script"),
                    }
                )
            st.dataframe(stage_rows, use_container_width=True, hide_index=True)

        with st.expander("历史任务", expanded=False):
            current_task_id = current_task.get("task_id")
            history_tasks = [task for task in tasks if task.get("task_id") != current_task_id]
            history_rows = []
            for history_task in history_tasks:
                history_rows.append(
                    {
                        "任务": history_task.get("label", history_task.get("step")),
                        "状态": history_task.get("status"),
                        "进度": f"{float(history_task.get('progress', 0.0)) * 100:.0f}%",
                        "脚本": history_task.get("script", ""),
                    }
                )
            if history_rows:
                st.dataframe(history_rows, use_container_width=True, hide_index=True)
            else:
                st.info("暂无历史任务")

    st.divider()
    st.subheader("📜 系统后台日志")
    if st.button("刷新最新日志", key="btn_log"):
        st.session_state.last_log_refresh = time.time()

    logs = safe_get("/api/logs/latest")
    if logs.get("error"):
        st.error(logs["error"])
    else:
        st.caption(f"日志文件: {logs.get('log_file', '')}")
        st.code(logs.get("log_text", "No logs yet."), language="bash")

    st.divider()
    st.subheader("🧾 稳定脚本映射")
    st.dataframe(
        [{"pipeline_step": step["key"], "script": step["script"], "purpose": step["desc"]} for step in PIPELINE],
        use_container_width=True,
        hide_index=True,
    )


if auto_refresh:
    task_payload = safe_get("/api/tasks")
    live_tasks = task_payload.get("tasks", []) if not task_payload.get("error") else []
    if any(task.get("status") == "running" for task in live_tasks):
        time.sleep(2)
        st.rerun()
