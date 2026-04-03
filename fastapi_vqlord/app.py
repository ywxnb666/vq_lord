import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(
    page_title="VQ-LoRD 实验平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("VQ-LoRD 多模态蒸馏实验平台")

# ============================================================
# 侧边栏：连接设置 + 全局路径（对应 common.sh）
# ============================================================
with st.sidebar:
    st.header("🔗 连接设置")
    server_ip = st.text_input("服务器IP地址", value="127.0.0.1")
    server_port = st.text_input("服务器端口", value="8000")
    BASE_URL = f"http://{server_ip}:{server_port}"

    if st.button("检查连接状态"):
        try:
            resp = requests.get(f"{BASE_URL}/", timeout=5)
            if resp.status_code == 200:
                st.success("✅ 连接成功")
            else:
                st.error(f"❌ HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"❌ 连接失败: {e}")

    st.divider()
    st.header("📁 全局路径")
    glob_root_dir = st.text_input("ROOT_DIR", value="/root/workspace/vq_lord")
    glob_python_bin = st.text_input("PYTHON_BIN", value="/root/autodl-tmp/conda/envs/align_vq/bin/python")
    glob_model_path = st.text_input("MODEL_PATH (学生模型)", value="/root/autodl-tmp/models/llama3-llava-next-8b-hf")
    glob_scienceqa_path = st.text_input("SCIENCEQA_PATH", value="/root/autodl-tmp/datasets/ScienceQA")
    glob_cuda_devices = st.text_input("CUDA_VISIBLE_DEVICES", value="0")

    st.divider()
    st.header("🤖 教师模型连接")
    glob_teacher_api_key = st.text_input("TEACHER_API_KEY", type="password", value="")
    glob_teacher_api_base = st.text_input("TEACHER_API_BASE", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
    glob_victim_model = st.text_input("VICTIM_MODEL", value="qwen3.5-flash-2026-02-23")

# 把侧边栏全局参数收拢成字典，每次发请求都带上
GLOBAL_ENV = {
    "ROOT_DIR": glob_root_dir,
    "PYTHON_BIN": glob_python_bin,
    "MODEL_PATH": glob_model_path,
    "SCIENCEQA_PATH": glob_scienceqa_path,
    "CUDA_VISIBLE_DEVICES": glob_cuda_devices,
    "TEACHER_API_KEY": glob_teacher_api_key,
    "TEACHER_API_BASE": glob_teacher_api_base,
    "VICTIM_MODEL": glob_victim_model,
}

# ============================================================
# Tabs
# ============================================================
tab_data, tab_s1, tab_s2, tab_s3, tab_eval, tab_info = st.tabs([
    "1. 数据筹备", "2. Stage1(VQ底座)", "3. Stage2(视觉蒸馏)",
    "4. Stage3(偏好对齐)", "5. 评测大盘", "任务状态和日志信息"
])

# ============================================================
# Tab 1: 数据筹备（教师采集 + 分桶预处理）
# ============================================================
with tab_data:
    st.header("1. 数据筹备")

    # ---------- 1A: 教师四字段标注采集 ----------
    st.subheader("1A. 教师四字段标注采集")
    st.info("调用教师模型 API，为 ScienceQA 图文样本生成结构化标注 "
            "(observed_facts / context / reasoning / answer)，缓存为 JSON。")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**数据范围**")
        tc_split = st.selectbox("SCIENCEQA_SPLIT (数据集分割)",
                                ["train", "validation", "test"], index=0,
                                key="tc_split")
        tc_train_num = st.number_input("TRAIN_NUM (样本数, 0=全量)", 0, 100000, 0,
                                       key="tc_train_num")
        tc_max_samples = st.number_input("MAX_SAMPLES (最大采集数, 0=全量)", 0, 100000, 0,
                                         key="tc_max_samples")
        tc_seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306,
                                  key="tc_seed")

        st.markdown("**运行时控制**")
        tc_num_workers = st.number_input("NUM_WORKERS (并发线程)", 1, 32, 1,
                                         key="tc_workers")
        tc_save_every = st.number_input("SAVE_EVERY (每N条保存)", 1, 500, 10,
                                        key="tc_save_every")
        tc_max_retries = st.number_input("MAX_RETRIES (失败重试)", 1, 10, 3,
                                         key="tc_retries")
        tc_sleep_sec = st.number_input("SLEEP_SEC (请求间隔秒)", 0, 60, 0,
                                       key="tc_sleep")

    with col_right:
        st.markdown("**教师行为**")
        tc_lang = st.selectbox("TEACHER_LANG (标注语言)", ["en", "zh"], index=0,
                               key="tc_lang")
        tc_enable_thinking = st.checkbox("TEACHER_ENABLE_THINKING (启用思维链)",
                                         value=False, key="tc_thinking")
        tc_strict = st.checkbox("STRICT_TEACHER_DISTILL (严格蒸馏模式)",
                                value=True, key="tc_strict")
        tc_collect = st.checkbox("COLLECT_TEACHER_DATA (执行采集)",
                                 value=True, key="tc_collect")

        st.markdown("**四字段 Token 预算**")
        tc_observed_max = st.number_input("observed_facts 上限", 32, 1024, 256,
                                          key="tc_obs_tok")
        tc_context_max = st.number_input("context_textual 上限", 32, 1024, 192,
                                         key="tc_ctx_tok")
        tc_reasoning_max = st.number_input("reasoning 上限", 32, 1024, 256,
                                           key="tc_rea_tok")
        tc_answer_max = st.number_input("answer 上限", 16, 512, 64,
                                        key="tc_ans_tok")
        tc_total_max = st.number_input("总 Token 上限", 64, 4096, 768,
                                       key="tc_total_tok")

    if st.button("🚀 开始采集教师数据", use_container_width=True, key="btn_collect"):
        if not glob_teacher_api_key:
            st.error("请在侧边栏填写 TEACHER_API_KEY")
        else:
            payload = {
                **GLOBAL_ENV,
                "SCIENCEQA_SPLIT": tc_split,
                "TRAIN_NUM": tc_train_num,
                "MAX_SAMPLES": tc_max_samples,
                "SCIENCEQA_SEED": tc_seed,
                "TEACHER_LANG": tc_lang,
                "TEACHER_ENABLE_THINKING": int(tc_enable_thinking),
                "STRICT_TEACHER_DISTILL": int(tc_strict),
                "COLLECT_TEACHER_DATA": int(tc_collect),
                "TEACHER_OBSERVED_MAX_TOKENS": tc_observed_max,
                "TEACHER_CONTEXT_MAX_TOKENS": tc_context_max,
                "TEACHER_REASONING_MAX_TOKENS": tc_reasoning_max,
                "TEACHER_ANSWER_MAX_TOKENS": tc_answer_max,
                "TEACHER_MAX_NEW_TOKENS_TOTAL": tc_total_max,
                "MAX_RETRIES": tc_max_retries,
                "NUM_WORKERS": tc_num_workers,
                "SAVE_EVERY": tc_save_every,
                "SLEEP_SEC": tc_sleep_sec,
            }
            try:
                resp = requests.post(f"{BASE_URL}/api/task/collect", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.success("✅ 采集任务已提交后台执行")
                else:
                    st.error(f"❌ 提交失败: HTTP {resp.status_code} — {resp.text}")
            except Exception as e:
                st.error(f"❌ 请求异常: {e}")

    st.divider()

    # ---------- 1B: 分桶预处理 ----------
    st.subheader("1B. ScienceQA 图像分桶预处理")
    st.info("按视觉 patch 数对样本分桶，生成批采样 JSON，提升后续训练吞吐。")

    pp_col1, pp_col2 = st.columns(2)

    with pp_col1:
        pp_split = st.selectbox("SCIENCEQA_SPLIT", ["train", "validation", "test"],
                                index=0, key="pp_split")
        pp_train_num = st.number_input("TRAIN_NUM (0=全量)", 0, 100000, 0,
                                       key="pp_train_num")
        pp_seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306,
                                  key="pp_seed")
        pp_bucket_by = st.selectbox("BUCKET_BY (分桶策略)",
                                    ["patches", "hw", "none"], index=0,
                                    key="pp_bucket_by")

    with pp_col2:
        pp_batch_size = st.number_input("BUCKET_BATCH_SIZE", 1, 128, 8,
                                        key="pp_bs")
        pp_drop_last = st.checkbox("BUCKET_DROP_LAST (丢弃尾包)", value=False,
                                   key="pp_drop")
        pp_shuffle = st.checkbox("SHUFFLE (桶间打乱)", value=True,
                                 key="pp_shuffle")
        pp_preview_buckets = st.number_input("PREVIEW_BUCKETS", 1, 50, 10,
                                             key="pp_prev_bk")
        pp_preview_batches = st.number_input("PREVIEW_BATCHES", 1, 50, 10,
                                             key="pp_prev_ba")

    if st.button("🚀 开始分桶预处理", use_container_width=True, key="btn_preprocess"):
        payload = {
            **GLOBAL_ENV,
            "SCIENCEQA_SPLIT": pp_split,
            "TRAIN_NUM": pp_train_num,
            "SCIENCEQA_SEED": pp_seed,
            "BUCKET_BY": pp_bucket_by,
            "BUCKET_BATCH_SIZE": pp_batch_size,
            "BUCKET_DROP_LAST": int(pp_drop_last),
            "SHUFFLE": int(pp_shuffle),
            "PREVIEW_BUCKETS": pp_preview_buckets,
            "PREVIEW_BATCHES": pp_preview_batches,
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/task/preprocess", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("✅ 预处理任务已提交后台执行")
            else:
                st.error(f"❌ 提交失败: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"❌ 请求异常: {e}")

# ============================================================
# Tab 2–5: 占位（下一步填充）
# ============================================================
with tab_s1:
    st.header("2. Stage1: 视觉离散表示学习 (Codebook)")
    st.info("冻结 LLM 主干，只训练 VQ stack（pre_quant → codebook → post_quant）。"
            "产物为 vq_codebook.pt，供 Stage2/3 复用。")

    # ---- 前置检查提示 ----
    st.warning("⚠️ 请确保 Tab1 的分桶预处理已完成，Stage1 启动时会自动检查预处理文件是否存在。"
               "若不存在，脚本会自动触发预处理（需要 MODEL_PATH 可访问）。")

    # ========== 数据与分桶 ==========
    with st.expander("📦 数据与分桶配置", expanded=False):
        s1d_col1, s1d_col2 = st.columns(2)
        with s1d_col1:
            s1_split = st.selectbox("SCIENCEQA_SPLIT", ["train", "validation", "test"],
                                    index=0, key="s1_split")
            s1_train_num = st.number_input("TRAIN_NUM (0=全量)", 0, 100000, 0,
                                           key="s1_train_num")
            s1_seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306,
                                      key="s1_seed")
        with s1d_col2:
            s1_bucket_by = st.selectbox("BUCKET_BY", ["patches", "hw", "none"],
                                        index=0, key="s1_bucket_by")
            s1_bucket_bs = st.number_input("BUCKET_BATCH_SIZE", 1, 128, 8,
                                           key="s1_bucket_bs")
            s1_bucket_drop = st.checkbox("BUCKET_DROP_LAST", value=False,
                                         key="s1_bucket_drop")

    # ========== 训练超参 ==========
    with st.expander("🎛️ 训练超参数", expanded=True):
        s1t_col1, s1t_col2, s1t_col3 = st.columns(3)
        with s1t_col1:
            s1_epochs = st.number_input("EPOCHS", 1, 200, 50, key="s1_epochs")
            s1_batch_size = st.number_input("BATCH_SIZE", 1, 64, 8, key="s1_bs")
            s1_grad_accum = st.number_input("GRAD_ACCUM", 1, 64, 8, key="s1_grad_accum")
        with s1t_col2:
            s1_lr = st.text_input("LR (全局学习率)", value="2e-5", key="s1_lr")
            s1_stage1_lr = st.text_input("STAGE1_LR (VQ 专用学习率)", value="7e-5",
                                         key="s1_stage1_lr")
            s1_max_length = st.number_input("MAX_LENGTH", 128, 4096, 1024,
                                            key="s1_max_length")
        with s1t_col3:
            s1_max_new_tokens = st.number_input("MAX_NEW_TOKENS", 32, 1024, 256,
                                                key="s1_max_new_tokens")
            s1_temperature = st.number_input("TEMPERATURE", 0.1, 5.0, 1.5,
                                             step=0.1, key="s1_temperature")
            s1_tau1 = st.text_input("TAU1", value="0.01", key="s1_tau1")

    # ========== VQ 损失权重 ==========
    with st.expander("⚖️ VQ 损失权重", expanded=True):
        s1w_col1, s1w_col2 = st.columns(2)
        with s1w_col1:
            s1_recon_weight = st.slider("STAGE1_RECON_WEIGHT (重建损失)", 0.0, 5.0, 1.0,
                                        step=0.05, key="s1_recon_w")
            s1_cosine_weight = st.slider("STAGE1_COSINE_WEIGHT (余弦损失)", 0.0, 2.0, 0.25,
                                         step=0.05, key="s1_cosine_w")
            s1_vq_weight = st.slider("STAGE1_VQ_WEIGHT (VQ commitment)", 0.0, 5.0, 1.0,
                                     step=0.05, key="s1_vq_w")
        with s1w_col2:
            s1_beta = st.slider("BETA (VQ 全局权重)", 0.0, 2.0, 0.25,
                                step=0.05, key="s1_beta")
            s1_grad_clip = st.number_input("STAGE1_GRAD_CLIP", 0.1, 20.0, 5.0,
                                           step=0.1, key="s1_grad_clip")

    # ========== VQ Codebook 配置 ==========
    with st.expander("📚 VQ Codebook 配置", expanded=False):
        s1c_col1, s1c_col2 = st.columns(2)
        with s1c_col1:
            s1_codebook_size = st.selectbox("VQ_CODEBOOK_SIZE",
                                            [256, 512, 1024, 2048, 4096, 8192],
                                            index=2, key="s1_cb_size")
            s1_commitment_cost = st.slider("VQ_COMMITMENT_COST", 0.0, 1.0, 0.25,
                                           step=0.05, key="s1_commit")
            s1_dead_threshold = st.number_input("VQ_DEAD_CODE_THRESHOLD", 0.1, 10.0, 1.0,
                                                step=0.1, key="s1_dead_thr")
        with s1c_col2:
            s1_usage_decay = st.number_input("VQ_USAGE_DECAY", 0.9, 1.0, 0.995,
                                             step=0.001, format="%.3f", key="s1_decay")
            s1_reset_interval = st.number_input("VQ_DEAD_CODE_RESET_INTERVAL (steps)",
                                                1, 200, 20, key="s1_reset_int")
            s1_legacy_loss = st.checkbox("VQ_LEGACY_LOSS (旧版损失)", value=False,
                                         key="s1_legacy")

    # ========== 模型配置 ==========
    with st.expander("🧠 模型与量化", expanded=False):
        s1m_col1, s1m_col2 = st.columns(2)
        with s1m_col1:
            s1_freeze_vision = st.checkbox("FREEZE_VISION_TOWER (冻结视觉编码器)",
                                           value=False, key="s1_freeze_vis")
            s1_use_lora = st.checkbox("USE_LORA", value=False, key="s1_use_lora")
            s1_use_4bit = st.checkbox("USE_4BIT (4bit量化)", value=False, key="s1_4bit")
        with s1m_col2:
            s1_lora_rank = st.selectbox("LORA_RANK", [16, 32, 64, 128], index=2,
                                        key="s1_lora_rank")
            s1_lora_alpha = st.number_input("LORA_ALPHA", 16, 512, 128,
                                            key="s1_lora_alpha")
            s1_model_dtype = st.selectbox("MODEL_DTYPE", ["bfloat16", "float16", "float32"],
                                          index=0, key="s1_dtype")

    # ========== 蒸馏复用 ==========
    with st.expander("🔄 蒸馏与复用设置", expanded=False):
        s1r_col1, s1r_col2 = st.columns(2)
        with s1r_col1:
            s1_collect_teacher = st.checkbox("COLLECT_TEACHER_DATA (Stage1中采集)",
                                             value=False, key="s1_collect")
            s1_strict_distill = st.checkbox("STRICT_TEACHER_DISTILL", value=False,
                                            key="s1_strict")
            s1_teacher_lang = st.selectbox("TEACHER_LANG", ["en", "zh"], index=0,
                                           key="s1_lang")
        with s1r_col2:
            s1_reuse_codebook = st.checkbox("REUSE_VQ_CODEBOOK (复用已有码本)",
                                            value=False, key="s1_reuse_cb")
            s1_reuse_stage2 = st.checkbox("REUSE_STAGE2", value=True,
                                          key="s1_reuse_s2")

    # ========== 日志与保存 ==========
    with st.expander("💾 日志与保存", expanded=False):
        s1s_col1, s1s_col2 = st.columns(2)
        with s1s_col1:
            s1_log_step = st.number_input("LOG_STEP", 1, 500, 20, key="s1_log_step")
            s1_save_step = st.number_input("SAVE_STEP", 10, 2000, 100, key="s1_save_step")
        with s1s_col2:
            s1_save_each_epoch = st.checkbox("SAVE_EACH_EPOCH", value=True,
                                             key="s1_save_epoch")

    # ========== 启动按钮 ==========
    st.divider()
    if st.button("🚀 启动 Stage 1 训练", use_container_width=True, key="btn_stage1"):
        payload = {
            **GLOBAL_ENV,
            # 数据与分桶
            "SCIENCEQA_SPLIT": s1_split,
            "TRAIN_NUM": s1_train_num,
            "SCIENCEQA_SEED": s1_seed,
            "BUCKET_BY": s1_bucket_by,
            "BUCKET_BATCH_SIZE": s1_bucket_bs,
            "BUCKET_DROP_LAST": int(s1_bucket_drop),
            "DISABLE_BUCKET_FOR_STAGE3": 0,
            "STAGE3_BUCKET_BATCH_SIZE": s1_bucket_bs,
            # 训练超参
            "EPOCHS": s1_epochs,
            "BATCH_SIZE": s1_batch_size,
            "GRAD_ACCUM": s1_grad_accum,
            "STAGE2_GRAD_ACCUM": 0,
            "STAGE3_GRAD_ACCUM": 0,
            "LR": s1_lr,
            "STAGE1_LR": s1_stage1_lr,
            "MAX_LENGTH": s1_max_length,
            "MAX_NEW_TOKENS": s1_max_new_tokens,
            "TEMPERATURE": s1_temperature,
            "TAU1": s1_tau1,
            # VQ 损失权重
            "STAGE1_RECON_WEIGHT": s1_recon_weight,
            "STAGE1_COSINE_WEIGHT": s1_cosine_weight,
            "STAGE1_VQ_WEIGHT": s1_vq_weight,
            "STAGE1_GRAD_CLIP": s1_grad_clip,
            "BETA": s1_beta,
            # VQ codebook
            "VQ_CODEBOOK_SIZE": s1_codebook_size,
            "VQ_COMMITMENT_COST": s1_commitment_cost,
            "VQ_DEAD_CODE_THRESHOLD": s1_dead_threshold,
            "VQ_USAGE_DECAY": s1_usage_decay,
            "VQ_DEAD_CODE_RESET_INTERVAL": s1_reset_interval,
            "VQ_LEGACY_LOSS": int(s1_legacy_loss),
            # 模型
            "FREEZE_VISION_TOWER": int(s1_freeze_vision),
            "USE_LORA": int(s1_use_lora),
            "LORA_RANK": s1_lora_rank,
            "LORA_ALPHA": s1_lora_alpha,
            "USE_4BIT": int(s1_use_4bit),
            "MODEL_DTYPE": s1_model_dtype,
            # 蒸馏复用
            "COLLECT_TEACHER_DATA": int(s1_collect_teacher),
            "STRICT_TEACHER_DISTILL": int(s1_strict_distill),
            "TEACHER_LANG": s1_teacher_lang,
            "REUSE_VQ_CODEBOOK": int(s1_reuse_codebook),
            "REUSE_STAGE2": int(s1_reuse_stage2),
            # 日志保存
            "LOG_STEP": s1_log_step,
            "SAVE_STEP": s1_save_step,
            "SAVE_EACH_EPOCH": int(s1_save_each_epoch),
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/task/stage1", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("✅ Stage1 训练任务已提交后台执行")
            else:
                st.error(f"❌ 提交失败: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"❌ 请求异常: {e}")

with tab_s2:
    st.header("3. Stage2: 教师视觉能力迁移")
    st.info("加载 Stage1 codebook（冻结 embedding），训练 LoRA + projector + pre/post_quant。"
            "训练完成后自动执行一次 validation 评测。")

    st.warning("⚠️ 前置条件：Stage1 的 `vq_codebook.pt` 必须已生成。"
               "脚本启动时会自动校验 `STAGE1_CODEBOOK_PATH` 是否存在。")

    # ========== 路径衔接 ==========
    with st.expander("📂 路径与衔接（Stage1 产物）", expanded=False):
        # 探测按钮
        s2_detect_col1, s2_detect_col2 = st.columns(2)
        with s2_detect_col1:
            if st.button("🔍 探测 Stage1 codebook 目录", key="s2_detect_s1"):
                try:
                    resp = requests.get(
                        f"{BASE_URL}/api/list_ckpt_dirs",
                        params={"root_dir": glob_root_dir, "prefix": "stage1_vq"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        dirs = resp.json().get("dirs", [])
                        if dirs:
                            for d in dirs:
                                cb_icon = "✅" if d["has_codebook"] else "❌"
                                st.text(f"  {cb_icon} codebook | {d['name']} → {d['path']}")
                        else:
                            st.warning("未找到 stage1_vq* 目录")
                    else:
                        st.error(f"HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"探测失败: {e}")
        with s2_detect_col2:
            if st.button("🔍 探测已有 Stage2 目录", key="s2_detect_s2"):
                try:
                    resp = requests.get(
                        f"{BASE_URL}/api/list_ckpt_dirs",
                        params={"root_dir": glob_root_dir, "prefix": "stage2_vision"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        dirs = resp.json().get("dirs", [])
                        if dirs:
                            for d in dirs:
                                a_icon = "✅" if d["has_adapter"] else "❌"
                                p_icon = "✅" if d["has_projector"] else "❌"
                                cb_icon = "✅" if d["has_codebook"] else "❌"
                                st.text(f"  {a_icon} adapter {p_icon} projector {cb_icon} codebook | {d['name']}")
                        else:
                            st.warning("未找到 stage2_vision* 目录")
                    else:
                        st.error(f"HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"探测失败: {e}")

        s2p_col1, s2p_col2 = st.columns(2)
        with s2p_col1:
            s2_stage1_codebook = st.text_input(
                "STAGE1_CODEBOOK_PATH",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage1_vq/vq_codebook.pt",
                key="s2_s1_cb_path",
                help="Stage1 产出的 codebook 文件路径。实际目录可能带 epoch 后缀，请先点击探测确认。"
            )
        with s2p_col2:
            s2_stage2_ckpt = st.text_input(
                "STAGE2_CKPT_PATH (Stage2 产物保存目录)",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage2_vision",
                key="s2_ckpt_path"
            )

    # ========== 数据与分桶 ==========
    with st.expander("📦 数据与分桶配置", expanded=False):
        s2d_col1, s2d_col2 = st.columns(2)
        with s2d_col1:
            s2_split = st.selectbox("SCIENCEQA_SPLIT", ["train", "validation", "test"],
                                    index=0, key="s2_split")
            s2_train_num = st.number_input("TRAIN_NUM (0=全量)", 0, 100000, 0,
                                           key="s2_train_num")
            s2_seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306,
                                      key="s2_seed")
        with s2d_col2:
            s2_bucket_by = st.selectbox("BUCKET_BY", ["patches", "hw", "none"],
                                        index=0, key="s2_bucket_by")
            s2_preprocess_bucket_bs = st.number_input(
                "PREPROCESS_BUCKET_BATCH_SIZE (预处理桶大小)",
                1, 128, 8, key="s2_pp_bucket_bs",
                help="用于定位预处理 JSON 文件名中的 bs 后缀"
            )
            s2_bucket_bs = st.number_input("BUCKET_BATCH_SIZE (运行时桶大小)",
                                           1, 128, 8, key="s2_bucket_bs")
            s2_bucket_drop = st.checkbox("BUCKET_DROP_LAST", value=False,
                                         key="s2_bucket_drop")

    # ========== Stage2 训练超参 ==========
    with st.expander("🎛️ 训练超参数", expanded=True):
        s2t_col1, s2t_col2, s2t_col3 = st.columns(3)
        with s2t_col1:
            s2_epochs = st.number_input("EPOCHS", 1, 50, 3, key="s2_epochs")
            s2_batch_size = st.number_input("BATCH_SIZE", 1, 64, 8, key="s2_bs")
            s2_grad_accum = st.number_input("GRAD_ACCUM", 1, 64, 4, key="s2_grad_accum")
            s2_stage2_grad_accum = st.number_input("STAGE2_GRAD_ACCUM", 1, 64, 4,
                                                    key="s2_s2_grad_accum")
        with s2t_col2:
            s2_lr = st.text_input("LR (全局学习率)", value="3e-5", key="s2_lr")
            s2_stage1_lr = st.text_input("STAGE1_LR (VQ 部分学习率)", value="5e-5",
                                         key="s2_stage1_lr")
            s2_max_length = st.number_input("MAX_LENGTH", 128, 4096, 1024,
                                            key="s2_max_length")
            s2_max_new_tokens = st.number_input("MAX_NEW_TOKENS", 32, 1024, 128,
                                                key="s2_max_new_tokens")
        with s2t_col3:
            s2_temperature = st.number_input("TEMPERATURE", 0.1, 5.0, 1.5,
                                             step=0.1, key="s2_temperature")
            s2_tau1 = st.text_input("TAU1", value="0.01", key="s2_tau1")

    # ========== Stage2 独有损失权重 ==========
    with st.expander("⚖️ Stage2 蒸馏损失权重", expanded=True):
        s2w_col1, s2w_col2 = st.columns(2)
        with s2w_col1:
            s2_answer_weight = st.slider("STAGE2_ANSWER_WEIGHT (答案损失)", 0.0, 3.0, 1.0,
                                         step=0.05, key="s2_ans_w")
            s2_rationale_weight = st.slider("STAGE2_RATIONALE_WEIGHT (推理链损失)", 0.0, 2.0, 0.2,
                                            step=0.05, key="s2_rat_w")
            s2_beta = st.slider("BETA (VQ 全局权重，Stage2 建议较低)", 0.0, 1.0, 0.05,
                                step=0.01, key="s2_beta")
        with s2w_col2:
            s2_prepost_lr_scale = st.slider("STAGE2_PREPOST_LR_SCALE (pre/post_quant LR缩放)",
                                            0.0, 1.0, 0.2, step=0.05, key="s2_prepost_lr")
            s2_vision_lr_scale = st.slider("STAGE2_VISION_LR_SCALE (视觉模块 LR缩放)",
                                           0.0, 1.0, 0.2, step=0.05, key="s2_vis_lr")
            s2_grad_clip = st.number_input("STAGE2_GRAD_CLIP", 0.1, 20.0, 1.0,
                                           step=0.1, key="s2_grad_clip")

    # ========== Stage1 继承参数（VQ Codebook） ==========
    with st.expander("📚 VQ Codebook 配置（继承自 Stage1）", expanded=False):
        st.caption("这些参数应与 Stage1 训练时保持一致，否则无法正确加载 codebook。")
        s2c_col1, s2c_col2 = st.columns(2)
        with s2c_col1:
            s2_codebook_size = st.selectbox("VQ_CODEBOOK_SIZE",
                                            [256, 512, 1024, 2048, 4096, 8192],
                                            index=2, key="s2_cb_size")
            s2_commitment_cost = st.slider("VQ_COMMITMENT_COST", 0.0, 1.0, 0.25,
                                           step=0.05, key="s2_commit")
            s2_dead_threshold = st.number_input("VQ_DEAD_CODE_THRESHOLD", 0.1, 10.0, 1.0,
                                                step=0.1, key="s2_dead_thr")
        with s2c_col2:
            s2_usage_decay = st.number_input("VQ_USAGE_DECAY", 0.9, 1.0, 0.995,
                                             step=0.001, format="%.3f", key="s2_decay")
            s2_reset_interval = st.number_input("VQ_DEAD_CODE_RESET_INTERVAL",
                                                1, 200, 10, key="s2_reset_int")
            s2_legacy_loss = st.checkbox("VQ_LEGACY_LOSS", value=False, key="s2_legacy")

    # ========== Stage1 继承参数（损失权重） ==========
    with st.expander("📐 Stage1 损失权重（继承，供训练入口使用）", expanded=False):
        st.caption("train_vq_lord3.py 是统一入口，Stage2 也需要传入 Stage1 的损失参数。")
        s2s1_col1, s2s1_col2 = st.columns(2)
        with s2s1_col1:
            s2_s1_recon = st.slider("STAGE1_RECON_WEIGHT", 0.0, 5.0, 1.0,
                                    step=0.05, key="s2_s1_recon")
            s2_s1_cosine = st.slider("STAGE1_COSINE_WEIGHT", 0.0, 2.0, 0.25,
                                     step=0.05, key="s2_s1_cosine")
        with s2s1_col2:
            s2_s1_vq = st.slider("STAGE1_VQ_WEIGHT", 0.0, 5.0, 1.0,
                                 step=0.05, key="s2_s1_vq")
            s2_s1_grad_clip = st.number_input("STAGE1_GRAD_CLIP", 0.1, 20.0, 5.0,
                                              step=0.1, key="s2_s1_grad_clip")

    # ========== 模型配置 ==========
    with st.expander("🧠 模型与量化", expanded=False):
        s2m_col1, s2m_col2 = st.columns(2)
        with s2m_col1:
            s2_freeze_vision = st.checkbox("FREEZE_VISION_TOWER", value=False,
                                           key="s2_freeze_vis")
            s2_use_lora = st.checkbox("USE_LORA", value=True, key="s2_use_lora")
            s2_use_4bit = st.checkbox("USE_4BIT", value=False, key="s2_4bit")
        with s2m_col2:
            s2_lora_rank = st.selectbox("LORA_RANK", [16, 32, 64, 128], index=2,
                                        key="s2_lora_rank")
            s2_lora_alpha = st.number_input("LORA_ALPHA", 16, 512, 128,
                                            key="s2_lora_alpha")
            s2_model_dtype = st.selectbox("MODEL_DTYPE", ["bfloat16", "float16", "float32"],
                                          index=0, key="s2_dtype")

    # ========== 蒸馏复用 ==========
    with st.expander("🔄 蒸馏与复用设置", expanded=False):
        s2r_col1, s2r_col2 = st.columns(2)
        with s2r_col1:
            s2_collect_teacher = st.checkbox("COLLECT_TEACHER_DATA", value=False,
                                             key="s2_collect")
            s2_strict_distill = st.checkbox("STRICT_TEACHER_DISTILL", value=False,
                                            key="s2_strict")
            s2_teacher_lang = st.selectbox("TEACHER_LANG", ["en", "zh"], index=0,
                                           key="s2_lang")
        with s2r_col2:
            s2_reuse_codebook = st.checkbox("REUSE_VQ_CODEBOOK (必须复用 Stage1 码本)",
                                            value=True, key="s2_reuse_cb")
            s2_reuse_stage2 = st.checkbox("REUSE_STAGE2 (复用已有 Stage2 ckpt)",
                                          value=False, key="s2_reuse_s2")

    # ========== 训练后自动评测 ==========
    with st.expander("📊 训练后自动评测", expanded=False):
        s2e_col1, s2e_col2 = st.columns(2)
        with s2e_col1:
            s2_eval_split = st.selectbox("EVAL_SPLIT", ["validation", "test"],
                                         index=0, key="s2_eval_split")
            s2_eval_max_samples = st.number_input("EVAL_MAX_SAMPLES (0=全量)", 0, 10000, 500,
                                                  key="s2_eval_max")
        with s2e_col2:
            s2_eval_max_new_tokens = st.number_input("EVAL_MAX_NEW_TOKENS", 16, 512, 64,
                                                     key="s2_eval_tokens")
            s2_eval_answer_mode = st.selectbox("EVAL_ANSWER_MODE",
                                               ["logits", "generate", "hybrid"],
                                               index=0, key="s2_eval_mode")

    # ========== 日志与保存 ==========
    with st.expander("💾 日志与保存", expanded=False):
        s2s_col1, s2s_col2 = st.columns(2)
        with s2s_col1:
            s2_log_step = st.number_input("LOG_STEP", 1, 500, 50, key="s2_log_step")
            s2_save_step = st.number_input("SAVE_STEP", 10, 2000, 100, key="s2_save_step")
        with s2s_col2:
            s2_save_each_epoch = st.checkbox("SAVE_EACH_EPOCH", value=True,
                                             key="s2_save_epoch")

    # ========== 启动按钮 ==========
    st.divider()
    if st.button("🚀 启动 Stage 2 训练", use_container_width=True, key="btn_stage2"):
        payload = {
            **GLOBAL_ENV,
            # 路径衔接
            "STAGE1_CODEBOOK_PATH": s2_stage1_codebook,
            "STAGE2_CKPT_PATH": s2_stage2_ckpt,
            # 数据与分桶
            "SCIENCEQA_SPLIT": s2_split,
            "TRAIN_NUM": s2_train_num,
            "SCIENCEQA_SEED": s2_seed,
            "BUCKET_BY": s2_bucket_by,
            "PREPROCESS_BUCKET_BATCH_SIZE": s2_preprocess_bucket_bs,
            "BUCKET_BATCH_SIZE": s2_bucket_bs,
            "BUCKET_DROP_LAST": int(s2_bucket_drop),
            "STAGE3_BUCKET_BATCH_SIZE": s2_bucket_bs,
            "DISABLE_BUCKET_FOR_STAGE3": 0,
            # 训练超参
            "EPOCHS": s2_epochs,
            "BATCH_SIZE": s2_batch_size,
            "GRAD_ACCUM": s2_grad_accum,
            "STAGE2_GRAD_ACCUM": s2_stage2_grad_accum,
            "STAGE3_GRAD_ACCUM": 0,
            "LR": s2_lr,
            "STAGE1_LR": s2_stage1_lr,
            "MAX_LENGTH": s2_max_length,
            "MAX_NEW_TOKENS": s2_max_new_tokens,
            "TEMPERATURE": s2_temperature,
            "TAU1": s2_tau1,
            # Stage2 独有损失
            "STAGE2_ANSWER_WEIGHT": s2_answer_weight,
            "STAGE2_RATIONALE_WEIGHT": s2_rationale_weight,
            "STAGE2_PREPOST_LR_SCALE": s2_prepost_lr_scale,
            "STAGE2_VISION_LR_SCALE": s2_vision_lr_scale,
            "STAGE2_GRAD_CLIP": s2_grad_clip,
            "BETA": s2_beta,
            # Stage1 继承损失
            "STAGE1_RECON_WEIGHT": s2_s1_recon,
            "STAGE1_COSINE_WEIGHT": s2_s1_cosine,
            "STAGE1_VQ_WEIGHT": s2_s1_vq,
            "STAGE1_GRAD_CLIP": s2_s1_grad_clip,
            # VQ codebook
            "VQ_CODEBOOK_SIZE": s2_codebook_size,
            "VQ_COMMITMENT_COST": s2_commitment_cost,
            "VQ_DEAD_CODE_THRESHOLD": s2_dead_threshold,
            "VQ_USAGE_DECAY": s2_usage_decay,
            "VQ_DEAD_CODE_RESET_INTERVAL": s2_reset_interval,
            "VQ_LEGACY_LOSS": int(s2_legacy_loss),
            # 模型
            "FREEZE_VISION_TOWER": int(s2_freeze_vision),
            "USE_LORA": int(s2_use_lora),
            "LORA_RANK": s2_lora_rank,
            "LORA_ALPHA": s2_lora_alpha,
            "USE_4BIT": int(s2_use_4bit),
            "MODEL_DTYPE": s2_model_dtype,
            # 蒸馏复用
            "COLLECT_TEACHER_DATA": int(s2_collect_teacher),
            "STRICT_TEACHER_DISTILL": int(s2_strict_distill),
            "TEACHER_LANG": s2_teacher_lang,
            "REUSE_VQ_CODEBOOK": int(s2_reuse_codebook),
            "REUSE_STAGE2": int(s2_reuse_stage2),
            # 训练后评测
            "EVAL_SPLIT": s2_eval_split,
            "EVAL_MAX_SAMPLES": s2_eval_max_samples,
            "EVAL_MAX_NEW_TOKENS": s2_eval_max_new_tokens,
            "EVAL_ANSWER_MODE": s2_eval_answer_mode,
            # 日志保存
            "LOG_STEP": s2_log_step,
            "SAVE_STEP": s2_save_step,
            "SAVE_EACH_EPOCH": int(s2_save_each_epoch),
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/task/stage2", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("✅ Stage2 训练任务已提交后台执行")
            else:
                st.error(f"❌ 提交失败: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"❌ 请求异常: {e}")

with tab_s3:
    st.header("4. Stage3: 偏好对齐 (Accuracy-First LoRD)")
    st.info("加载 Stage2 全套产物（LoRA + projector + codebook），执行多 Period 迭代的 LoRD 偏好训练。"
            "训练后自动运行 validation + test 双评测。")

    st.warning("⚠️ 前置条件：Stage1 codebook 与 Stage2 checkpoint 必须已生成。"
               "注意：实际产物目录名可能带 epoch 后缀（如 `stage1_vq_epoch40`、`stage2_vision_epoch3`），"
               "请点击下方'探测'按钮确认准确路径。")

    # ========== 路径衔接 ==========
    with st.expander("📂 路径与衔接（Stage1/2 产物）", expanded=True):
        # 探测按钮
        s3_detect_col1, s3_detect_col2, s3_detect_col3 = st.columns(3)
        with s3_detect_col1:
            if st.button("🔍 探测 Stage1 目录", key="s3_detect_s1"):
                try:
                    resp = requests.get(
                        f"{BASE_URL}/api/list_ckpt_dirs",
                        params={"root_dir": glob_root_dir, "prefix": "stage1_vq"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        dirs = resp.json().get("dirs", [])
                        if dirs:
                            for d in dirs:
                                cb_icon = "✅" if d["has_codebook"] else "❌"
                                st.text(f"  {cb_icon} codebook | {d['name']} → {d['path']}")
                        else:
                            st.warning("未找到 stage1_vq* 目录")
                    else:
                        st.error(f"HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"探测失败: {e}")
        with s3_detect_col2:
            if st.button("🔍 探测 Stage2 目录", key="s3_detect_s2"):
                try:
                    resp = requests.get(
                        f"{BASE_URL}/api/list_ckpt_dirs",
                        params={"root_dir": glob_root_dir, "prefix": "stage2_vision"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        dirs = resp.json().get("dirs", [])
                        if dirs:
                            for d in dirs:
                                a_icon = "✅" if d["has_adapter"] else "❌"
                                p_icon = "✅" if d["has_projector"] else "❌"
                                cb_icon = "✅" if d["has_codebook"] else "❌"
                                st.text(f"  {a_icon} adapter {p_icon} proj {cb_icon} cb | {d['name']}")
                        else:
                            st.warning("未找到 stage2_vision* 目录")
                    else:
                        st.error(f"HTTP {resp.status_code}")
                except Exception as e:
                    st.error(f"探测失败: {e}")
        with s3_detect_col3:
            if st.button("🔍 探测 Stage3 已有产物", key="s3_detect_s3"):
                try:
                    # 探测 stage3_sub / stage3_lord / stage3_resume
                    found_any = False
                    for prefix in ["stage3_sub", "stage3_lord", "stage3_resume"]:
                        resp = requests.get(
                            f"{BASE_URL}/api/list_ckpt_dirs",
                            params={"root_dir": glob_root_dir, "prefix": prefix},
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            dirs = resp.json().get("dirs", [])
                            for d in dirs:
                                found_any = True
                                a_icon = "✅" if d["has_adapter"] else "❌"
                                cb_icon = "✅" if d["has_codebook"] else "❌"
                                st.text(f"  {a_icon} adapter {cb_icon} codebook | {d['name']}")
                    if not found_any:
                        st.info("未找到 Stage3 已有产物（首次训练无需关注）")
                except Exception as e:
                    st.error(f"探测失败: {e}")

        s3p_col1, s3p_col2 = st.columns(2)
        with s3p_col1:
            s3_stage1_codebook = st.text_input(
                "STAGE1_CODEBOOK_PATH",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage1_vq/vq_codebook.pt",
                key="s3_s1_cb_path",
                help="实际目录可能是 stage1_vq_epoch40 等，请先探测确认"
            )
            s3_stage2_ckpt = st.text_input(
                "STAGE2_CKPT_PATH",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage2_vision",
                key="s3_s2_ckpt_path",
                help="实际目录可能是 stage2_vision_epoch3 等，请先探测确认"
            )
        with s3p_col2:
            s3_final_adapter = st.text_input(
                "STAGE3_FINAL_ADAPTER_PATH (最终产物目录)",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage3_lord_final",
                key="s3_final_path"
            )
            s3_resume_save_path = st.text_input(
                "STAGE3_RESUME_SAVE_PATH (断点保存目录)",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage3_resume_latest",
                key="s3_resume_save"
            )
            s3_resume_path = st.text_input(
                "STAGE3_RESUME_PATH (从此路径恢复训练，留空=从头开始)",
                value="",
                key="s3_resume_from",
                help="若要断点续训，填入 stage3_resume_latest 或某个 stage3_sub1_period{N} 的路径"
            )

    # ========== 数据与分桶 ==========
    with st.expander("📦 数据与分桶配置", expanded=False):
        s3d_col1, s3d_col2 = st.columns(2)
        with s3d_col1:
            s3_split = st.selectbox("SCIENCEQA_SPLIT", ["train", "validation", "test"],
                                    index=0, key="s3_split")
            s3_train_num = st.number_input("TRAIN_NUM (0=全量)", 0, 100000, 0,
                                           key="s3_train_num")
            s3_seed = st.number_input("SCIENCEQA_SEED", 0, 99999999, 20240306,
                                      key="s3_seed")
        with s3d_col2:
            s3_bucket_by = st.selectbox("BUCKET_BY", ["patches", "hw", "none"],
                                        index=0, key="s3_bucket_by")
            s3_bucket_bs = st.number_input("BUCKET_BATCH_SIZE (预处理桶大小)", 1, 128, 8,
                                           key="s3_bucket_bs")
            s3_stage3_bucket_bs = st.number_input("STAGE3_BUCKET_BATCH_SIZE (Stage3运行时桶大小)",
                                                   1, 128, 16, key="s3_s3_bucket_bs")
            s3_bucket_drop = st.checkbox("BUCKET_DROP_LAST", value=False, key="s3_bucket_drop")
            s3_disable_bucket = st.checkbox("DISABLE_BUCKET_FOR_STAGE3", value=False,
                                            key="s3_disable_bucket")

    # ========== 训练结构 ==========
    with st.expander("🏗️ 训练结构（Period / Sub-Stage）", expanded=True):
        s3st_col1, s3st_col2, s3st_col3 = st.columns(3)
        with s3st_col1:
            s3_epochs = st.number_input("EPOCHS", 1, 200, 50, key="s3_epochs")
            s3_sub_stage_num = st.number_input("SUB_STAGE_NUM", 1, 10, 1, key="s3_sub_stage")
        with s3st_col2:
            s3_period_num = st.number_input("PERIOD_NUM", 1, 200, 50, key="s3_period")
            s3_sub_set_num = st.number_input("SUB_SET_NUM (0=全量)", 0, 100000, 0, key="s3_sub_set")
        with s3st_col3:
            s3_batch_size = st.number_input("BATCH_SIZE", 1, 64, 2, key="s3_bs")
            s3_grad_accum = st.number_input("GRAD_ACCUM", 1, 64, 4, key="s3_grad_accum")
            s3_stage2_grad_accum = st.number_input("STAGE2_GRAD_ACCUM (parser filler)", 1, 64, 4,
                                                    key="s3_s2_grad_accum")
            s3_stage3_grad_accum = st.number_input("STAGE3_GRAD_ACCUM", 1, 64, 4,
                                                    key="s3_s3_grad_accum")

    # ========== 学习率与采样 ==========
    with st.expander("🎛️ 学习率与采样", expanded=True):
        s3lr_col1, s3lr_col2, s3lr_col3 = st.columns(3)
        with s3lr_col1:
            s3_lr = st.text_input("LR (全局学习率)", value="3e-5", key="s3_lr")
            s3_lr_scale = st.slider("STAGE3_LR_SCALE", 0.0, 1.0, 0.2,
                                    step=0.05, key="s3_lr_scale")
        with s3lr_col2:
            s3_grad_clip = st.number_input("STAGE3_GRAD_CLIP", 0.1, 20.0, 1.0,
                                           step=0.1, key="s3_grad_clip")
            s3_train_projector = st.checkbox("STAGE3_TRAIN_PROJECTOR", value=False,
                                             key="s3_train_proj")
        with s3lr_col3:
            s3_temperature = st.slider("TEMPERATURE (采样温度)", 0.1, 3.0, 1.2,
                                       step=0.1, key="s3_temp")
            s3_max_new_tokens = st.number_input("MAX_NEW_TOKENS (候选生成)", 32, 1024, 128,
                                                key="s3_max_new")
            s3_max_length = st.number_input("MAX_LENGTH", 128, 4096, 1024, key="s3_max_len")

    # ========== 冷启动阈值 ==========
    with st.expander("❄️ 冷启动阈值", expanded=True):
        s3tau_col1, s3tau_col2 = st.columns(2)
        with s3tau_col1:
            s3_tau1 = st.text_input("TAU1 (冷启动阈值)", value="0.001", key="s3_tau1")
        with s3tau_col2:
            s3_tau_delta = st.text_input("TAU_DELTA (冷启动增量)", value="0.005", key="s3_tau_delta")

    # ========== Accuracy-First 损失权重 ==========
    with st.expander("🎯 Accuracy-First 损失权重", expanded=True):
        s3mc_col1, s3mc_col2 = st.columns(2)
        with s3mc_col1:
            s3_mc_weight = st.slider("STAGE3_MC_WEIGHT (多选CE主损失)", 0.0, 5.0, 1.0,
                                     step=0.05, key="s3_mc_w")
            s3_obj_weight = st.slider("STAGE3_OBJ_WEIGHT (LoRD对比目标)", 0.0, 1.0, 0.05,
                                      step=0.01, key="s3_obj_w")
        with s3mc_col2:
            s3_reg_weight = st.slider("STAGE3_REG_WEIGHT (PPO-clip正则)", 0.0, 2.0, 0.30,
                                      step=0.05, key="s3_reg_w")
            s3_answer_anchor = st.slider("STAGE3_ANSWER_ANCHOR_WEIGHT (答案锚定)", 0.0, 3.0, 1.0,
                                         step=0.05, key="s3_ans_anchor")
            s3_force_cold_period0 = st.checkbox("STAGE3_FORCE_COLD_START_PERIOD0",
                                                value=False, key="s3_force_cold")

    # ========== 四字段 Field Weight ==========
    with st.expander("📝 四字段权重", expanded=False):
        s3fw_col1, s3fw_col2 = st.columns(2)
        with s3fw_col1:
            s3_fw_observed = st.slider("observed_facts_visual", 0.0, 3.0, 1.15,
                                       step=0.05, key="s3_fw_obs")
            s3_fw_context = st.slider("context_textual", 0.0, 3.0, 0.30,
                                      step=0.05, key="s3_fw_ctx")
        with s3fw_col2:
            s3_fw_reasoning = st.slider("reasoning", 0.0, 3.0, 1.35,
                                        step=0.05, key="s3_fw_rsn")
            s3_fw_answer = st.slider("answer", 0.0, 3.0, 1.50,
                                     step=0.05, key="s3_fw_ans")

    # ========== 排序与负样本 ==========
    with st.expander("🔀 排序与负样本增强", expanded=False):
        s3neg_col1, s3neg_col2 = st.columns(2)
        with s3neg_col1:
            s3_pair_correctness = st.checkbox("STAGE3_PAIR_USE_ANSWER_CORRECTNESS (基于正确率排序)",
                                              value=True, key="s3_pair_corr")
            s3_vic_include_context = st.checkbox("STAGE3_VIC_INCLUDE_CONTEXT (y_vic 包含 context)",
                                                 value=False, key="s3_vic_ctx")
        with s3neg_col2:
            s3_wrong_image_enable = st.checkbox("STAGE3_WRONG_IMAGE_ENABLE (错图负样本)",
                                                value=False, key="s3_wrong_img")
            s3_wrong_image_weight = st.slider("STAGE3_WRONG_IMAGE_WEIGHT", 0.0, 1.0, 0.2,
                                              step=0.05, key="s3_wrong_w")
            s3_wrong_image_margin = st.number_input("STAGE3_WRONG_IMAGE_MARGIN", 0.0, 5.0, 0.0,
                                                    step=0.1, key="s3_wrong_m")

    # ========== 教师缓存与 Token 预算 ==========
    with st.expander("📋 教师缓存与 Token 预算", expanded=False):
        s3tc_col1, s3tc_col2 = st.columns(2)
        with s3tc_col1:
            s3_teacher_lang = st.selectbox("TEACHER_LANG", ["en", "zh"], index=0, key="s3_lang")
            s3_collect_teacher = st.checkbox("COLLECT_TEACHER_DATA", value=False, key="s3_collect")
            s3_strict_distill = st.checkbox("STRICT_TEACHER_DISTILL", value=False, key="s3_strict")
            s3_teacher_cache_path = st.text_input(
                "TEACHER_CACHE_PATH (留空=自动推导)",
                value="", key="s3_cache_path",
                help="留空时由 run_stage3.sh 根据 VICTIM_MODEL/SPLIT/TRAIN_NUM/SEED 自动拼接"
            )
        with s3tc_col2:
            s3_tc_obs = st.number_input("TEACHER_OBSERVED_MAX_TOKENS", 32, 1024, 256, key="s3_tc_obs")
            s3_tc_ctx = st.number_input("TEACHER_CONTEXT_MAX_TOKENS", 32, 1024, 192, key="s3_tc_ctx")
            s3_tc_rsn = st.number_input("TEACHER_REASONING_MAX_TOKENS", 32, 1024, 256, key="s3_tc_rsn")
            s3_tc_ans = st.number_input("TEACHER_ANSWER_MAX_TOKENS", 16, 512, 64, key="s3_tc_ans")
            s3_tc_total = st.number_input("TEACHER_MAX_NEW_TOKENS_TOTAL", 64, 4096, 768, key="s3_tc_total")

    # ========== Stage1/2 继承参数 ==========
    with st.expander("📐 Stage1/2 继承参数（填充 parser 必需）", expanded=False):
        st.caption("train_vq_lord3.py 是统一入口，Stage3 也需要传入 Stage1/2 的损失参数。")
        s3inh_col1, s3inh_col2 = st.columns(2)
        with s3inh_col1:
            s3_s1_lr = st.text_input("STAGE1_LR", value="5e-5", key="s3_s1_lr")
            s3_s1_recon = st.slider("STAGE1_RECON_WEIGHT", 0.0, 5.0, 1.0, step=0.05, key="s3_s1_recon")
            s3_s1_cosine = st.slider("STAGE1_COSINE_WEIGHT", 0.0, 2.0, 0.25, step=0.05, key="s3_s1_cos")
            s3_s1_vq = st.slider("STAGE1_VQ_WEIGHT", 0.0, 5.0, 1.0, step=0.05, key="s3_s1_vq")
            s3_s1_grad_clip = st.number_input("STAGE1_GRAD_CLIP", 0.1, 20.0, 5.0, step=0.1, key="s3_s1_gc")
        with s3inh_col2:
            s3_s2_ans_w = st.slider("STAGE2_ANSWER_WEIGHT", 0.0, 3.0, 1.0, step=0.05, key="s3_s2_aw")
            s3_s2_rat_w = st.slider("STAGE2_RATIONALE_WEIGHT", 0.0, 2.0, 0.2, step=0.05, key="s3_s2_rw")
            s3_s2_prepost_lr = st.slider("STAGE2_PREPOST_LR_SCALE", 0.0, 1.0, 0.2, step=0.05, key="s3_s2_pp")
            s3_s2_vis_lr = st.slider("STAGE2_VISION_LR_SCALE", 0.0, 1.0, 0.2, step=0.05, key="s3_s2_vl")
            s3_s2_grad_clip = st.number_input("STAGE2_GRAD_CLIP", 0.1, 20.0, 1.0, step=0.1, key="s3_s2_gc")

    # ========== VQ / 模型 ==========
    with st.expander("📚 VQ Codebook 与模型", expanded=False):
        s3vq_col1, s3vq_col2 = st.columns(2)
        with s3vq_col1:
            s3_codebook_size = st.selectbox("VQ_CODEBOOK_SIZE", [256, 512, 1024, 2048, 4096, 8192],
                                            index=2, key="s3_cb_size")
            s3_commitment_cost = st.slider("VQ_COMMITMENT_COST", 0.0, 1.0, 0.25, step=0.05, key="s3_commit")
            s3_dead_threshold = st.number_input("VQ_DEAD_CODE_THRESHOLD", 0.1, 10.0, 1.0, step=0.1, key="s3_dead")
        with s3vq_col2:
            s3_usage_decay = st.number_input("VQ_USAGE_DECAY", 0.9, 1.0, 0.995,
                                             step=0.001, format="%.3f", key="s3_decay")
            s3_reset_interval = st.number_input("VQ_DEAD_CODE_RESET_INTERVAL", 1, 200, 10, key="s3_reset")
            s3_legacy_loss = st.checkbox("VQ_LEGACY_LOSS", value=False, key="s3_legacy")

        s3m_col1, s3m_col2 = st.columns(2)
        with s3m_col1:
            s3_freeze_vision = st.checkbox("FREEZE_VISION_TOWER", value=False, key="s3_freeze_vis")
            s3_beta = st.slider("BETA (VQ 全局权重)", 0.0, 1.0, 0.05, step=0.01, key="s3_beta")
            s3_use_lora = st.checkbox("USE_LORA", value=True, key="s3_use_lora")
        with s3m_col2:
            s3_lora_rank = st.selectbox("LORA_RANK", [16, 32, 64, 128], index=2, key="s3_lora_rank")
            s3_lora_alpha = st.number_input("LORA_ALPHA", 16, 512, 128, key="s3_lora_alpha")
            s3_use_4bit = st.checkbox("USE_4BIT", value=False, key="s3_4bit")
            s3_model_dtype = st.selectbox("MODEL_DTYPE", ["bfloat16", "float16", "float32"],
                                          index=0, key="s3_dtype")

    # ========== 蒸馏复用 ==========
    with st.expander("🔄 蒸馏与复用设置", expanded=False):
        s3r_col1, s3r_col2 = st.columns(2)
        with s3r_col1:
            s3_reuse_codebook = st.checkbox("REUSE_VQ_CODEBOOK", value=True, key="s3_reuse_cb")
        with s3r_col2:
            s3_reuse_stage2 = st.checkbox("REUSE_STAGE2", value=True, key="s3_reuse_s2")

    # ========== Stage3 Period 内置评测 ==========
    with st.expander("📊 Stage3 Period 内置评测", expanded=False):
        s3ev_col1, s3ev_col2 = st.columns(2)
        with s3ev_col1:
            s3_eval_every = st.number_input("STAGE3_EVAL_EVERY_PERIOD (每N Period评测)",
                                            1, 50, 1, key="s3_eval_every")
            s3_eval_max = st.number_input("STAGE3_EVAL_MAX_SAMPLES (0=全量)", 0, 10000, 0,
                                          key="s3_eval_max")
            s3_eval_train_num = st.number_input("STAGE3_EVAL_TRAIN_NUM (0=全量)", 0, 100000, 0,
                                                key="s3_eval_train_num")
        with s3ev_col2:
            s3_eval_split = st.selectbox("STAGE3_EVAL_SCIENCEQA_SPLIT",
                                         ["validation", "test"], index=0, key="s3_eval_split")
            s3_eval_path = st.text_input("STAGE3_EVAL_SCIENCEQA_PATH (留空=使用全局)",
                                         value="", key="s3_eval_path")
            s3_eval_answer_mode = st.selectbox("STAGE3_EVAL_ANSWER_MODE",
                                               ["logits", "generate", "hybrid"],
                                               index=0, key="s3_eval_mode")

    # ========== 训练后最终评测 ==========
    with st.expander("📈 训练后最终评测", expanded=False):
        s3fe_col1, s3fe_col2 = st.columns(2)
        with s3fe_col1:
            s3_final_eval_max = st.number_input("EVAL_MAX_SAMPLES (最终评测, 0=全量)", 0, 10000, 0,
                                                key="s3_final_max")
        with s3fe_col2:
            s3_final_eval_tokens = st.number_input("EVAL_MAX_NEW_TOKENS", 16, 512, 64,
                                                   key="s3_final_tokens")
            s3_final_eval_mode = st.selectbox("EVAL_ANSWER_MODE (最终)",
                                              ["logits", "generate", "hybrid"],
                                              index=0, key="s3_final_mode")

    # ========== 日志 / 保存 / 断点续训 ==========
    with st.expander("💾 日志、保存与断点续训", expanded=False):
        s3sv_col1, s3sv_col2 = st.columns(2)
        with s3sv_col1:
            s3_log_step = st.number_input("LOG_STEP", 1, 500, 100, key="s3_log_step")
            s3_save_step = st.number_input("SAVE_STEP (0=仅按epoch保存)", 0, 2000, 0, key="s3_save_step")
            s3_save_each_epoch = st.checkbox("SAVE_EACH_EPOCH", value=True, key="s3_save_epoch")
        with s3sv_col2:
            s3_resume_save_optimizer = st.checkbox("STAGE3_RESUME_SAVE_OPTIMIZER (保存优化器状态)",
                                                   value=True, key="s3_resume_opt")
            s3_resume_save_interval = st.number_input("STAGE3_RESUME_SAVE_INTERVAL (每N Period保存断点)",
                                                      1, 50, 1, key="s3_resume_int")
    

    # ========== 启动按钮 ==========
    st.divider()
    if st.button("🔥 启动 Stage 3 训练", use_container_width=True, key="btn_stage3"):
        payload = {
            **GLOBAL_ENV,
            # 路径衔接
            "STAGE1_CODEBOOK_PATH": s3_stage1_codebook,
            "STAGE2_CKPT_PATH": s3_stage2_ckpt,
            "STAGE3_FINAL_ADAPTER_PATH": s3_final_adapter,
            "STAGE3_RESUME_SAVE_PATH": s3_resume_save_path,
            "STAGE3_RESUME_PATH": s3_resume_path,
            # 数据与分桶
            "SCIENCEQA_SPLIT": s3_split,
            "TRAIN_NUM": s3_train_num,
            "SCIENCEQA_SEED": s3_seed,
            "BUCKET_BY": s3_bucket_by,
            "BUCKET_BATCH_SIZE": s3_bucket_bs,
            "STAGE3_BUCKET_BATCH_SIZE": s3_stage3_bucket_bs,
            "BUCKET_DROP_LAST": int(s3_bucket_drop),
            "DISABLE_BUCKET_FOR_STAGE3": int(s3_disable_bucket),
            # 训练结构
            "EPOCHS": s3_epochs,
            "SUB_STAGE_NUM": s3_sub_stage_num,
            "PERIOD_NUM": s3_period_num,
            "SUB_SET_NUM": s3_sub_set_num,
            "BATCH_SIZE": s3_batch_size,
            "GRAD_ACCUM": s3_grad_accum,
            "STAGE2_GRAD_ACCUM": s3_stage2_grad_accum,
            "STAGE3_GRAD_ACCUM": s3_stage3_grad_accum,
            # 学习率与采样
            "LR": s3_lr,
            "STAGE3_LR_SCALE": s3_lr_scale,
            "STAGE3_GRAD_CLIP": s3_grad_clip,
            "STAGE3_TRAIN_PROJECTOR": int(s3_train_projector),
            "TEMPERATURE": s3_temperature,
            "MAX_NEW_TOKENS": s3_max_new_tokens,
            "MAX_LENGTH": s3_max_length,
            "TAU1": s3_tau1,
            "TAU_DELTA": s3_tau_delta,
            # Accuracy-First 损失
            "STAGE3_MC_WEIGHT": s3_mc_weight,
            "STAGE3_OBJ_WEIGHT": s3_obj_weight,
            "STAGE3_REG_WEIGHT": s3_reg_weight,
            "STAGE3_ANSWER_ANCHOR_WEIGHT": s3_answer_anchor,
            "STAGE3_FORCE_COLD_START_PERIOD0": int(s3_force_cold_period0),
            # 四字段权重
            "STAGE3_FIELD_WEIGHT_OBSERVED": s3_fw_observed,
            "STAGE3_FIELD_WEIGHT_CONTEXT": s3_fw_context,
            "STAGE3_FIELD_WEIGHT_REASONING": s3_fw_reasoning,
            "STAGE3_FIELD_WEIGHT_ANSWER": s3_fw_answer,
            # 排序与负样本
            "STAGE3_PAIR_USE_ANSWER_CORRECTNESS": int(s3_pair_correctness),
            "STAGE3_VIC_INCLUDE_CONTEXT": int(s3_vic_include_context),
            "STAGE3_WRONG_IMAGE_ENABLE": int(s3_wrong_image_enable),
            "STAGE3_WRONG_IMAGE_WEIGHT": s3_wrong_image_weight,
            "STAGE3_WRONG_IMAGE_MARGIN": s3_wrong_image_margin,
            # 教师缓存
            "TEACHER_LANG": s3_teacher_lang,
            "COLLECT_TEACHER_DATA": int(s3_collect_teacher),
            "STRICT_TEACHER_DISTILL": int(s3_strict_distill),
            "TEACHER_CACHE_PATH": s3_teacher_cache_path,
            "TEACHER_OBSERVED_MAX_TOKENS": s3_tc_obs,
            "TEACHER_CONTEXT_MAX_TOKENS": s3_tc_ctx,
            "TEACHER_REASONING_MAX_TOKENS": s3_tc_rsn,
            "TEACHER_ANSWER_MAX_TOKENS": s3_tc_ans,
            "TEACHER_MAX_NEW_TOKENS_TOTAL": s3_tc_total,
            # Stage1/2 继承
            "STAGE1_LR": s3_s1_lr,
            "STAGE1_RECON_WEIGHT": s3_s1_recon,
            "STAGE1_COSINE_WEIGHT": s3_s1_cosine,
            "STAGE1_VQ_WEIGHT": s3_s1_vq,
            "STAGE1_GRAD_CLIP": s3_s1_grad_clip,
            "STAGE2_ANSWER_WEIGHT": s3_s2_ans_w,
            "STAGE2_RATIONALE_WEIGHT": s3_s2_rat_w,
            "STAGE2_PREPOST_LR_SCALE": s3_s2_prepost_lr,
            "STAGE2_VISION_LR_SCALE": s3_s2_vis_lr,
            "STAGE2_GRAD_CLIP": s3_s2_grad_clip,
            # VQ / 模型
            "VQ_CODEBOOK_SIZE": s3_codebook_size,
            "VQ_COMMITMENT_COST": s3_commitment_cost,
            "VQ_DEAD_CODE_THRESHOLD": s3_dead_threshold,
            "VQ_USAGE_DECAY": s3_usage_decay,
            "VQ_DEAD_CODE_RESET_INTERVAL": s3_reset_interval,
            "VQ_LEGACY_LOSS": int(s3_legacy_loss),
            "FREEZE_VISION_TOWER": int(s3_freeze_vision),
            "BETA": s3_beta,
            "USE_LORA": int(s3_use_lora),
            "LORA_RANK": s3_lora_rank,
            "LORA_ALPHA": s3_lora_alpha,
            "USE_4BIT": int(s3_use_4bit),
            "MODEL_DTYPE": s3_model_dtype,
            "REUSE_VQ_CODEBOOK": int(s3_reuse_codebook),
            "REUSE_STAGE2": int(s3_reuse_stage2),
            # Stage3 内置评测
            "STAGE3_EVAL_EVERY_PERIOD": s3_eval_every,
            "STAGE3_EVAL_MAX_SAMPLES": s3_eval_max,
            "STAGE3_EVAL_SCIENCEQA_SPLIT": s3_eval_split,
            "STAGE3_EVAL_SCIENCEQA_PATH": s3_eval_path,
            "STAGE3_EVAL_TRAIN_NUM": s3_eval_train_num,
            "STAGE3_EVAL_ANSWER_MODE": s3_eval_answer_mode,
            # 训练后评测
            "EVAL_MAX_SAMPLES": s3_final_eval_max,
            "EVAL_MAX_NEW_TOKENS": s3_final_eval_tokens,
            "EVAL_ANSWER_MODE": s3_final_eval_mode,
            # 日志与保存
            "LOG_STEP": s3_log_step,
            "SAVE_STEP": s3_save_step,
            "SAVE_EACH_EPOCH": int(s3_save_each_epoch),
            "STAGE3_RESUME_SAVE_OPTIMIZER": int(s3_resume_save_optimizer),
            "STAGE3_RESUME_SAVE_INTERVAL": s3_resume_save_interval,
        }
        try:
            resp = requests.post(f"{BASE_URL}/api/task/stage3", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("✅ Stage3 训练任务已提交后台执行")
            else:
                st.error(f"❌ 提交失败: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"❌ 请求异常: {e}")

with tab_eval:
    st.header("5. 在线评测大盘")
    st.info("对 Stage2 / Stage3 训练产物在 ScienceQA test set 上进行准确率评测。"
            "评测完成后返回 ACCURACY / FORMAT_RATE / N 三个指标。")

    eval_tab_s2, eval_tab_s3 = st.tabs(["Stage2 评测", "Stage3 评测"])

    # ============================================================
    # Stage2 评测
    # ============================================================
    with eval_tab_s2:
        st.subheader("Stage2 产物评测")

        # 路径探测
        if st.button("🔍 探测 Stage2 checkpoint 目录", key="eval_s2_detect"):
            try:
                resp = requests.get(
                    f"{BASE_URL}/api/list_ckpt_dirs",
                    params={"root_dir": glob_root_dir, "prefix": "stage2_vision"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    dirs = resp.json().get("dirs", [])
                    if dirs:
                        for d in dirs:
                            a_icon = "✅" if d["has_adapter"] else "❌"
                            cb_icon = "✅" if d["has_codebook"] else "❌"
                            p_icon = "✅" if d["has_projector"] else "❌"
                            st.text(f"  {a_icon} adapter {p_icon} projector {cb_icon} codebook | {d['name']} → {d['path']}")
                    else:
                        st.warning("未找到 stage2_vision* 目录")
                else:
                    st.error(f"HTTP {resp.status_code}")
            except Exception as e:
                st.error(f"探测失败: {e}")

        ev2_col1, ev2_col2 = st.columns(2)
        with ev2_col1:
            ev2_ckpt_path = st.text_input(
                "STAGE2_CKPT_PATH (评测 checkpoint 目录)",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage2_vision",
                key="ev2_ckpt_path",
                help="实际目录可能是 stage2_vision_epoch3 等，请先探测确认"
            )
            ev2_split = st.selectbox("EVAL_SPLIT", ["test", "validation"], index=0, key="ev2_split")
            ev2_max_samples = st.number_input("EVAL_MAX_SAMPLES (0=全量)", 0, 10000, 0, key="ev2_max")
        with ev2_col2:
            ev2_max_new_tokens = st.number_input("EVAL_MAX_NEW_TOKENS", 16, 1024, 128, key="ev2_tokens")
            ev2_answer_mode = st.selectbox("EVAL_ANSWER_MODE",
                                           ["logits", "generate", "hybrid"],
                                           index=0, key="ev2_mode")
            ev2_use_vq = st.checkbox("USE_VQ", value=True, key="ev2_use_vq")
            ev2_use_4bit = st.checkbox("USE_4BIT", value=False, key="ev2_4bit")
            ev2_codebook_size = st.selectbox("VQ_CODEBOOK_SIZE",
                                             [256, 512, 1024, 2048, 4096, 8192],
                                             index=2, key="ev2_cb_size")

        if st.button("🚀 启动 Stage2 评测", use_container_width=True, key="btn_eval_s2"):
            payload = {
                **GLOBAL_ENV,
                "STAGE2_CKPT_PATH": ev2_ckpt_path,
                "EVAL_SPLIT": ev2_split,
                "EVAL_MAX_SAMPLES": ev2_max_samples,
                "EVAL_MAX_NEW_TOKENS": ev2_max_new_tokens,
                "EVAL_ANSWER_MODE": ev2_answer_mode,
                "USE_VQ": int(ev2_use_vq),
                "USE_4BIT": int(ev2_use_4bit),
                "VQ_CODEBOOK_SIZE": ev2_codebook_size,
                "FREEZE_VISION_TOWER": 0,
            }
            try:
                resp = requests.post(f"{BASE_URL}/api/task/eval_stage2", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.success("✅ Stage2 评测任务已提交后台")
                else:
                    st.error(f"❌ HTTP {resp.status_code} — {resp.text}")
            except Exception as e:
                st.error(f"❌ 请求异常: {e}")

        # 结果拉取
        st.divider()
        st.markdown("**📊 Stage2 评测结果**")
        if st.button("拉取最新 Stage2 评测结果", key="btn_fetch_ev2"):
            try:
                resp = requests.get(
                    f"{BASE_URL}/api/eval_result",
                    params={
                        "root_dir": glob_root_dir,
                        "prefix": "stage2",
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("found"):
                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.metric("ACCURACY", f"{data['accuracy']:.4f}")
                        with res_col2:
                            st.metric("FORMAT_RATE", f"{data['format_rate']:.4f}")
                        with res_col3:
                            st.metric("N (样本数)", data["n"])
                        st.caption(f"结果文件: {data['result_file']}")
                    else:
                        st.info(data.get("message", "暂无结果文件"))
                else:
                    st.warning(f"HTTP {resp.status_code}")
            except Exception as e:
                st.error(f"❌ {e}")

    # ============================================================
    # Stage3 评测
    # ============================================================
    with eval_tab_s3:
        st.subheader("Stage3 产物评测")

        # 路径探测
        if st.button("🔍 探测 Stage3 checkpoint 目录", key="eval_s3_detect"):
            try:
                found_any = False
                for prefix in ["stage3_sub", "stage3_lord", "stage3_resume"]:
                    resp = requests.get(
                        f"{BASE_URL}/api/list_ckpt_dirs",
                        params={"root_dir": glob_root_dir, "prefix": prefix},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        dirs = resp.json().get("dirs", [])
                        for d in dirs:
                            found_any = True
                            a_icon = "✅" if d["has_adapter"] else "❌"
                            cb_icon = "✅" if d["has_codebook"] else "❌"
                            st.text(f"  {a_icon} adapter {cb_icon} codebook | {d['name']} → {d['path']}")
                if not found_any:
                    st.warning("未找到任何 Stage3 产物目录")
            except Exception as e:
                st.error(f"探测失败: {e}")

        ev3_col1, ev3_col2 = st.columns(2)
        with ev3_col1:
            ev3_adapter_path = st.text_input(
                "STAGE3_FINAL_ADAPTER_PATH (评测 adapter 目录)",
                value=f"{glob_root_dir}/vq_lord_ckpts/stage3_sub1_period7",
                key="ev3_adapter_path",
                help="需要包含 adapter_config.json 和 vq_codebook.pt"
            )
            ev3_split = st.selectbox("EVAL_SPLIT", ["test", "validation"], index=0, key="ev3_split")
            ev3_max_samples = st.number_input("EVAL_MAX_SAMPLES (0=全量)", 0, 10000, 0, key="ev3_max")
        with ev3_col2:
            ev3_max_new_tokens = st.number_input("EVAL_MAX_NEW_TOKENS", 16, 1024, 128, key="ev3_tokens")
            ev3_answer_mode = st.selectbox("EVAL_ANSWER_MODE",
                                           ["logits", "generate", "hybrid"],
                                           index=0, key="ev3_mode")
            ev3_use_vq = st.checkbox("USE_VQ", value=True, key="ev3_use_vq")
            ev3_use_4bit = st.checkbox("USE_4BIT", value=False, key="ev3_4bit")
            ev3_codebook_size = st.selectbox("VQ_CODEBOOK_SIZE",
                                             [256, 512, 1024, 2048, 4096, 8192],
                                             index=2, key="ev3_cb_size")

        if st.button("🚀 启动 Stage3 评测", use_container_width=True, key="btn_eval_s3"):
            payload = {
                **GLOBAL_ENV,
                "STAGE3_FINAL_ADAPTER_PATH": ev3_adapter_path,
                "EVAL_SPLIT": ev3_split,
                "EVAL_MAX_SAMPLES": ev3_max_samples,
                "EVAL_MAX_NEW_TOKENS": ev3_max_new_tokens,
                "EVAL_ANSWER_MODE": ev3_answer_mode,
                "USE_VQ": int(ev3_use_vq),
                "USE_4BIT": int(ev3_use_4bit),
                "VQ_CODEBOOK_SIZE": ev3_codebook_size,
                "FREEZE_VISION_TOWER": 0,
            }
            try:
                resp = requests.post(f"{BASE_URL}/api/task/eval_stage3", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.success("✅ Stage3 评测任务已提交后台")
                else:
                    st.error(f"❌ HTTP {resp.status_code} — {resp.text}")
            except Exception as e:
                st.error(f"❌ 请求异常: {e}")

        # 结果拉取
        st.divider()
        st.markdown("**📊 Stage3 评测结果**")
        if st.button("拉取最新 Stage3 评测结果", key="btn_fetch_ev3"):
            try:
                resp = requests.get(
                    f"{BASE_URL}/api/eval_result",
                    params={
                        "root_dir": glob_root_dir,
                        "prefix": "stage3",
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("found"):
                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.metric("ACCURACY", f"{data['accuracy']:.4f}")
                        with res_col2:
                            st.metric("FORMAT_RATE", f"{data['format_rate']:.4f}")
                        with res_col3:
                            st.metric("N (样本数)", data["n"])
                        st.caption(f"结果文件: {data['result_file']}")
                    else:
                        st.info(data.get("message", "暂无结果文件"))
                else:
                    st.warning(f"HTTP {resp.status_code}")
            except Exception as e:
                st.error(f"❌ {e}")

with tab_info:
    st.subheader("📋 任务运行状态")
    if st.button("刷新任务状态", key="btn_task_status"):
        try:
            resp = requests.get(f"{BASE_URL}/api/tasks", timeout=5)
            if resp.status_code == 200:
                tasks = resp.json()
                if tasks:
                    for name, info in tasks.items():
                        status = info.get("status", "unknown")
                        icon = {"running": "🔄", "success": "✅", "failed": "❌"}.get(status, "❓")
                        elapsed = ""
                        if info.get("start_time"):
                            end = info.get("end_time") or time.time()
                            elapsed = f" ({int(end - info['start_time'])}s)"
                        st.text(f"{icon} {name}: {status}{elapsed}")
                else:
                    st.info("暂无任务记录")
            else:
                st.warning(f"HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    st.divider()

    # ---- 日志面板 ----
    st.subheader("📜 系统后台日志")
    log_tail_lines = st.slider("显示最后 N 行", 20, 200, 80, key="log_tail")
    if st.button("刷新最新日志", key="btn_log"):
        try:
            resp = requests.get(
                f"{BASE_URL}/api/logs/latest",
                params={
                    "root_dir": glob_root_dir,
                    "subdir": "logs/apis",
                    "tail": log_tail_lines,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                log_file = data.get("log_file", "")
                log_text = data.get("log_text", "No logs yet.")
                st.caption(f"日志文件: {log_file}")
                st.code(log_text, language="bash")
            else:
                st.warning(f"HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")