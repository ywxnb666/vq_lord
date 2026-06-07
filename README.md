# MICAD

MICAD is a multimodal training and evaluation workspace centered on the `VQ-LoRD` pipeline. The repository combines:

- a three-stage visual-token / distillation / alignment training flow for vision-language models,
- teacher-data collection and preprocessing utilities,
- benchmark evaluation scripts for datasets such as ScienceQA, A-OKVQA, IconQA, and TextVQA,
- and a lightweight web control panel built with Streamlit + FastAPI for launching long-running jobs.

The codebase is research-oriented rather than packaged as a polished library. Most workflows are driven through shell scripts in `scripts2/`, which wire together model paths, datasets, checkpoints, and stage-specific hyperparameters.

## What This Repository Contains

At a high level, the training flow is:

1. Stage 0: train or reuse a VQ codebook on top of the student model's visual pathway.
2. Stage 1: distill structured teacher annotations into the student model's vision stack.
3. Stage 2: run LoRD-style preference/alignment training with cached teacher signals and periodic evaluation.

Alongside that core pipeline, the repo also includes:

- dataset preprocessing for bucketed multimodal batches,
- teacher annotation sampling with OpenAI-compatible, Qwen-compatible, and Gemini-compatible APIs,
- evaluation scripts for ScienceQA and related benchmarks,
- a browser-based control surface to trigger shell jobs and inspect task state.

## Repository Layout

```text
MICAD/
|-- fastapi_vqlord/         # Streamlit UI + FastAPI backend for launching jobs
|-- scripts2/               # Main shell entrypoints for preprocessing, training, and evaluation
|-- vq_lord3/
|   |-- data/               # Dataset loading, teacher data collection, dataset adapters
|   |-- data/preprocess/    # ScienceQA preprocessing / bucketing utilities
|   |-- evaluation/         # Evaluation runners and result post-processing
|   |-- experiments/        # Small runnable experiment entrypoints
|   |-- models/             # VQ modules, student model wrappers, loss definitions
|   |-- sample/             # Portable teacher sampler package and configs
|   |-- training/           # Main training implementation
|   `-- utils/              # Runtime verification and cache utilities
|-- vq_lord_test_results/   # Output directory for evaluation JSON artifacts
|-- re.txt                  # Python dependency snapshot used by this project
`-- README.md
```

## Core Components

### 1. Training pipeline

The primary training entrypoint is:

- `vq_lord3/training/train_vq_lord.py`

This script loads a student vision-language model, attaches the VQ module, prepares multimodal training data, and dispatches the requested stage:

- `--stage 0`: visual codebook / VQ training
- `--stage 1`: structured teacher distillation into the visual stack
- `--stage 2`: LoRD-style alignment and preference optimization

The implementation includes support for:

- LoRA-based fine-tuning,
- optional 4-bit loading,
- DDP / `torchrun` multi-GPU training,
- checkpoint reuse between stages,
- bucketed batching for multimodal datasets,
- multiple dataset modes such as `scienceqa`, `aokvqa`, and `iconqa`.

### 2. Shell-based orchestration

Most day-to-day usage happens through the scripts in `scripts2/`:

- `data_preprocess.sh`: build bucketed preprocessing metadata
- `teacher_model_data_collect.sh`: collect teacher annotations
- `run_stage0.sh`: train the VQ codebook stage
- `run_stage1.sh`: train the visual distillation stage
- `run_stage2.sh`: train the alignment stage
- `test_*.sh` and `eval_*.sh`: run evaluations and full benchmark pipelines

`scripts2/common.sh` is the central environment bootstrapper. It defines default paths, dataset profiles, distributed-launch behavior, cache locations, and log/checkpoint directories.

### 3. Teacher sampling

The portable sampler in `vq_lord3/sample/` is a cleaner, self-contained package for collecting teacher outputs. It supports multiple provider styles and keeps a training-compatible cache schema.

Useful files:

- `vq_lord3/sample/README.md`
- `vq_lord3/sample/sampler/cli.py`
- `vq_lord3/sample/configs/*.json`

This part is helpful if you want to generate or validate teacher caches without running the full training pipeline.

### 4. Web control panel

`fastapi_vqlord/` provides a simple experiment-control interface:

- `app.py`: Streamlit frontend with forms for preprocessing, Stage 0, Stage 1, Stage 2, and evaluation
- `backend.py`: FastAPI service that launches shell scripts asynchronously and tracks task state

The backend exposes endpoints such as:

- `/api/task/collect`
- `/api/task/preprocess`
- `/api/task/stage0`
- `/api/task/stage1`
- `/api/task/stage2`
- `/api/task/eval_stage1`
- `/api/task/eval_stage2`

This is especially useful when the code is being run on a remote Linux GPU box and you want a lightweight control surface instead of manually editing environment variables for every run.

## Supported Data / Benchmarks

From the current code and scripts, the project is set up around these datasets:

- ScienceQA
- A-OKVQA
- IconQA
- TextVQA

The dataset profile logic in `scripts2/common.sh` currently provides first-class defaults for:

- `scienceqa`
- `aokvqa`
- `iconqa`

Different scripts may assume local dataset mirrors or cached artifacts already exist.

## Environment and Dependencies

The repository is designed for a Linux training environment with CUDA-enabled GPUs. The checked-in scripts assume:

- Bash is available
- PyTorch with CUDA is installed
- Hugging Face model/dataset caches are available locally or through a mirror
- `torchrun` is available for multi-GPU stages
- large vision-language checkpoints are already downloaded

The top-level `re.txt` records the core Python stack used by the project, including:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `trl`
- `openai`
- `tensorboard`

For the web UI, there is an additional frontend dependency file:

- `fastapi_vqlord/requirements_frontend.txt`

## Typical Workflow

### Option A: Run from shell scripts

1. Set local paths through environment variables such as `ROOT_DIR`, `PYTHON_BIN`, `MODEL_PATH`, and dataset paths.
2. Preprocess the dataset:

```bash
bash scripts2/data_preprocess.sh
```

3. Run Stage 0:

```bash
bash scripts2/run_stage0.sh
```

4. Run Stage 1:

```bash
bash scripts2/run_stage1.sh
```

5. Run Stage 2:

```bash
bash scripts2/run_stage2.sh
```

6. Evaluate using the relevant `test_*.sh` or `eval_*.sh` script.

### Option B: Use the control panel

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Fill in your runtime paths, model path, API keys, and dataset settings.
4. Launch preprocessing / training / evaluation jobs from the browser.

The exact startup commands depend on your Python environment, but conceptually the split is:

```bash
python fastapi_vqlord/backend.py
streamlit run fastapi_vqlord/app.py
```

If you use this path, verify the backend serving method first, since some environments may prefer launching the FastAPI app through `uvicorn`.

## Notes for Contributors

- The repository currently contains many hard-coded defaults for the author's training environment. Expect to adapt paths before running anything.
- The codebase is optimized for experimentation and iteration, not for pip installation.
- Checkpoint and cache reuse are important across stages; deleting intermediate artifacts can break downstream scripts.
- Some scripts are tuned for multi-GPU machines and assume `CUDA_VISIBLE_DEVICES` and `DDP_NPROC` are set consistently.
- The `sample/` subpackage is the cleanest entrypoint if your goal is only teacher-data collection.

## Suggested First Improvements

If you plan to keep contributing to this repository, the highest-impact next steps would likely be:

- replace machine-specific defaults in `scripts2/common.sh` with documented `.env` or config files,
- add a single reproducible setup guide for Linux environments,
- document expected checkpoint/output directory structures,
- clarify the relationship between MICAD and the `VQ-LoRD` naming used throughout the code,
- add a minimal end-to-end example with a tiny subset of one dataset.

## Status

The previous root `README.md` was empty. This version is intended to give new collaborators enough context to understand the repository structure and start navigating the training pipeline safely.
