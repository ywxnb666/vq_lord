# Portable Teacher Sampler

This directory is a clean sampling package derived from `../teac`.

The sampling logic is intentionally preserved:

- structured teacher prompt
- strict JSON extraction and partial recovery
- four-field normalization
- semantic issue flags
- retry behavior
- training-ready resume rule
- output cache schema

The new code only reorganizes configuration, dataset loading, provider selection, validation, and command-line entry points.

## Install

```bash
cd /root/workspace/vq_lord/vq_lord3/sample
pip install -r requirements.txt
```

## Configure

Use a JSON config from `configs/` and set credentials with environment variables.

OpenAI-compatible example:

```bash
export TEACHER_MODEL="gpt-5.2"
export TEACHER_API_BASE="https://your-api-base/v1"
export TEACHER_API_KEY="your-key"
python -m sampler.cli validate-dataset --config configs/example.openai.json
python -m sampler.cli sample --config configs/example.openai.json
```

Qwen-compatible example:

```bash
export QWEN_MODEL="qwen3.5-flash-2026-02-23"
export QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_API_KEY="your-key"
export QWEN_ENABLE_THINKING="false"
python -m sampler.cli sample --config configs/example.qwen.json
```

RightCodes-style content items:

```bash
export TEACHER_MODEL="gpt-5.2"
export TEACHER_API_BASE="https://your-base/v1"
export TEACHER_API_KEY="your-key"
python -m sampler.cli sample --config configs/example.rightcodes.json
```

Gemini native `generateContent`:

```bash
export GEMINI_MODEL="gemini-2.5-pro"
export GEMINI_API_BASE="https://generativelanguage.googleapis.com"
export GEMINI_API_KEY="your-key"
python -m sampler.cli sample --config configs/example.gemini.json
```

## Override Config Values

CLI overrides are useful for one-off runs:

```bash
python -m sampler.cli sample \
  --config configs/example.openai.json \
  --dataset aokvqa \
  --dataset-path /root/autodl-tmp/datasets/A-OKVQA/data \
  --provider-type rightcodes \
  --model gpt-5.4 \
  --base-url https://your-base/v1 \
  --api-key "$TEACHER_API_KEY" \
  --output-path outputs/aokvqa_teacher_gpt-5.4_train_n8.json
```

Priority order:

1. CLI arguments
2. environment variables referenced in JSON
3. JSON config values
4. code defaults

## Dataset Sources

Supported dataset names:

- `scienceqa`
- `textvqa`
- `aokvqa`
- `iconqa` with `source_type=local_parquet`

Supported source types:

- `hf_or_parquet`: reuse the original HuggingFace/parquet loading logic from `common.py`
- `local_parquet`: load local `.parquet` files from a file or directory

Local parquet examples:

```json
{
  "name": "aokvqa",
  "source_type": "local_parquet",
  "path": "/root/autodl-tmp/datasets/A-OKVQA/data",
  "split": "train",
  "count": 6200
}
```

Aligned sampling from an existing cache:

```json
{
  "name": "aokvqa",
  "source_type": "local_parquet",
  "path": "/root/autodl-tmp/datasets/A-OKVQA/data",
  "split": "train",
  "count": 6200,
  "alignment_path": "../teac/output_qwen/aokvqa_teacher_qwen3.5-flash-2026-02-23_train_n6200_stratified.json"
}
```

## Validation

Validate dataset loading without calling the teacher API:

```bash
python -m sampler.cli validate-dataset --config configs/example.openai.json
```

Validate a generated cache:

```bash
python -m sampler.cli validate-cache --path outputs/aokvqa_teacher_gpt-5.2_train_n8.json
```

## Output Schema

The output cache keeps the old training-compatible schema:

```json
{
  "format_version": "v2",
  "dataset": "aokvqa",
  "dataset_path": "...",
  "split": "train",
  "teacher_model": "gpt-5.2",
  "victim_model": "gpt-5.2",
  "budget": {},
  "sample_count": 1,
  "samples": {
    "aokvqa::train::1001": {
      "format_version": "v2",
      "dataset": "AOKVQA",
      "sample": {},
      "meta": {},
      "raw_teacher_response": "...",
      "issues": [],
      "valid": true,
      "teacher_model": "gpt-5.2",
      "collected_at": "YYYY-MM-DD HH:MM:SS",
      "observed_facts_visual": "...",
      "context_textual": "...",
      "reasoning": "...",
      "answer": "...",
      "teacher_annotation": {
        "format_version": "v2",
        "observed_facts_visual": "...",
        "context_textual": "...",
        "reasoning": "...",
        "answer": "..."
      }
    }
  }
}
```

A sample is skipped during resume only when `valid=true` and all four required fields are present both at the record top level and inside `teacher_annotation`.
