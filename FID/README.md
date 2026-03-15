# Reconstruction FID Evaluation (VQGAN)

This folder contains an **independent** script to evaluate reconstruction FID with the strict path:

`real image -> encoder -> quantize -> decoder -> reconstructed image`

## Script
- `recon_fid_eval.py`

## Backends
- `torch-fidelity` (preferred if installed)
- fallback: `pytorch-fid`

## Example
```bash
/root/autodl-tmp/conda/envs/align_vq/bin/python3 /root/workspace/align_vq/FID/recon_fid_eval.py \
  --taming-root /root/workspace/taming-transformers \
  --config /root/workspace/taming-transformers/configs/custom_vqgan.yaml \
  --ckpt /path/to/vqgan.ckpt \
  --codebook-path /root/autodl-tmp/vq_lord_ckpts/stage1_vq/vq_codebook.pt \
  --scienceqa-path /root/autodl-tmp/datasets/ScienceQA \
  --scienceqa-split validation \
  --max-samples 2000 \
  --preprocess-mode letterbox \
  --fid-backend auto \
  --keep-images
```

## Important Notes
- The script checks codebook shape before injection.
- If your codebook shape does not match the VQGAN quantizer shape, it will fail fast with a clear error.
- For chart/white-background images, use `--preprocess-mode letterbox` to avoid geometric/text distortion from stretching.
- FID is sample-size sensitive; prefer large subsets (e.g. 1k~10k+ if available).
