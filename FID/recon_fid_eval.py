#!/usr/bin/env python3
"""
Independent reconstruction-FID evaluation for VQGAN (taming-transformers).

Pipeline (strict reconstruction path):
real image -> encoder -> quantize(codebook lookup) -> decoder -> reconstructed image
Then compute FID(real_set, recon_set) with torch-fidelity or pytorch-fid.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# Set valid OMP value before importing torch to avoid libgomp warnings.
_omp = os.environ.get("OMP_NUM_THREADS", "")
if not _omp.isdigit() or int(_omp) <= 0:
    os.environ["OMP_NUM_THREADS"] = "8"

import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_omp_env() -> None:
    # Avoid libgomp invalid OMP_NUM_THREADS warnings.
    v = os.environ.get("OMP_NUM_THREADS", "")
    if not v.isdigit() or int(v) <= 0:
        os.environ["OMP_NUM_THREADS"] = "8"


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _letterbox(img: Image.Image, size: int, fill: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Aspect-preserving resize + padding (default white background for chart-like images)."""
    img = _to_rgb(img)
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {img.size}")

    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), resample=Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (size, size), color=fill)
    off_x = (size - nw) // 2
    off_y = (size - nh) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    img = _to_rgb(img)
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {img.size}")
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    return img.resize((size, size), resample=Image.Resampling.BICUBIC)


def _stretch_resize(img: Image.Image, size: int) -> Image.Image:
    return _to_rgb(img).resize((size, size), resample=Image.Resampling.BICUBIC)


def _preprocess_image(img: Image.Image, size: int, mode: str) -> Image.Image:
    if mode == "letterbox":
        return _letterbox(img, size)
    if mode == "crop":
        return _center_crop_resize(img, size)
    if mode == "stretch":
        return _stretch_resize(img, size)
    raise ValueError(f"unsupported preprocess mode: {mode}")


def _pil_to_model_tensor(img: Image.Image) -> torch.Tensor:
    """RGB PIL [0,255] -> float tensor in [-1,1], shape [3,H,W]."""
    arr = np.asarray(_to_rgb(img), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    t = torch.from_numpy(arr)
    return t * 2.0 - 1.0


def _model_tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Tensor in [-1,1], shape [3,H,W] -> uint8 HWC RGB."""
    x = t.detach().float().cpu().clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().clamp(0, 255).byte().numpy()
    x = np.transpose(x, (1, 2, 0))
    return x


def _save_uint8_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


@dataclass
class Sample:
    idx: int
    image: Image.Image


class FolderImageSource:
    def __init__(self, root: str, max_samples: int, seed: int):
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"images_dir not found: {root}")
        files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files = sorted(files)
        if not files:
            raise RuntimeError(f"no images found under: {root}")

        if max_samples > 0 and len(files) > max_samples:
            rng = random.Random(seed)
            files = rng.sample(files, max_samples)
            files = sorted(files)

        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterable[Sample]:
        for i, p in enumerate(self.files):
            img = Image.open(p)
            yield Sample(idx=i, image=img)


class ScienceQAImageSource:
    def __init__(self, scienceqa_path: str, split: str, max_samples: int, seed: int):
        from datasets import load_dataset

        ds = load_dataset(scienceqa_path, split=split)
        indices = [i for i, row in enumerate(ds) if row.get("image") is not None]
        if not indices:
            raise RuntimeError(f"ScienceQA split has no images: {scienceqa_path}::{split}")

        if max_samples > 0 and len(indices) > max_samples:
            rng = random.Random(seed)
            indices = rng.sample(indices, max_samples)

        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterable[Sample]:
        for out_i, ds_i in enumerate(self.indices):
            row = self.ds[ds_i]
            img = row["image"]
            yield Sample(idx=out_i, image=img)


def _add_taming_to_path(taming_root: str) -> None:
    root = Path(taming_root)
    if not root.exists():
        raise FileNotFoundError(f"taming_root not found: {taming_root}")
    sys.path.insert(0, str(root))


def _resolve_model_size(cfg_obj, user_size: int) -> int:
    if user_size > 0:
        return int(user_size)

    # Try common locations in taming config.
    try:
        size = int(cfg_obj.model.params.ddconfig.resolution)
        if size > 0:
            return size
    except Exception:
        pass
    return 256


def _load_taming_model(config_path: str, ckpt_path: str, device: str, taming_root: str):
    _add_taming_to_path(taming_root)
    from omegaconf import OmegaConf
    from main import instantiate_from_config

    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"unsupported checkpoint format: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Model] loaded ckpt: {ckpt_path}")
    print(f"[Model] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model, cfg


def _extract_codebook_tensor(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in [
            "quantize.embedding.weight",
            "codebook",
            "embedding.weight",
            "vq.embedding.weight",
        ]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    raise RuntimeError("cannot find codebook tensor in provided file")


def _inject_codebook(model, codebook_path: str) -> None:
    if not hasattr(model, "quantize"):
        raise RuntimeError("model has no quantize module")
    if not hasattr(model.quantize, "embedding"):
        raise RuntimeError("model.quantize has no embedding")

    target = model.quantize.embedding.weight
    loaded = torch.load(codebook_path, map_location="cpu")
    src = _extract_codebook_tensor(loaded).detach().cpu()

    if tuple(src.shape) != tuple(target.shape):
        raise RuntimeError(
            f"codebook shape mismatch: loaded={tuple(src.shape)} vs model={tuple(target.shape)}"
        )

    with torch.no_grad():
        target.copy_(src.to(device=target.device, dtype=target.dtype))

    print(f"[Codebook] injected from: {codebook_path}, shape={tuple(src.shape)}")


def _compute_fid(real_dir: Path, recon_dir: Path, device: str, batch_size: int, backend: str) -> Tuple[str, float]:
    backends = [backend] if backend != "auto" else ["torch-fidelity", "pytorch-fid"]

    last_err = None
    for b in backends:
        try:
            if b == "torch-fidelity":
                import torch_fidelity

                metrics = torch_fidelity.calculate_metrics(
                    input1=str(real_dir),
                    input2=str(recon_dir),
                    fid=True,
                    isc=False,
                    kid=False,
                    prc=False,
                    cuda=device.startswith("cuda"),
                    batch_size=batch_size,
                    verbose=False,
                )
                return b, float(metrics["frechet_inception_distance"])

            if b == "pytorch-fid":
                from pytorch_fid.fid_score import calculate_fid_given_paths

                score = calculate_fid_given_paths(
                    paths=[str(real_dir), str(recon_dir)],
                    batch_size=batch_size,
                    device=device,
                    dims=2048,
                    num_workers=0,
                )
                return b, float(score)

            raise ValueError(f"unknown backend: {b}")
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"failed to compute FID with backend={backend}: {repr(last_err)}")


def _batched(iterable: Iterable[Sample], batch_size: int) -> Iterable[List[Sample]]:
    batch: List[Sample] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruction FID evaluation for taming VQGAN")

    parser.add_argument("--taming-root", type=str, default="/root/workspace/taming-transformers")
    parser.add_argument("--config", type=str, required=True, help="taming VQGAN config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="taming VQGAN checkpoint (.ckpt)")
    parser.add_argument(
        "--codebook-path",
        type=str,
        default="/root/autodl-tmp/vq_lord_ckpts/stage1_vq/vq_codebook.pt",
        help="optional codebook tensor file to inject before evaluation",
    )
    parser.add_argument("--skip-codebook-inject", action="store_true")

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--images-dir", type=str, default="", help="evaluate images from local folder")
    src.add_argument("--scienceqa-path", type=str, default="/root/autodl-tmp/datasets/ScienceQA")
    parser.add_argument("--scienceqa-split", type=str, default="validation")

    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240306)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--target-size", type=int, default=0, help="0 means auto from config")
    parser.add_argument("--preprocess-mode", choices=["letterbox", "crop", "stretch"], default="letterbox")

    parser.add_argument("--fid-backend", choices=["auto", "torch-fidelity", "pytorch-fid"], default="auto")
    parser.add_argument("--output-root", type=str, default="/root/workspace/align_vq/FID/runs")
    parser.add_argument("--keep-images", action="store_true", help="keep real/recon image folders after FID")

    return parser.parse_args()


def main() -> None:
    _ensure_omp_env()
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    out_root = Path(args.output_root)
    run_dir = out_root / f"recon_fid_{_timestamp()}"
    real_dir = run_dir / "real"
    recon_dir = run_dir / "recon"
    run_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Run] output dir: {run_dir}")

    model, cfg = _load_taming_model(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=args.device,
        taming_root=args.taming_root,
    )

    target_size = _resolve_model_size(cfg, args.target_size)
    print(f"[Run] target_size={target_size}, preprocess_mode={args.preprocess_mode}")

    if not args.skip_codebook_inject and args.codebook_path:
        _inject_codebook(model, args.codebook_path)

    if args.images_dir:
        source = FolderImageSource(root=args.images_dir, max_samples=args.max_samples, seed=args.seed)
        source_desc = f"folder:{args.images_dir}"
    else:
        source = ScienceQAImageSource(
            scienceqa_path=args.scienceqa_path,
            split=args.scienceqa_split,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        source_desc = f"scienceqa:{args.scienceqa_path}::{args.scienceqa_split}"

    n = len(source)
    if n < 128:
        print(f"[Warn] sample count={n} is small for stable FID; use >= 1k when possible.")
    print(f"[Run] sample count={n} from {source_desc}")

    # Reconstruction pipeline with explicit encoder->quantize->decoder path.
    with torch.no_grad():
        for batch in tqdm(_batched(iter(source), args.batch_size), total=math.ceil(n / args.batch_size), desc="Reconstruct"):
            model_inputs = []
            sample_ids = []

            for sample in batch:
                prep = _preprocess_image(sample.image, size=target_size, mode=args.preprocess_mode)
                real_np = np.asarray(_to_rgb(prep), dtype=np.uint8)
                _save_uint8_rgb(real_dir / f"{sample.idx:07d}.png", real_np)

                model_inputs.append(_pil_to_model_tensor(prep))
                sample_ids.append(sample.idx)

            x = torch.stack(model_inputs, dim=0).to(args.device)
            h = model.encoder(x)
            h = model.quant_conv(h)
            quant, _, _ = model.quantize(h)
            dec = model.decode(quant)

            for i, sid in enumerate(sample_ids):
                rec_np = _model_tensor_to_uint8(dec[i])
                _save_uint8_rgb(recon_dir / f"{sid:07d}.png", rec_np)

    backend_used, fid_score = _compute_fid(
        real_dir=real_dir,
        recon_dir=recon_dir,
        device=args.device,
        batch_size=args.fid_batch_size,
        backend=args.fid_backend,
    )

    report = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "source": source_desc,
        "num_samples": n,
        "target_size": target_size,
        "preprocess_mode": args.preprocess_mode,
        "config": args.config,
        "ckpt": args.ckpt,
        "codebook_path": None if args.skip_codebook_inject else args.codebook_path,
        "fid_backend": backend_used,
        "fid": fid_score,
        "imagenet_vqgan_reference": "best reconstruction FID is typically around 4.9~8.0",
    }

    report_path = run_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"FID ({backend_used}): {fid_score:.6f}")
    print(f"report: {report_path}")
    print("=" * 70)

    if not args.keep_images:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(recon_dir, ignore_errors=True)
        print("[Run] removed temporary real/recon image folders (use --keep-images to keep them)")


if __name__ == "__main__":
    main()
