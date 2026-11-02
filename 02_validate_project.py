#!/usr/bin/env python3
"""Validate CurricularFace environment, data, model, and optionally resume logic.

Quick validation:
    !python 02_validate_project.py

Include a two-process checkpoint/resume smoke test:
    !python 02_validate_project.py --checkpoint-test

The checkpoint test uses separate temporary folders under Drive and removes
nothing automatically.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path("/content/ABSS-CurricularFace-Implementation")
DATA_ROOT = Path("/content/CASIA-WebFace/data")
MANIFEST = DATA_ROOT / "casia_train.txt"
EXPECTED_IMAGES = 490_623
EXPECTED_IDENTITIES = 10_572


def check(name: str, fn) -> bool:
    print(f"\n[CHECK] {name}")
    try:
        result = fn()
        print(f"[PASS] {name}")
        if result is not None:
            print(result)
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        return False


def environment():
    import torch
    import torchvision
    import numpy
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("NumPy:", numpy.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))


def source_and_config():
    if not REPO.is_dir():
        raise FileNotFoundError(REPO)
    sys.path.insert(0, str(REPO))
    import config
    cfg = config.configurations[1]
    print("Backbone:", cfg["BACKBONE_NAME"])
    print("Head:", cfg["HEAD_NAME"])
    print("Batch size:", cfg["BATCH_SIZE"])
    print("Epochs:", cfg["NUM_EPOCH"])
    print("Milestones:", cfg["STAGES"])
    if cfg["BACKBONE_NAME"] != "IR_101" or cfg["HEAD_NAME"] != "CurricularFace":
        raise AssertionError("Unexpected backbone/head configuration.")
    if cfg["NUM_EPOCH"] != 50 or cfg["STAGES"] != [28, 38, 46]:
        raise AssertionError("Unexpected epoch count or LR milestones.")
    subprocess.run([sys.executable, "-m", "py_compile", str(REPO / "train.py")], check=True)
    return "Configuration and train.py syntax are valid."


def dataset_integrity():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    root = DATA_ROOT
    count = 0
    labels = set()
    missing = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed manifest line {line_no}: {line[:100]}")
            rel, label = parts
            labels.add(int(label))
            image_path = root / rel
            if not image_path.is_file() and len(missing) < 5:
                missing.append((line_no, str(image_path)))
            count += 1
    print("Manifest records:", count)
    print("Identity labels:", len(labels))
    print("Missing examples:", missing)
    if missing:
        raise FileNotFoundError(missing)
    if count != EXPECTED_IMAGES:
        print(f"WARNING: expected historical count {EXPECTED_IMAGES}, found {count}")
    if len(labels) != EXPECTED_IDENTITIES:
        print(f"WARNING: expected historical identities {EXPECTED_IDENTITIES}, found {len(labels)}")
    return "Manifest paths resolve to existing files."


def dataset_batch():
    import torch
    sys.path.insert(0, str(REPO))
    from dataset.datasets import FaceDataset
    cfg = __import__("config").configurations[1]
    ds = FaceDataset(cfg["RECORD_DIR"], cfg["DATA_ROOT"], cfg["INPUT_SIZE"])
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")
    image, label = ds[0]
    if tuple(image.shape) != (3, 112, 112):
        raise AssertionError(f"Unexpected image shape: {image.shape}")
    if not torch.isfinite(image).all():
        raise AssertionError("Image contains NaN or Inf.")
    print("Dataset length:", len(ds))
    print("Sample shape:", tuple(image.shape), "dtype:", image.dtype)
    print("Sample label:", int(label), "pixel range:", (float(image.min()), float(image.max())))
    return "Dataset sample loads and transforms correctly."


def model_forward_backward():
    import torch
    sys.path.insert(0, str(REPO))
    from backbone.model_irse import IR_101
    from head.metrics import CurricularFace
    torch.cuda.empty_cache()
    backbone = IR_101(input_size=(112, 112)).cuda().train()
    head = CurricularFace(in_features=512, out_features=10_572).cuda().train()
    batch_size = min(8, int(__import__("config").configurations[1]["BATCH_SIZE"]))
    images = torch.randn(batch_size, 3, 112, 112, device="cuda")
    labels = torch.arange(batch_size, device="cuda") % 10_572
    embeddings = backbone(images)
    logits = head(embeddings, labels)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    if embeddings.shape != (batch_size, 512):
        raise AssertionError(f"Unexpected embedding shape: {embeddings.shape}")
    if logits.shape != (batch_size, 10_572):
        raise AssertionError(f"Unexpected logits shape: {logits.shape}")
    print("Embedding:", tuple(embeddings.shape), "logits:", tuple(logits.shape))
    print("Loss:", float(loss.detach()))
    del backbone, head, images, labels, embeddings, logits, loss
    torch.cuda.empty_cache()
    return "IR-101 + CurricularFace forward/backward passed on GPU."


def checkpoint_resume_test():
    if not (REPO / "train.py").is_file():
        raise FileNotFoundError(REPO / "train.py")
    import config
    cfg = config.configurations[1]
    batch_size = cfg["BATCH_SIZE"]
    # Dedicated smoke-test output folders avoid touching production checkpoints.
    base = Path("/content/drive/MyDrive/ABSS-CurricularFace-Implementation/validation_smoke")
    model_dir, log_dir = base / "checkpoints", base / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    latest = model_dir / "latest_checkpoint.pth"
    if latest.exists():
        latest.unlink()
    common = [
        sys.executable, str(REPO / "train.py"),
        "--checkpoint-every-steps", "1",
        "--model-root", str(model_dir),
        "--log-root", str(log_dir),
    ]
    subprocess.run(common + ["--max-steps", "2"], cwd=REPO, check=True)
    if not latest.is_file():
        raise RuntimeError("First smoke run did not create latest_checkpoint.pth")
    subprocess.run(common + ["--max-steps", "3", "--resume", str(latest)], cwd=REPO, check=True)
    import torch
    state = torch.load(latest, map_location="cpu", weights_only=False)
    print("Checkpoint global_step:", state.get("global_step"))
    print("Checkpoint next_batch:", state.get("next_batch"))
    if int(state.get("global_step", -1)) != 3:
        raise AssertionError("Resume test did not advance global_step to 3.")
    return "Training checkpoint resumed in a second process."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-test", action="store_true",
                        help="Run two short training processes to verify checkpoint/resume.")
    args = parser.parse_args()
    results = []
    results.append(check("Environment and GPU", environment))
    results.append(check("Repository, config, and Python syntax", source_and_config))
    results.append(check("Dataset manifest and file integrity", dataset_integrity))
    results.append(check("Dataset loading and transform", dataset_batch))
    results.append(check("IR-101 + CurricularFace GPU forward/backward", model_forward_backward))
    if args.checkpoint_test:
        results.append(check("Checkpoint save and cross-process resume", checkpoint_resume_test))
    print("\n" + "=" * 64)
    print(f"VALIDATION RESULT: {sum(results)}/{len(results)} checks passed")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
