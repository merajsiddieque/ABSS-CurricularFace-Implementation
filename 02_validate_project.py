#!/usr/bin/env python3
"""Validate CurricularFace environment, dataset, model, and checkpoint resume."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path("/content/ABSS-CurricularFace-Implementation")
DATA_ROOT = Path("/content/CASIA-WebFace/data/data")
MANIFEST = DATA_ROOT / "casia_train.txt"

EXPECTED_IMAGES = 490_623
EXPECTED_IDENTITIES = 10_572
EMBEDDING_SIZE = 512


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


def load_config():
    if not REPO.is_dir():
        raise FileNotFoundError(REPO)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import config
    return config.configurations[1]


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
    print(
        "VRAM GiB:",
        round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    )


def source_and_config():
    cfg = load_config()

    print("Backbone:", cfg["BACKBONE_NAME"])
    print("Head:", cfg["HEAD_NAME"])
    print("Batch size:", cfg["BATCH_SIZE"])
    print("Epochs:", cfg["NUM_EPOCH"])
    print("Milestones:", cfg["STAGES"])
    print("Configured DATA_ROOT:", cfg["DATA_ROOT"])
    print("Configured RECORD_DIR:", cfg["RECORD_DIR"])

    if cfg["BACKBONE_NAME"] != "IR_101":
        raise AssertionError("Expected IR_101 backbone.")

    if cfg["HEAD_NAME"] != "CurricularFace":
        raise AssertionError("Expected CurricularFace head.")

    if cfg["NUM_EPOCH"] != 50 or cfg["STAGES"] != [28, 38, 46]:
        raise AssertionError("Unexpected epoch count or LR milestones.")

    if Path(cfg["DATA_ROOT"]).resolve() != DATA_ROOT.resolve():
        raise AssertionError(
            f"Incorrect DATA_ROOT. Expected: {DATA_ROOT}"
        )

    if Path(cfg["RECORD_DIR"]).resolve() != MANIFEST.resolve():
        raise AssertionError(
            f"Incorrect RECORD_DIR. Expected: {MANIFEST}"
        )

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(REPO / "train.py")],
        check=True
    )

    return "Configuration and training script syntax are valid."


def dataset_integrity():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    count = 0
    labels = set()
    missing = []
    malformed = []

    with MANIFEST.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            parts = line.strip().rsplit(maxsplit=1)

            if len(parts) != 2:
                malformed.append(line_no)
                continue

            rel, label = parts

            try:
                label = int(label)
            except ValueError:
                malformed.append(line_no)
                continue

            labels.add(label)
            image_path = DATA_ROOT / rel

            if not image_path.is_file() and len(missing) < 5:
                missing.append((line_no, str(image_path)))

            count += 1

    print("Manifest records:", count)
    print("Identity labels:", len(labels))
    print("Malformed lines:", malformed[:5])
    print("Missing examples:", missing)

    if malformed:
        raise ValueError(f"Malformed manifest entries: {malformed[:5]}")

    if missing:
        raise FileNotFoundError(missing)

    if count != EXPECTED_IMAGES:
        print(
            f"WARNING: Expected historical count {EXPECTED_IMAGES}, "
            f"found {count}"
        )

    if len(labels) != EXPECTED_IDENTITIES:
        print(
            f"WARNING: Expected historical identities "
            f"{EXPECTED_IDENTITIES}, found {len(labels)}"
        )

    return "All manifest paths resolve to existing files."


def dataset_batch():
    import torch
    from torchvision import transforms
    from dataset.datasets import FaceDataset

    # Dataset constructor expects:
    # FaceDataset(root_dir, record_dir, transform, Train=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    ds = FaceDataset(
        str(DATA_ROOT),
        str(MANIFEST),
        transform
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    image, label = ds[0]

    if tuple(image.shape) != (3, 112, 112):
        raise AssertionError(f"Unexpected image shape: {image.shape}")

    if not torch.isfinite(image).all():
        raise AssertionError("Image contains NaN or Inf.")

    print("Dataset length:", len(ds))
    print("Sample shape:", tuple(image.shape))
    print("Sample dtype:", image.dtype)
    print("Sample label:", int(label))
    print(
        "Pixel range:",
        (float(image.min()), float(image.max()))
    )

    return "Dataset sample loads and transforms correctly."


def model_forward_backward():
    import torch
    from backbone.model_irse import IR_101
    from head.metrics import CurricularFace

    cfg = load_config()

    torch.cuda.empty_cache()

    backbone = None
    head = None

    try:
        backbone = IR_101(input_size=(112, 112)).cuda().train()

        head = CurricularFace(
            in_features=EMBEDDING_SIZE,
            out_features=EXPECTED_IDENTITIES
        ).cuda().train()

        batch_size = min(8, int(cfg["BATCH_SIZE"]))

        images = torch.randn(
            batch_size, 3, 112, 112, device="cuda"
        )

        labels = torch.arange(
            batch_size, device="cuda"
        ) % EXPECTED_IDENTITIES

        backbone_output = backbone(images)

        print(
            "Backbone output type:",
            type(backbone_output).__name__
        )

        if isinstance(backbone_output, (tuple, list)):
            print(
                "Backbone output elements:",
                [
                    tuple(x.shape) if torch.is_tensor(x)
                    else type(x).__name__
                    for x in backbone_output
                ]
            )

            # IR_101 returns (512-d embedding, auxiliary features).
            embeddings = backbone_output[0]
        else:
            embeddings = backbone_output

        if not torch.is_tensor(embeddings):
            raise TypeError(
                f"Expected embedding tensor, got {type(embeddings).__name__}"
            )

        if tuple(embeddings.shape) != (batch_size, EMBEDDING_SIZE):
            raise AssertionError(
                f"Unexpected embedding shape: {tuple(embeddings.shape)}"
            )

        # CurricularFace receives only the 512-dimensional embeddings.
        logits = head(embeddings, labels)

        if isinstance(logits, (tuple, list)):
            raise TypeError(
                "CurricularFace returned a tuple/list. "
                "Inspect the head output before computing cross-entropy."
            )

        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()

        if logits.shape != (batch_size, EXPECTED_IDENTITIES):
            raise AssertionError(
                f"Unexpected logits shape: {logits.shape}"
            )

        if not torch.isfinite(loss):
            raise AssertionError("Loss contains NaN or Inf.")

        print("Embedding:", tuple(embeddings.shape))
        print("Logits:", tuple(logits.shape))
        print("Loss:", float(loss.detach()))

        return "IR-101 + CurricularFace forward/backward passed."

    finally:
        del backbone, head
        torch.cuda.empty_cache()


def checkpoint_resume_test():
    if not (REPO / "train.py").is_file():
        raise FileNotFoundError(REPO / "train.py")

    cfg = load_config()

    if Path(cfg["DATA_ROOT"]).resolve() != DATA_ROOT.resolve():
        raise AssertionError("config.py DATA_ROOT is incorrect.")

    if Path(cfg["RECORD_DIR"]).resolve() != MANIFEST.resolve():
        raise AssertionError("config.py RECORD_DIR is incorrect.")

    base = Path(
        "/content/drive/MyDrive/ABSS-CurricularFace-Implementation/"
        "validation_smoke"
    )

    model_dir = base / "checkpoints"
    log_dir = base / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest = model_dir / "latest_checkpoint.pth"

    if latest.exists():
        latest.unlink()

    common = [
        sys.executable,
        str(REPO / "train.py"),
        "--checkpoint-every-steps", "1",
        "--model-root", str(model_dir),
        "--log-root", str(log_dir),
    ]

    subprocess.run(
        common + ["--max-steps", "2"],
        cwd=REPO,
        check=True
    )

    if not latest.is_file():
        raise RuntimeError(
            "First smoke run did not create latest_checkpoint.pth"
        )

    subprocess.run(
        common + [
            "--max-steps", "3",
            "--resume", str(latest)
        ],
        cwd=REPO,
        check=True
    )

    import torch

    state = torch.load(
        latest,
        map_location="cpu",
        weights_only=False
    )

    print("Checkpoint global_step:", state.get("global_step"))
    print("Checkpoint next_batch:", state.get("next_batch"))

    if int(state.get("global_step", -1)) != 3:
        raise AssertionError(
            "Resume test did not advance global_step to 3."
        )

    return "Checkpoint resumed successfully in a second process."


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-test",
        action="store_true",
        help="Run two short training processes to verify checkpoint/resume."
    )

    args = parser.parse_args()

    results = []

    results.append(check("Environment and GPU", environment))

    results.append(
        check(
            "Repository, config, and Python syntax",
            source_and_config
        )
    )

    results.append(
        check(
            "Dataset manifest and file integrity",
            dataset_integrity
        )
    )

    results.append(
        check(
            "Dataset loading and transform",
            dataset_batch
        )
    )

    results.append(
        check(
            "IR-101 + CurricularFace GPU forward/backward",
            model_forward_backward
        )
    )

    if args.checkpoint_test:
        results.append(
            check(
                "Checkpoint save and cross-process resume",
                checkpoint_resume_test
            )
        )

    print("\n" + "=" * 64)

    print(
        f"VALIDATION RESULT: {sum(results)}/{len(results)} checks passed"
    )

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
