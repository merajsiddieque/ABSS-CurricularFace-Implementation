#!/usr/bin/env python3
"""Launch CurricularFace training and resume from the latest Drive checkpoint.

Run again after a Colab runtime reset. If latest_checkpoint.pth exists, it is
passed to train.py with --resume. Checkpoints are written by train.py.

Examples:
    !python 04_train_resume.py
    !python 04_train_resume.py --checkpoint-every-steps 100
    !python 04_train_resume.py --max-steps 20   # controlled smoke run only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/ABSS-CurricularFace-Implementation")
DEFAULT_MODEL_ROOT = Path("/content/drive/MyDrive/ABSS-CurricularFace-Implementation/checkpoints")
DEFAULT_LOG_ROOT = Path("/content/drive/MyDrive/ABSS-CurricularFace-Implementation/logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional total global-step limit, mainly for smoke tests.")
    parser.add_argument("--resume-from", default=None,
                        help="Explicit checkpoint path; otherwise latest_checkpoint.pth is used.")
    parser.add_argument("--start-fresh", action="store_true",
                        help="Do not resume. Does not delete existing checkpoints.")
    args = parser.parse_args()

    train_py = REPO / "train.py"
    if not train_py.is_file():
        raise SystemExit(f"Training script not found: {train_py}")
    if args.checkpoint_every_steps < 1:
        raise SystemExit("--checkpoint-every-steps must be >= 1")

    model_root = Path(args.model_root)
    log_root = Path(args.log_root)
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    latest = model_root / "latest_checkpoint.pth"

    resume = None
    if not args.start_fresh:
        if args.resume_from:
            resume = Path(args.resume_from)
        elif latest.is_file():
            resume = latest
    if resume is not None and not resume.is_file():
        raise SystemExit(f"Resume checkpoint does not exist: {resume}")

    cmd = [
        sys.executable, str(train_py),
        "--checkpoint-every-steps", str(args.checkpoint_every_steps),
        "--model-root", str(model_root),
        "--log-root", str(log_root),
    ]
    if resume:
        cmd += ["--resume", str(resume)]
        print(f"Resuming from: {resume}")
    else:
        print("No resume checkpoint selected; starting from configured initial state.")
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]

    print("Repository:", REPO)
    print("Checkpoint interval:", args.checkpoint_every_steps, "steps")
    print("Model output:", model_root)
    print("Log output:", log_root)
    print("\nLaunching training. Use the same model root when restarting.")
    subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
