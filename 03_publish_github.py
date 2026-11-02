#!/usr/bin/env python3
"""Safely commit and push the current Colab project to GitHub.

Create Colab Secrets named GITHUB_USERNAME and GITHUB_TOKEN first.
This script never prints the token and never embeds it in the remote URL.

Usage:
    !python 03_publish_github.py --message "Update training and validation scripts"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/content/ABSS-CurricularFace-Implementation")
REMOTE_URL = "https://github.com/merajsiddieque/ABSS-CurricularFace-Implementation.git"


def secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore
        return str(userdata.get(name) or "").strip()
    except Exception:
        return ""


def git(args: list[str], env=None, capture=False):
    return subprocess.run(
        ["git", *args], cwd=REPO, env=env, check=True,
        text=True, capture_output=capture
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--all", action="store_true",
                        help="Stage all changes, including untracked files. Review .gitignore first.")
    args = parser.parse_args()

    if not (REPO / ".git").exists():
        raise SystemExit(f"Not a Git repository: {REPO}")

    username = secret("GITHUB_USERNAME")
    token = secret("GITHUB_TOKEN")
    if not username or not token:
        raise SystemExit("Missing GITHUB_USERNAME or GITHUB_TOKEN in Colab Secrets.")

    # Keep the remote URL credential-free.
    git(["remote", "set-url", "origin", REMOTE_URL])
    status = git(["status", "--short"], capture=True).stdout.strip()
    print("Working tree before staging:")
    print(status or "(clean)")
    if not status:
        print("Nothing to publish.")
        return

    if args.all:
        git(["add", "-A"])
    else:
        # Stage tracked modifications plus untracked source/config/docs, but skip
        # common data, checkpoint, log, and cache paths.
        git(["add", "-u"])
        for path in [
            "README.md", "config.py", "train.py", "evaluate.py",
            "verify_dataset.py", "requirements.txt", "dataset",
            "backbone", "head", "loss", "util", "reports",
            "01_setup_session.py", "02_validate_project.py",
            "03_publish_github.py", "04_train_resume.py",
        ]:
            candidate = REPO / path
            if candidate.exists():
                git(["add", "--", path])

    staged = git(["diff", "--cached", "--name-only"], capture=True).stdout.strip()
    if not staged:
        print("No eligible source changes staged. Use --all only after reviewing ignored files.")
        return
    print("\nFiles staged:")
    print(staged)

    # Refuse obvious large data/checkpoint artifacts.
    forbidden = (".zip", ".pth", ".pt", ".ckpt", ".tar", ".tar.gz")
    bad = [line for line in staged.splitlines() if line.lower().endswith(forbidden)]
    if bad:
        raise SystemExit("Refusing to publish data/checkpoint artifacts:\n" + "\n".join(bad))

    git(["config", "user.name", username])
    git(["config", "user.email", f"{username}@users.noreply.github.com"])
    git(["commit", "-m", args.message])

    with tempfile.TemporaryDirectory() as tmp:
        askpass = Path(tmp) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  *Username*) printf "%s\\n" "$GH_USER" ;;\n'
            '  *Password*) printf "%s\\n" "$GH_TOKEN" ;;\n'
            '  *) exit 1 ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        env = os.environ.copy()
        env.update({
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "GH_USER": username,
            "GH_TOKEN": token,
        })
        git(["push", "-u", "origin", args.branch], env=env)
    print("Push completed successfully.")


if __name__ == "__main__":
    main()
