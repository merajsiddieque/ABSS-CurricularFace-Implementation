
#!/usr/bin/env python3
"""Prepare a Google Colab session for CurricularFace.

Google Drive must already be mounted in Colab.
The repository is cloned only when the target directory does not exist.
Existing working code is never overwritten or pulled automatically.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/merajsiddieque/ABSS-CurricularFace-Implementation.git"
REPO_DIR = Path("/content/ABSS-CurricularFace-Implementation")
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/ABSS-CurricularFace-Implementation")

DEFAULT_DATA_ZIP_CANDIDATES = [
    DRIVE_PROJECT_DIR / "data.zip",
    Path("/content/drive/MyDrive/data.zip"),
    Path("/content/data.zip"),
]

DATA_ROOT = Path("/content/CASIA-WebFace/data")
EXPECTED_DATASET_DIR = DATA_ROOT / "CASIA-WebFace"
MANIFEST = DATA_ROOT / "casia_train.txt"


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def check_drive() -> None:
    """Verify that Google Drive is already mounted."""
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted or MyDrive is unavailable. "
            "Mount Drive in the Colab notebook before running this script."
        )

    print("Google Drive already mounted. Skipping mount.")


def locate_zip(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Dataset ZIP not found: {candidate}")

    for candidate in DEFAULT_DATA_ZIP_CANDIDATES:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find data.zip. Put it in "
        f"{DRIVE_PROJECT_DIR}/data.zip or pass --data-zip /path/to/data.zip."
    )


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")

        root = destination.resolve()

        for member in zf.infolist():
            target = (destination / member.filename).resolve()

            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")

        zf.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", help="Path to CASIA-WebFace data.zip")
    parser.add_argument("--repo-dir", default=str(REPO_DIR))
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)

    # Verify Drive without attempting to mount it.
    check_drive()

    DRIVE_PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # Clone only if the repository does not exist.
    if not repo_dir.exists():
        run(["git", "clone", REPO_URL, str(repo_dir)])
    else:
        print(f"Repository already exists: {repo_dir}")
        print("Not pulling or overwriting existing working code.")

    # Dataset preparation.
    if not args.skip_extract:
        zip_path = locate_zip(args.data_zip)
        print(f"Using dataset archive: {zip_path}")

        if EXPECTED_DATASET_DIR.exists() and any(EXPECTED_DATASET_DIR.iterdir()):
            print(
                f"Dataset already extracted at {EXPECTED_DATASET_DIR}; "
                "skipping extraction."
            )
        else:
            local_zip = Path("/content/data.zip")

            if zip_path.resolve() != local_zip.resolve():
                shutil.copy2(zip_path, local_zip)

            print(f"Extracting {local_zip} into {DATA_ROOT} ...")
            safe_extract(local_zip, DATA_ROOT)

    print("\nSession setup summary")
    print(f"Repository: {repo_dir}")
    print(
        f"Dataset directory: {EXPECTED_DATASET_DIR} "
        f"(exists={EXPECTED_DATASET_DIR.is_dir()})"
    )
    print(f"Manifest: {MANIFEST} (exists={MANIFEST.is_file()})")
    print("Next: run 02_validate_project.py before training.")


if __name__ == "__main__":
    main()
