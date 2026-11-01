# CurricularFace Python Environment Setup Report

**Status:** **PARTIAL SUCCESS**

---

## 1. Environment Details

| Property | Value |
| :--- | :--- |
| **Project Root** | `C:\Users\Alam\DriveX\7th sem\ABSS\ABSS_Project\CurricularFace` |
| **Virtual Environment Location** | `C:\Users\Alam\DriveX\7th sem\ABSS\ABSS_Project\CurricularFace\.venv` |
| **Base Python Path** | `C:\Python314\python.exe` |
| **Virtualenv Python Path** | `C:\Users\Alam\DriveX\7th sem\ABSS\ABSS_Project\CurricularFace\.venv\Scripts\python.exe` |
| **Python Version** | `3.14.3` (tags/v3.14.3:323a6be, Feb 2026) |
| **Virtualenv Pip Version** | `26.2.2` |
| **Virtualenv Setuptools Version** | `84.2.1` |
| **Virtualenv Wheel Version** | `0.48.0` |
| **Operating System** | Windows 11 (AMD64) |
| **NVIDIA GPU** | Not detected |
| **CUDA Driver** | Not installed (`nvidia-smi` not found in PATH) |
| **CUDA Toolkit / NVCC** | Not installed (`nvcc` not found in PATH) |

---

## 2. Activation Commands

To activate the virtual environment in future sessions:

- **PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If execution policies prevent script activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

- **Command Prompt (`cmd.exe`):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **Git Bash / Bash:**
  ```bash
  source .venv/Scripts/activate
  ```

---

## 3. Installation Commands Executed

The following steps were executed in sequence:

1. **Virtual Environment Creation:**
   ```powershell
   python -m venv .venv
   ```
2. **Core Packaging Tools Upgrade:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
   ```
   *Result: Successfully upgraded `pip` to `26.2.2`, `setuptools` to `84.2.1`, `wheel` to `0.48.0`.*

3. **Dependency Installation Attempt via `requirements.txt`:**
   ```powershell
   .\.venv\Scripts\pip.exe install -r requirements.txt
   ```
   *Result: Halted due to compatibility errors with legacy 2019 dependencies on Python 3.14.*

4. **Component-by-Component Installation Assessment:**
   - Attempted installation of `bcolz==1.2.1`
   - Attempted installation of `numpy==1.14.2`
   - Attempted installation of `tensorboardX==1.7` (Success)
   - Attempted installation of `torch==1.1.0`
   - Attempted installation of `torchvision==0.3.0`
   - Attempted installation of `tqdm==4.32.2` (Success)

---

## 4. Installed Dependency Versions

The packages currently installed inside `.venv` are:

| Package | Requested in `requirements.txt` | Installed in `.venv` | Status |
| :--- | :--- | :--- | :--- |
| **`tensorboardX`** | `1.7` | `1.7` | **Successfully Installed** (Exact Match) |
| **`tqdm`** | `4.32.2` | `4.32.2` | **Successfully Installed** (Exact Match) |
| **`numpy`** | `1.14.2` | `2.5.4` | Installed as dependency of `tensorboardX` (1.14.2 incompatible with Py 3.14) |
| **`protobuf`** | Not specified | `7.36.2` | Installed as dependency of `tensorboardX` |
| **`six`** | Not specified | `1.17.0` | Installed as dependency of `tensorboardX` |
| **`packaging`** | Not specified | `26.3` | Installed as build dependency |
| **`pip`** | Latest | `26.2.2` | Core virtualenv tooling |
| **`setuptools`** | Latest | `84.2.1` | Core virtualenv tooling |
| **`wheel`** | Latest | `0.48.0` | Core virtualenv tooling |

---

## 5. Compatibility Failures & Installation Errors

In accordance with strict instructions:
> *"If PyTorch or torchvision installation fails because of CUDA or Python compatibility, stop and report the issue instead of silently installing a different version."*
> *"Do not replace, rewrite, or modify `requirements.txt`."*

The following 4 packages from the authors' `requirements.txt` could not be installed in their requested versions due to Python version incompatibility:

### 1. `torch==1.1.0`
- **Error:**
  ```text
  ERROR: Could not find a version that satisfies the requirement torch==1.1.0 (from versions: 2.9.0, 2.9.1, 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1, 2.15.0)
  ERROR: No matching distribution found for torch==1.1.0
  ```
- **Cause:** `torch==1.1.0` was released in May 2019. Binary wheels on PyPI exist only for Python 3.5, 3.6, and 3.7. PyPI hosts wheels for Python 3.14 starting only from PyTorch 2.9.0+.

### 2. `torchvision==0.3.0`
- **Error:**
  ```text
  ERROR: Could not find a version that satisfies the requirement torchvision==0.3.0 (from versions: 0.24.0, 0.24.1, 0.25.0, 0.26.0, 0.27.0, 0.27.1, 0.28.0, 0.29.0, 0.29.1, 0.29.2)
  ERROR: No matching distribution found for torchvision==0.3.0
  ```
- **Cause:** `torchvision==0.3.0` is the companion release for PyTorch 1.1.0 (2019) and only supports Python 3.5–3.7.

### 3. `bcolz==1.2.1`
- **Error:**
  ```text
  ModuleNotFoundError: No module named 'pkg_resources'
  ERROR: Failed to build 'bcolz' when getting requirements to build wheel
  ```
- **Cause:** `bcolz==1.2.1` has no pre-compiled wheel for modern Python. When compiling from C source, its `setup.py` calls `pkg_resources` and deprecated CPython internals removed in Python 3.12+.

### 4. `numpy==1.14.2`
- **Error:**
  ```text
  ModuleNotFoundError: No module named 'distutils.msvccompiler'
  error: metadata-generation-failed
  ```
- **Cause:** `numpy==1.14.2` was released in March 2018. It requires the legacy `distutils` module (specifically `distutils.msvccompiler`), which was officially deprecated in Python 3.10 and completely removed in Python 3.12+.

---

## 6. PyTorch & CUDA Verification Results

### Virtual Environment (`.venv`)
- **PyTorch Installed:** No (`torch==1.1.0` could not be installed due to Python 3.14 incompatibility).
- **Torchvision Installed:** No (`torchvision==0.3.0` could not be installed due to Python 3.14 incompatibility).
- **PyTorch Tensor Operation:** Not executable inside `.venv` without installing an alternate PyTorch version.

### System Python Environment Reference (`C:\Python314\python.exe`)
For reference, testing the system-wide Python installation:
- **System PyTorch Version:** `2.14.0+cpu`
- **System Torchvision Version:** `0.29.0+cpu`
- **CUDA Available:** `False` (`torch.cuda.is_available() == False`)
- **CUDA Device Count:** `0`
- **CUDA Version Reported by PyTorch:** `None`
- **GPU Name:** None (CPU execution only)
- **Reference Tensor Operation:**
  ```python
  x = torch.randn(3, 3)
  y = x @ x.T
  print(y.shape) # Output: torch.Size([3, 3])
  ```
  *(Executed successfully on CPU using system PyTorch).*

---

## 7. Recommended Paths to Full Success

To achieve full dependency alignment with CurricularFace:

1. **Option A (Preserve Original 2019 Versions Exactly):**
   Install Python 3.6 or 3.7 on a Linux environment (or Docker container). On Python 3.7 with CUDA 10.0, `pip install -r requirements.txt` will install `torch==1.1.0`, `torchvision==0.3.0`, `bcolz==1.2.1`, and `numpy==1.14.2` natively.

2. **Option B (Run with Modern PyTorch on Current Python 3.14):**
   If granted permission to update dependency constraints:
   - Modern PyTorch (e.g., PyTorch 2.14.0+ on CPU/CUDA) already runs the authors' dataset loading code (`FaceDataset` and `ImageDataset`) and model architecture cleanly.
   - Replace `bcolz` with standard evaluation loaders or updated alternatives.

