# CurricularFace Legacy Environment Setup Report

**Status:** **HALTED (Python 3.7 Unavailable)**

---

## 1. Inspection of Available Python Installations

An inspection of the local system for Python installations was conducted:

| Command / Check | Result |
| :--- | :--- |
| **`python --version`** | `Python 3.14.3` |
| **`py -0p`** | ` -V:3.14 * C:\Python314\python.exe`<br>` -V:3.13 C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\python3.13.exe` |
| **`py -3.7 --version`** | `No suitable Python runtime found` |
| **Standard Path Checks** | `C:\Python37` (Not found)<br>`C:\Program Files\Python37` (Not found)<br>`C:\Users\Alam\AppData\Local\Programs\Python\Python37` (Not found) |

**Conclusion:** Python 3.7 is **not installed** on this machine and is unavailable through the Windows Python Launcher (`py.exe`).

---

## 2. Environment Creation Status

- **Environment Name:** `.venv37`
- **Creation Status:** **NOT CREATED**
- **Existing Environment Status:** The existing `.venv` environment (Python 3.14) was preserved without modifications.

In accordance with Step 1 instructions:
> *"If Python 3.7 is unavailable: Do not install it automatically. Report this and provide the recommended installation source. Stop before creating the new environment."*

Execution was halted prior to creating `.venv37` or modifying any system configuration.

---

## 3. Hardware & Compute Availability

| Property | Status | Notes |
| :--- | :--- | :--- |
| **CPU Architecture** | AMD64 (x86_64) | Active |
| **NVIDIA GPU** | Not detected | No dedicated GPU accessible |
| **CUDA Driver** | Not installed | `nvidia-smi` not found in PATH |
| **CUDA Toolkit (NVCC)** | Not installed | `nvcc` not found in PATH |

---

## 4. Remaining Compatibility Issues with `requirements.txt`

The official CurricularFace dependencies specified in [`requirements.txt`](requirements.txt) are:

```text
bcolz==1.2.1
numpy==1.14.2
tensorboardX==1.7
torch==1.1.0
torchvision==0.3.0
tqdm==4.32.2
```

These legacy packages have the following requirements:
1. **`torch==1.1.0` and `torchvision==0.3.0`:** Released in May 2019. Pre-compiled wheels on PyPI exist only for Python 3.5, 3.6, and 3.7.
2. **`numpy==1.14.2` and `bcolz==1.2.1`:** Rely on `distutils.msvccompiler` and `pkg_resources`, both of which were removed in Python 3.12+.
3. Python 3.13 and Python 3.14 (the only currently installed runtimes) cannot build or install these 2018–2019 binary distributions.

---

## 5. Recommended Installation Sources for Python 3.7

To enable creation of `.venv37` and native installation of the exact 2019 dependencies, Python 3.7 (64-bit) can be installed using one of the following official options:

### Option 1: Official Python 3.7 Windows Installer (Recommended)
- **Version:** Python 3.7.9 (final 64-bit installer release for Windows)
- **Download URL:** [https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe)
- **Installation instruction:**
  - Run the installer.
  - Check *"Add Python 3.7 to PATH"* or allow the Windows Python Launcher (`py.exe`) to register it.

### Option 2: Conda / Miniconda Environment
If Conda is preferred to manage legacy runtimes without altering global Windows PATH:
```cmd
conda create -n curricularface_py37 python=3.7
conda activate curricularface_py37
pip install -r requirements.txt
```

### Option 3: Docker Container
Using the official PyTorch 1.1.0 image:
```bash
docker run -it --rm -v "C:\Users\Alam\DriveX\7th sem\ABSS\ABSS_Project\CurricularFace":/workspace pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
```

---

## 6. Next Steps Once Python 3.7 Is Available

Once Python 3.7 is registered with the launcher (`py -0p` lists `3.7`), the setup can proceed automatically:

```powershell
# 1. Create .venv37
py -3.7 -m venv .venv37

# 2. Activate
.\.venv37\Scripts\Activate.ps1

# 3. Upgrade basic packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install exact requirements
python -m pip install -r requirements.txt
```

