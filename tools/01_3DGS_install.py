#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3DGS_install_auto_venv_v13.py

Robust installer for INRIA 3D Gaussian Splatting (gaussian-splatting) that handles:
- Legacy GPUs like GTX 1060 (sm_61) which require Python <= 3.11 for compatible PyTorch CUDA wheels.
- Newer GPUs like T4 (sm_75) and L4 (sm_89) that work with recent Python/PyTorch.

Key behavior:
- Detect GPU (via nvidia-smi) and infer compute capability (best-effort mapping).
- If legacy GPU (sm < 7.0) AND current Python >= 3.12:
    - Ensure python3.11 exists (apt-get install; uses sudo if needed and --allow-sudo passed).
    - Create a dedicated venv (.venv_3dgs by default) using python3.11.
    - Re-run this installer inside that venv and continue.
- Install PyTorch (cu118 for legacy; cu121 for newer by default), repo deps, and build CUDA extensions.
- Install CUDA extensions NON-editable (avoids PEP660 editable "finder-only" installs).

Usage:
  python3 tools/3DGS_install_auto_venv_v13.py --auto --allow-sudo

Options:
  --venv .venv_3dgs
  --prefix .third_party/3dgs
  --repo-url https://github.com/camenduru/gaussian-splatting
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple


# -------------------------
# small helpers
# -------------------------

def info(msg: str) -> None:
    print(f"[3DGS_install] {msg}")

def warn(msg: str) -> None:
    print(f"[3DGS_install] WARNING: {msg}")

def die(msg: str, code: int = 2) -> None:
    print(f"[3DGS_install] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)

def capture(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def which(prog: str) -> Optional[str]:
    return shutil.which(prog)

def in_venv() -> bool:
    return (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix) or bool(os.environ.get("VIRTUAL_ENV"))

def pip_cmd(python_exe: str) -> List[str]:
    return [python_exe, "-m", "pip"]

def is_debian_like() -> bool:
    try:
        txt = Path("/etc/os-release").read_text(encoding="utf-8", errors="ignore").lower()
        return any(x in txt for x in ("ubuntu", "debian", "pop", "mint"))
    except Exception:
        return False

def can_sudo_noninteractive() -> bool:
    if not which("sudo"):
        return False
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def try_install_python311(allow_sudo: bool) -> bool:
    """Try to install python3.11 packages via apt-get. Returns True if python3.11 is available after."""
    if which("python3.11"):
        return True
    if not is_debian_like() or not which("apt-get"):
        return False

    prefix: List[str] = []
    if which("sudo"):
        if can_sudo_noninteractive():
            prefix = ["sudo", "-n"]
        elif allow_sudo:
            prefix = ["sudo"]
        else:
            warn(
                "python3.11 not found. I can install it via apt, but sudo would prompt for a password.\n"
                "Re-run with --allow-sudo to let me run sudo, or run manually:\n"
                "  sudo apt-get update\n"
                "  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev\n"
            )
            return False

    info("Attempting to install python3.11 via apt-get…")
    try:
        run(prefix + ["apt-get", "update"])
        run(prefix + ["apt-get", "install", "-y", "python3.11", "python3.11-venv", "python3.11-dev"])
    except Exception as e:
        warn(f"Automatic apt-get install failed: {e}")
        return False
    return which("python3.11") is not None

def ensure_venv(venv_dir: Path, python_exe: str) -> str:
    py = venv_dir / "bin" / "python"
    if py.exists():
        return str(py)
    info(f"Creating venv: {venv_dir} (python={python_exe})")
    run([python_exe, "-m", "venv", str(venv_dir)])
    return str(py)

def reexec_in_venv(venv_python: str, argv: List[str]) -> None:
    # add a marker to avoid loops
    new_argv = [venv_python, argv[0]] + [a for a in argv[1:] if a != "--_reexeced"] + ["--_reexeced"]
    info("Re-executing installer inside dedicated venv…")
    run(new_argv)
    raise SystemExit(0)

def detect_nvcc() -> Optional[str]:
    nvcc = which("nvcc")
    if not nvcc:
        return None
    try:
        out = capture([nvcc, "--version"])
        for line in out.splitlines():
            if "release" in line.lower():
                return line.strip()
        return out.strip().splitlines()[-1] if out.strip() else "present"
    except Exception:
        return "present"

def nvidia_smi_ok() -> bool:
    smi = which("nvidia-smi")
    if not smi:
        return False
    try:
        capture([smi, "-L"])
        return True
    except Exception:
        return False

def gpu_name() -> Optional[str]:
    smi = which("nvidia-smi")
    if not smi:
        return None
    try:
        out = capture([smi, "--query-gpu=name", "--format=csv,noheader"])
        return out.strip().splitlines()[0].strip() or None
    except Exception:
        try:
            out = capture([smi, "-L"])
            m = re.search(r"GPU\s+0:\s+(.+?)\s+\(UUID:", out)
            return m.group(1).strip() if m else None
        except Exception:
            return None

def map_name_to_cc(name: str) -> Optional[str]:
    n = name.lower()
    if "gtx 1060" in n:
        return "6.1"
    if "t4" in n:
        return "7.5"
    if re.search(r"\bl4\b", n):
        return "8.9"
    # add a couple more common ones
    if "rtx 30" in n or "rtx 3080" in n or "rtx 3090" in n:
        return "8.6"
    if "rtx 40" in n or "rtx 4080" in n or "rtx 4090" in n:
        return "8.9"
    return None

def cc_tuple(cc: str) -> Tuple[int, int]:
    a, b = cc.split(".", 1)
    return int(a), int(b)

def is_legacy_cc(cc: str) -> bool:
    return cc_tuple(cc) < (7, 0)

def choose_torch_packages(cc: Optional[str], torch_cuda: str) -> Tuple[List[str], Optional[str]]:
    """Return (packages, extra_index_url).

    Important:
      * Pascal / sm_61 GPUs (e.g. GTX 1060) require PyTorch wheels that still ship kernels for sm_<70.
        Recent PyTorch CUDA 12.x wheels may drop these architectures, producing:
            'no kernel image is available for execution on the device'
      * A robust workaround is to install an older-but-still-modern PyTorch build from the CUDA 12.1
        wheel channel, pinned to a version known to work on Pascal.
    """

    legacy = bool(cc and is_legacy_cc(cc))

    # Decide wheel channel.
    if torch_cuda == "auto":
        # Use cu121 by default; it works on T4/L4 and (with the pinned versions below) on Pascal.
        torch_cuda = "cu121"
    if torch_cuda == "cpu":
        return ["torch", "torchvision", "torchaudio"], None

    # Pinned set for legacy GPUs (Pascal sm_61).
    if legacy:
        # NOTE: Keep these three versions in sync.
        pkgs = ["torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"]
        return pkgs, "https://download.pytorch.org/whl/cu121"

    # Non-legacy: unpinned, but use requested channel.
    extra = f"https://download.pytorch.org/whl/{torch_cuda}"
    return ["torch", "torchvision", "torchaudio"], extra



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="Auto-detect and install.")
    ap.add_argument("--allow-sudo", action="store_true", help="Allow sudo (may prompt) to install python3.11 when needed.")
    ap.add_argument("--venv", default=".venv_3dgs", help="Dedicated venv path for 3DGS.")
    ap.add_argument("--prefix", default=".third_party/3dgs", help="Where to put the repo.")
    ap.add_argument("--repo-url", default="https://github.com/camenduru/gaussian-splatting", help="Repo URL")
    ap.add_argument("--branch", default=None, help="Optional branch/tag")
    ap.add_argument("--torch-cuda", default="auto", choices=["auto", "cu118", "cu121", "cpu"], help="Torch wheel flavor")
    ap.add_argument("--cuda-arch-list", default=None, help="Override TORCH_CUDA_ARCH_LIST (e.g. 6.1, 7.5, 8.9)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Reinstall and reclone.")
    ap.add_argument("--_reexeced", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    prefix = (repo_root / args.prefix).resolve() if not Path(args.prefix).is_absolute() else Path(args.prefix).expanduser().resolve()
    repo_dir = prefix / "gaussian-splatting"

    info(f"OS: {platform.system()} {platform.release()}")
    info(f"Python: {sys.version.split()[0]}  (executable: {sys.executable})")
    info(f"Repo dir: {repo_dir}")

    gname = gpu_name() if nvidia_smi_ok() else None
    cc = map_name_to_cc(gname) if gname else None
    if gname:
        info(f"GPU: {gname}")
    else:
        warn("No NVIDIA GPU detected via nvidia-smi (or nvidia-smi missing).")

    if cc:
        info(f"Compute capability (best-effort): sm_{cc}")
    else:
        warn("Could not infer compute capability. You may need --cuda-arch-list.")

    need_legacy = bool(cc and is_legacy_cc(cc))

    # nvcc requirement for CUDA extensions
    if args.torch_cuda != "cpu":
        nvcc = detect_nvcc()
        if not nvcc:
            die("CUDA toolkit not found (nvcc missing). Install CUDA toolkit so `nvcc --version` works.")
        info(f"nvcc: {nvcc}")

    # If legacy GPU + Python >=3.12, we MUST move to python3.11 venv.
    if args.auto and need_legacy and sys.version_info >= (3, 12) and not args._reexeced:
        info("Legacy GPU detected (sm_<70) on Python >= 3.12: need a Python 3.11 venv for compatible PyTorch.")
        ok = try_install_python311(args.allow_sudo)
        if not which("python3.11") and not ok:
            die(
                "python3.11 is required for GTX 1060-class GPUs, but it is not available.\n"
                "Install it and rerun:\n"
                "  sudo apt-get update\n"
                "  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev\n"
            )
        venv_dir = (repo_root / args.venv).resolve() if not Path(args.venv).is_absolute() else Path(args.venv).expanduser().resolve()
        venv_py = ensure_venv(venv_dir, "python3.11")
        reexec_in_venv(venv_py, sys.argv)

    # If we reexeced but still 3.12 on legacy, stop.
    if need_legacy and sys.version_info >= (3, 12):
        die(
            "Legacy GPU detected but current Python is still >= 3.12.\n"
            "Make sure python3.11 exists and rerun with --auto."
        )

    if in_venv():
        info("Virtual environment detected.")
    else:
        warn("Not running in a virtual environment; installs may affect system Python.")

    py = sys.executable

    run(pip_cmd(py) + ["install", "-U", "pip", "setuptools", "wheel"])

    # Torch arch list (important for compiling extensions)
    chosen_arch = args.cuda_arch_list or cc
    if chosen_arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = chosen_arch
        info(f"TORCH_CUDA_ARCH_LIST={chosen_arch}")
    else:
        warn("TORCH_CUDA_ARCH_LIST not set; builds may target the wrong architecture.")

    torch_pkgs, extra_index = choose_torch_packages(cc, args.torch_cuda)
    cmd = pip_cmd(py) + ["install"] + torch_pkgs
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    run(cmd)

    # quick torch sanity
    run([py, "-c", "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"])

    # Clone/update repo
    prefix.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and not args.force:
        info(f"Repo already exists: {repo_dir} (skipping clone)")
    else:
        if repo_dir.exists():
            info(f"Removing existing repo (force): {repo_dir}")
            shutil.rmtree(repo_dir)
        run(["git", "clone", "--recursive", args.repo_url, str(repo_dir)])
        if args.branch:
            run(["git", "checkout", args.branch], cwd=repo_dir)
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)

    # Install requirements
    req = repo_dir / "requirements.txt"
    if req.exists():
        run(pip_cmd(py) + ["install", "-r", str(req)])
    run(pip_cmd(py) + ["install", "plyfile"])
    run(pip_cmd(py) + ["install", "tqdm", "imageio", "opencv-python", "pyyaml"])

    # Build/install CUDA extensions non-editable
    env = os.environ.copy()
    env["MAX_JOBS"] = str(args.jobs)

    def install_submodule(rel: str) -> None:
        p = repo_dir / rel
        if not p.exists():
            die(f"Missing submodule directory: {p}")
        run(pip_cmd(py) + ["install", "--no-build-isolation", str(p)], env=env)

    install_submodule("submodules/diff-gaussian-rasterization")
    install_submodule("submodules/simple-knn")

    # import tests
    run([py, "-c", "import simple_knn, diff_gaussian_rasterization; print('extensions OK')"])

    info("Done.")
    print("\nNext:")
    print(f"  Activate venv: {os.environ.get('VIRTUAL_ENV', '(current python)')}")
    print(f"  Train: {repo_dir/'train.py'}")
    print("  Example: python train.py -s /path/to/scene --eval\n")


if __name__ == "__main__":
    main()
