#!/usr/bin/env python3
"""
tools/colmap_install.py

Installer for COLMAP on Ubuntu/Debian with an explicit choice:
  --cpu : install distro COLMAP via apt (usually "without CUDA")
  --gpu : build a *stable release tag* of COLMAP from source with CUDA enabled

Goal of this version:
- Build a **stable release tag** (default: 3.9.1) instead of the moving master/dev branch.
  This avoids surprises like extra heavyweight deps (e.g. vendored ONNX Runtime) that can
  appear on dev branches.

Important:
- DO NOT run this script with sudo.
  It will call sudo only for apt steps.

Examples:
  python tools/colmap_install.py --cpu
  python tools/colmap_install.py --gpu
  python tools/colmap_install.py --gpu --colmap-tag 3.9.1
  python tools/colmap_install.py --purge

Notes about CUDA:
- This script uses the simple route: install `nvidia-cuda-toolkit` via apt if `nvcc` is missing.
  (On some systems, you may prefer NVIDIA's official CUDA repo — but this keeps it automated.)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# -------------------------- helpers --------------------------

def _die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"[colmap_install] {msg}")


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def sh(cmd: list[str], check: bool = True, cwd: Optional[Path] = None) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, cwd=str(cwd) if cwd else None).returncode


def sh_out(cmd: list[str], check: bool = False, cwd: Optional[Path] = None) -> str:
    print("$ " + " ".join(cmd))
    p = subprocess.run(cmd, check=check, cwd=str(cwd) if cwd else None,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout


def which(p: str) -> Optional[str]:
    return shutil.which(p)


def get_cuda_version(nvcc_path: Optional[str] = None) -> Optional[str]:
    """Return CUDA version as 'MAJOR.MINOR' from `nvcc --version`, or None if unavailable."""
    nvcc = nvcc_path or which("nvcc")
    if not nvcc:
        return None
    out = sh_out([nvcc, "--version"], check=False)
    # Typical line: "Cuda compilation tools, release 12.2, V12.2.140"
    m = re.search(r"release\s+(\d+\.\d+)", out)
    if m:
        return m.group(1)
    m = re.search(r"V(\d+\.\d+)", out)
    if m:
        return m.group(1)
    return None



def build_info_path(prefix: Path) -> Path:
    # Stored next to the installed binaries so it survives source/build cleanups.
    return (prefix / "install" / ".colmap_build_info.json").resolve()


def read_build_info(prefix: Path) -> Optional[dict]:
    path = build_info_path(prefix)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_build_info(prefix: Path, info: dict) -> None:
    path = build_info_path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def looks_like_cuda_build(colmap_bin: Path) -> bool:
    # Best-effort sanity check: many COLMAP builds print "without CUDA" in -h output when CUDA is disabled.
    out = sh_out([str(colmap_bin), "-h"], check=False)
    return "without CUDA" not in out


def can_skip_gpu_build(prefix: Path, tag: str, cuda_arch: str, cuda_version_req: Optional[str]) -> bool:
    colmap_bin = (prefix / "install" / "bin" / "colmap").resolve()
    if not colmap_bin.exists():
        return False

    info = read_build_info(prefix) or {}
    if not info:
        return False

    if not info.get("cuda_enabled", False):
        return False
    if info.get("tag") != tag:
        return False
    if info.get("cuda_arch") != cuda_arch:
        return False

    # Extra safety: ensure the binary still *behaves* like a CUDA build.
    return looks_like_cuda_build(colmap_bin)

def is_debian_like() -> bool:
    return Path("/etc/debian_version").exists()


def bootstrap_python() -> Path:
    """Return a usable system Python for bootstrapping venv creation.

    We prefer sys.executable (the interpreter running this script). If it doesn't
    exist (rare, but can happen with a broken/removed venv), we fall back to
    'python3' found in PATH.
    """
    py = Path(os.sys.executable)
    if py.exists():
        return py
    import shutil
    found = shutil.which("python3") or shutil.which("python")
    if not found:
        raise RuntimeError("No usable Python found (python3/python not in PATH).")
    return Path(found)

def in_venv() -> bool:
    # If VIRTUAL_ENV is set but the interpreter is missing (broken venv), ignore it.
    try:
        if not Path(os.sys.executable).exists():
            return False
    except Exception:
        return False
    return (hasattr(os.sys, "base_prefix") and os.sys.prefix != os.sys.base_prefix) or bool(os.environ.get("VIRTUAL_ENV"))


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> Path:
    py = venv_python(venv_dir)
    if py.exists():
        return py
    print(f"[colmap_install] Creating venv: {venv_dir}")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    sh([str(bootstrap_python()), "-m", "venv", str(venv_dir)])
    py = venv_python(venv_dir)
    if not py.exists():
        _die(f"Failed to create venv python at: {py}")
    return py


def ensure_python_deps(req_path: Path, venv_dir: Path) -> None:
    print("\n== Python deps (venv) ==")
    if not req_path.exists():
        print(f"[colmap_install] requirements.txt not found at {req_path}. Skipping.")
        return

    if in_venv():
        py = Path(os.sys.executable)
        if not py.exists():
            py = bootstrap_python()
        venv_used = os.environ.get("VIRTUAL_ENV", "(active venv)")
    else:
        py = ensure_venv(venv_dir)
        venv_used = str(venv_dir)

    pip_cmd = [str(py), "-m", "pip"]
    sh(pip_cmd + ["install", "--upgrade", "pip"], check=False)
    sh(pip_cmd + ["install", "-r", str(req_path)])

    print(f"\n[colmap_install] Python deps installed into: {venv_used}")
    if not in_venv():
        print("Activate it with:")
        print(f"  source {venv_dir}/bin/activate")


# -------------------------- CUDA detection/install --------------------------

def detect_nvidia_gpu() -> bool:
    if not which("nvidia-smi"):
        return False
    out = sh_out(["nvidia-smi", "-L"], check=False).strip()
    return bool(out) and "GPU" in out


def detect_cuda_toolkit() -> bool:
    return which("nvcc") is not None


def install_cuda_toolkit_via_apt() -> None:
    print("\n== Installing CUDA toolkit (apt: nvidia-cuda-toolkit) ==")
    sh(["sudo", "apt-get", "update"])
    sh(["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"])


# -------------------------- COLMAP install modes --------------------------

def apt_install_cpu_colmap() -> None:
    if not is_debian_like():
        _die("--cpu (apt install) is only implemented for Debian/Ubuntu.")
    print("\n== Installing COLMAP (CPU) via apt ==")
    sh(["sudo", "apt-get", "update"])
    sh(["sudo", "apt-get", "install", "-y",
        "python3-full", "python3-venv",
        "colmap",
        "ffmpeg",
        "git",
        "cmake",
        "build-essential",
    ], check=False)

    sh(["sudo", "apt-get", "install", "-y",
        "qtbase5-dev",
        "libqt5opengl5-dev",
        "libx11-dev",
    ], check=False)


def ensure_build_deps_cuda() -> None:
    if not is_debian_like():
        _die("CUDA build deps install is only implemented for Debian/Ubuntu.")

    print("\n== Installing build deps for COLMAP (CUDA build) ==")
    sh(["sudo", "apt-get", "update"])
    sh(["sudo", "apt-get", "install", "-y",
        "python3-full", "python3-venv",
        "git",
        "cmake",
        "build-essential",
        "ninja-build",
        "pkg-config",
        "libboost-program-options-dev",
        "libboost-filesystem-dev",
        "libboost-graph-dev",
        "libboost-system-dev",
        "libeigen3-dev",
        "libflann-dev",
        "libfreeimage-dev",
        "libopenimageio-dev",
        "openimageio-tools",
        "libmetis-dev",
        "libgoogle-glog-dev",
        "libgtest-dev",
        "libsqlite3-dev",
        "libglew-dev",
        "qtbase5-dev",
        "libqt5opengl5-dev",
        "libx11-dev",
        "libcgal-dev",
        "libceres-dev",
    ])



def apply_known_patches(src_dir: Path) -> None:
    """Apply small source patches needed for some distro/compiler combinations.

    Patches currently applied:
      1) mvs/workspace.h: ensure <memory> is included when std::unique_ptr is used.
         (Fixes: "'unique_ptr' is not a member of std" / missing header)
      2) image/line.cc: ensure <memory> is included *outside* any extern "C" block.
         (Fixes: "template with C linkage" explosions when <memory> gets included inside extern "C")
    """

    # ---- Patch 1: workspace.h (<memory>) ----
    workspace_h = src_dir / "src" / "colmap" / "mvs" / "workspace.h"
    if workspace_h.exists():
        txt = workspace_h.read_text(encoding="utf-8", errors="replace")
        if ("std::unique_ptr" in txt) and ("#include <memory>" not in txt):
            lines = txt.splitlines(True)
            out = []
            inserted = False

            # Prefer inserting after a standard STL include.
            for ln in lines:
                out.append(ln)
                if (not inserted) and ln.startswith("#include") and ("<vector>" in ln or "<string>" in ln):
                    out.append("#include <memory>\n")
                    inserted = True

            if not inserted:
                # Fallback: after the last include, or at top.
                last_inc = None
                for i, ln in enumerate(out):
                    if ln.startswith("#include"):
                        last_inc = i
                if last_inc is None:
                    out.insert(0, "#include <memory>\n")
                else:
                    out.insert(last_inc + 1, "#include <memory>\n")

            workspace_h.write_text("".join(out), encoding="utf-8")
            print("[colmap_install] Patch applied: workspace.h add <memory>")

    # ---- Patch 2: line.cc (<memory> outside extern \"C\") ----
    line_cc = src_dir / "src" / "colmap" / "image" / "line.cc"
    if line_cc.exists():
        txt = line_cc.read_text(encoding="utf-8", errors="replace")
        if "std::unique_ptr" in txt:
            lines = txt.splitlines(True)

            # Remove any existing include; we'll re-add it in the safe location.
            lines = [ln for ln in lines if ln.strip() != "#include <memory>"]

            # If file has an extern "C" block, insert BEFORE it.
            extern_idx = None
            for i, ln in enumerate(lines):
                if ln.lstrip().startswith('extern "C"'):
                    extern_idx = i
                    break

            if extern_idx is not None:
                insert_at = extern_idx
            else:
                # Otherwise insert after last #include.
                last_inc = None
                for i, ln in enumerate(lines):
                    if ln.startswith("#include"):
                        last_inc = i
                insert_at = 0 if last_inc is None else last_inc + 1

            lines.insert(insert_at, "#include <memory>\n")
            new_txt = "".join(lines)
            if new_txt != txt:
                line_cc.write_text(new_txt, encoding="utf-8")
                print('[colmap_install] Patch applied: line.cc ensure <memory> outside extern "C"')


def checkout_stable_tag(src_dir: Path, tag: str) -> None:
    # Fetch tags and checkout the requested tag
    sh(["git", "-C", str(src_dir), "fetch", "--tags", "--force"])
    # Verify tag exists
    tags = sh_out(["git", "-C", str(src_dir), "tag", "-l", tag], check=False).strip()
    if not tags:
        # Provide a helpful hint
        tail = sh_out(["git", "-C", str(src_dir), "tag"], check=False).strip().splitlines()[-20:]
        hint = "\n  ".join(tail) if tail else "(no tags listed)"
        _die(f"Requested tag '{tag}' not found in repo tags.\nRecent tags:\n  {hint}")
    sh(["git", "-C", str(src_dir), "checkout", "-f", tag])
    sh(["git", "-C", str(src_dir), "submodule", "update", "--init", "--recursive"])


def build_colmap_cuda(prefix: Path, jobs: int, tag: str, cuda_arch: str, cuda_version: Optional[str]) -> Path:
    """
    Build a stable COLMAP release tag with CUDA enabled into prefix/install.
    Returns the installed colmap binary path.
    """
    ensure_build_deps_cuda()

    prefix = prefix.resolve()
    src_dir = prefix / "src"
    build_dir = prefix / "build"
    install_dir = prefix / "install"
    colmap_bin = install_dir / "bin" / "colmap"

    prefix.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print("\n== Cloning COLMAP ==")
        sh(["git", "clone", "--recursive", "https://github.com/colmap/colmap.git", str(src_dir)])
    else:
        print("\n== Updating COLMAP repo ==")
        sh(["git", "-C", str(src_dir), "fetch", "--all", "--prune"])
        sh(["git", "-C", str(src_dir), "submodule", "update", "--init", "--recursive"])

    print(f"\n== Checking out stable release tag: {tag} ==")
    checkout_stable_tag(src_dir, tag)
    apply_known_patches(src_dir)

    # Clean build dir for safety (stable tag changes can confuse incremental builds)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    print("\n== Configuring (CMake, CUDA_ENABLED=ON) ==")
    sh([
        "cmake",
        "-S", str(src_dir),
        "-B", str(build_dir),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCUDA_ENABLED=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
    ])

    print("\n== Building ==")
    sh(["cmake", "--build", str(build_dir), f"-j{jobs}"])

    print("\n== Installing ==")
    sh(["cmake", "--install", str(build_dir)])

    if not colmap_bin.exists():
        _die(f"Build finished but colmap binary not found at: {colmap_bin}")

    print("\n[colmap_install] COLMAP (CUDA) installed to:", install_dir)
    print("For this shell:")
    print(f"  export PATH=\"{install_dir / 'bin'}:$PATH\"")
    print("Or set colmap_bin in your YAML config to that full path.")


    # Record build metadata so future runs can skip rebuilds when everything matches.
    write_build_info(prefix, {
        "tag": tag,
        "cuda_enabled": True,
        "cuda_arch": cuda_arch,
        "cuda_version": cuda_version,
    })

    return colmap_bin


# -------------------------- purge --------------------------

def purge(prefix: Path) -> None:
    print("\n== Purge requested ==")
    if is_debian_like():
        print("[colmap_install] Removing apt package 'colmap' (if installed)...")
        sh(["sudo", "apt-get", "remove", "-y", "colmap"], check=False)
        sh(["sudo", "apt-get", "autoremove", "-y"], check=False)

    if prefix.exists():
        print(f"[colmap_install] Removing local prefix: {prefix}")
        shutil.rmtree(prefix)
    else:
        print(f"[colmap_install] Local prefix not found: {prefix}")

    print("[colmap_install] Purge done.")


# -------------------------- main --------------------------

def main() -> None:
    if _is_root():
        _die(
            "Do NOT run this script with sudo.\n"
            "Run it as your user. It will call sudo only for apt/build steps.\n"
            "Example:\n"
            "  python tools/colmap_install.py --cpu\n"
            "  python tools/colmap_install.py --gpu"
        )

    if platform.system() != "Linux":
        _die("This installer currently supports Linux only.")

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--cpu", action="store_true", help="Install CPU COLMAP via apt.")
    g.add_argument("--gpu", action="store_true", help="Build CUDA-enabled COLMAP from a stable release tag.")

    ap.add_argument("--purge", action="store_true", help="Remove COLMAP (apt) and local build prefix, then exit.")
    ap.add_argument("--python-only", action="store_true", help="Only install python deps into venv and exit.")

    ap.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt")
    ap.add_argument("--venv", default=".venv", help="Venv directory to use/create when not already in a venv.")

    ap.add_argument("--prefix", default=".third_party/colmap_cuda", help="Prefix dir for --gpu build install.")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8, help="Parallel build jobs for --gpu build.")

    ap.add_argument("--cuda-toolkit", choices=["auto", "apt"], default="auto",
                    help="How to install CUDA toolkit if missing (default: auto->apt).")

    ap.add_argument("--colmap-tag", default="3.9.1",
                    help="COLMAP git tag to build for --gpu (stable release). Example: 3.9.1")

    ap.add_argument("--cuda-arch", default="native",
                    help="Value for CMAKE_CUDA_ARCHITECTURES (e.g. native, all-major, 61 for GTX 1060).")

    ap.add_argument("--cuda-version", default="auto",
                    help="Requested CUDA version (MAJOR.MINOR) for --gpu. Default: auto (use installed nvcc version). Example: 12.2")

    args = ap.parse_args()

    req_path = Path(args.requirements).resolve()
    venv_dir = Path(args.venv).resolve()
    prefix = Path(args.prefix).resolve()

    # Resolve requested CUDA version for --gpu
    cuda_version_installed = get_cuda_version()
    if args.gpu:
        if args.cuda_version and args.cuda_version != "auto":
            cuda_version_req = args.cuda_version
        else:
            cuda_version_req = cuda_version_installed
    else:
        cuda_version_req = None

    print(f"[colmap_install] OS: {platform.system()} {platform.release()}")

    if args.purge:
        purge(prefix)
        return

    if args.gpu and args.cuda_version and args.cuda_version != "auto":
        if not cuda_version_installed:
            _die("Requested --cuda-version but nvcc was not found. Install CUDA toolkit first (nvcc).")
        if cuda_version_installed != cuda_version_req:
            _die(f"Installed CUDA (nvcc) version is {cuda_version_installed}, but you requested {cuda_version_req}. "
                 "This script currently installs CUDA via apt without pinning versions; please install the requested CUDA version, "
                 "or use --cuda-version auto.")

    # Enforce explicit choice unless python-only
    if not args.python_only and not (args.cpu or args.gpu):
        _die("You must choose exactly one: --cpu or --gpu (or use --python-only / --purge).")

    if args.python_only:
        ensure_python_deps(req_path, venv_dir)
        return

    if args.cpu:
        apt_install_cpu_colmap()

    if args.gpu:
        if not detect_nvidia_gpu():
            _die("No NVIDIA GPU detected (nvidia-smi -L empty or not found). Can't proceed with --gpu.")
        print("[colmap_install] NVIDIA GPU detected.")

        if not detect_cuda_toolkit():
            print("[colmap_install] CUDA toolkit (nvcc) not found.")
            if args.cuda_toolkit in ("auto", "apt"):
                install_cuda_toolkit_via_apt()
        else:
            print("[colmap_install] CUDA toolkit detected (nvcc found).")

        if not detect_cuda_toolkit():
            _die("CUDA toolkit still not detected after install attempt. (nvcc not found)")

        if can_skip_gpu_build(
            prefix=prefix,
            tag=args.colmap_tag,
            cuda_arch=args.cuda_arch,
            cuda_version_req=cuda_version_req,
        ):
            built_bin = (prefix / "install" / "bin" / "colmap").resolve()
            info = read_build_info(prefix) or {}
            print("[colmap_install] Detected an existing matching CUDA build. Skipping compilation.")
            if info:
                print(f"[colmap_install] Using tag={info.get('tag')} commit={info.get('git_commit')} arch={info.get('cuda_arch')}")
        else:
            built_bin = build_colmap_cuda(
                prefix=prefix,
                jobs=int(args.jobs),
                tag=args.colmap_tag,
                cuda_arch=args.cuda_arch,
                cuda_version=cuda_version_req,
            )

        # Quick sanity: verify "without CUDA" is NOT printed (best-effort)
        out = sh_out([str(built_bin), "-h"], check=False)
        if "without CUDA" in out:
            print("[colmap_install] WARNING: Built COLMAP still reports 'without CUDA'.")
            print("[colmap_install] This typically means CUDA was not detected by CMake.")
        else:
            print("[colmap_install] OK: CUDA build does not report 'without CUDA'.")

    # Python deps (always)
    ensure_python_deps(req_path, venv_dir)

    print("\n== Quick check ==")
    if args.cpu and which("colmap"):
        sh(["colmap", "-h"], check=False)
        print("[colmap_install] OK: apt colmap appears callable.")
    if args.gpu:
        cuda_colmap = prefix / "install" / "bin" / "colmap"
        if cuda_colmap.exists():
            print(f"[colmap_install] CUDA colmap binary: {cuda_colmap}")
            print("[colmap_install] Put this in your YAML:")
            print(f"  colmap_bin: {cuda_colmap}")

    print("[colmap_install] done.")


if __name__ == "__main__":
    main()
