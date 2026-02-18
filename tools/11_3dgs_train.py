#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3dgs_train_autoswitch_debug.py

Same as autoswitch, but prints diagnostics about which python candidates were tested and why they failed.

Usage:
  python3 tools/3dgs_train_autoswitch_debug.py --config test2.yaml --debug
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple


def _info(msg: str) -> None:
    print(f"[3dgs_train] {msg}")


def _warn(msg: str) -> None:
    print(f"[3dgs_train] WARNING: {msg}")


def _die(msg: str, code: int = 2) -> None:
    print(f"[3dgs_train] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _run(cmd, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        _die("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _probe_import(python: str, mod: str) -> Tuple[bool, str]:
    try:
        p = subprocess.run([python, "-c", f"import {mod}; print('ok')"],
                           check=True, capture_output=True, text=True)
        return True, (p.stdout + p.stderr).strip()
    except subprocess.CalledProcessError as e:
        out = ((e.stdout or "") + (e.stderr or "")).strip()
        return False, out
    except FileNotFoundError:
        return False, "python not found"
    except PermissionError:
        return False, "permission denied"
    except Exception as e:
        return False, str(e)


def _has_any_flag(args: List[str], flags: List[str]) -> bool:
    s = set(args)
    return any(f in s for f in flags)


def _inject_kv_flag(args: List[str], flags: List[str], value: Any) -> List[str]:
    """Append flag+value if none of the flags are already present."""
    if value is None:
        return args
    if _has_any_flag(args, flags):
        return args
    return args + [flags[0], str(value)]

def _candidate_list(repo_root: Path, cfg: Dict[str, Any], cli_python: Optional[str]) -> List[str]:
    t = (cfg.get("three_dgs") or {})
    cands: List[str] = []
    if cli_python:
        cands.append(cli_python)
    if t.get("python"):
        cands.append(str(t["python"]))
    venv_dir = t.get("venv_dir")
    if venv_dir:
        cands.append(str(Path(venv_dir).expanduser() / "bin" / "python"))
    cands.append(str((repo_root / ".venv_3dgs" / "bin" / "python").resolve()))
    cands.append(sys.executable)

    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for c in cands:
        if not c:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _pick_python(repo_root: Path, cfg: Dict[str, Any], cli_python: Optional[str], debug: bool) -> str:
    cands = _candidate_list(repo_root, cfg, cli_python)

    results = []
    for py in cands:
        ok_t, out_t = _probe_import(py, "torch")
        ok_s, out_s = _probe_import(py, "simple_knn") if ok_t else (False, "(skipped)")
        ok_d, out_d = _probe_import(py, "diff_gaussian_rasterization") if ok_t else (False, "(skipped)")
        results.append((py, ok_t, ok_s, ok_d, out_t, out_s, out_d))

    for py, ok_t, ok_s, ok_d, *_ in results:
        if ok_t and ok_s and ok_d:
            return py
    for py, ok_t, *_ in results:
        if ok_t:
            return py

    if debug:
        _warn("Python candidate probe results:")
        for py, ok_t, ok_s, ok_d, out_t, out_s, out_d in results:
            _warn(f"- {py}")
            _warn(f"  torch={ok_t} | simple_knn={ok_s} | diff_gaussian_rasterization={ok_d}")
            if out_t:
                _warn(f"  torch output: {out_t[:600]}")
            if ok_t and not ok_s and out_s:
                _warn(f"  simple_knn output: {out_s[:600]}")
            if ok_t and not ok_d and out_d:
                _warn(f"  diff_gaussian_rasterization output: {out_d[:600]}")
    _die(
        "Could not find a python environment with torch installed.\n"
        "Set three_dgs.venv_dir or three_dgs.python in YAML, or pass --python /path/to/venv/bin/python."
    )
    return sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=None)
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = _repo_root()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        _die(f"Config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    t = (cfg.get("three_dgs") or {})

    repo_dir = Path(t.get("repo_dir") or (repo_root / ".third_party" / "3dgs" / "gaussian-splatting")).expanduser()
    if not repo_dir.is_absolute():
        repo_dir = (repo_root / repo_dir).resolve()
    if not (repo_dir / "train.py").exists():
        _die(f"3DGS repo not found or missing train.py: {repo_dir}")

    # Work directory: base folder for relative scene/output paths
    work_dir_raw = (
        (t.get("work_dir") if isinstance(t, dict) else None)
        or cfg.get("work_dir")
        or ((cfg.get("paths") or {}).get("work_dir") if isinstance(cfg.get("paths"), dict) else None)
    )
    work_dir = Path(work_dir_raw).expanduser() if work_dir_raw else repo_root
    if not work_dir.is_absolute():
        work_dir = (repo_root / work_dir).resolve()

    # Scene dir (relative paths are resolved against work_dir)
    scene_dir = Path(t.get("scene_dir") or (work_dir / "3dgs_scene")).expanduser()
    if not scene_dir.is_absolute():
        scene_dir = (work_dir / scene_dir).resolve()
    if not scene_dir.exists():
        _die(f"Scene dir not found: {scene_dir}")

    # Output dir (defaults to work_dir/3dgs/output; relative paths are resolved against work_dir)
    output_dir = Path(t.get("output_dir") or (work_dir / "3dgs" / "output")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (work_dir / output_dir).resolve()
    _ensure_dir(output_dir)

    train_cfg = (t.get("train") or {})
    iters = int(args.iters or train_cfg.get("iterations") or 30000)

    if args.eval and args.no_eval:
        _die("Can't use both --eval and --no-eval")
    eval_flag = bool(train_cfg.get("eval", True))
    if args.eval:
        eval_flag = True
    if args.no_eval:
        eval_flag = False

    extra_args = list(train_cfg.get("extra_args") or [])
    if args.extra:
        extra_args += args.extra

    # Resolution / downscale handling:
    # In the official 3DGS repo, --resolution / -r is the downsample factor:
    #   1 = full-res, 2 = half, 4 = quarter, ...
    # Your YAML may express it as "resolution", "r", "downscale", or "divide_by".
    res_val = (
        train_cfg.get("resolution")
        if train_cfg.get("resolution") is not None
        else train_cfg.get("r")
        if train_cfg.get("r") is not None
        else train_cfg.get("downscale")
        if train_cfg.get("downscale") is not None
        else train_cfg.get("divide_by")
        if train_cfg.get("divide_by") is not None
        else (t.get("resolution") if t.get("resolution") is not None else t.get("divide_by"))
    )
    if res_val is not None:
        extra_args = _inject_kv_flag(extra_args, ["--resolution", "-r"], res_val)
        _info(f"resolution: {res_val} (passed via --resolution/-r unless already in extra_args)")

    py = _pick_python(repo_root, cfg, args.python, args.debug)

    _info(f"python:    {py}")
    _info(f"repo_dir:  {repo_dir}")
    _info(f"work_dir:  {work_dir}")
    _info(f"scene_dir: {scene_dir}")
    _info(f"output_dir:{output_dir}")
    _info(f"iters:     {iters}")
    _info(f"eval:      {eval_flag}")

    env = os.environ.copy()
    cuda_vis = args.cuda
    if cuda_vis is None and train_cfg.get("cuda_visible_devices") is not None:
        cuda_vis = str(train_cfg.get("cuda_visible_devices"))
    if cuda_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)
        _info(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    else:
        _info("CUDA_VISIBLE_DEVICES not set")

    cmd = [py, str((repo_dir / "train.py").resolve()), "-s", str(scene_dir), "--iterations", str(iters)]

    # Ensure outputs go to our chosen output directory (instead of repo_dir/output)
    # Most gaussian-splatting forks support --model_path (sometimes -m). We try --model_path.
    extra_args = _inject_kv_flag(extra_args, ["--model_path", "-m"], str(output_dir))

    if eval_flag:
        cmd.append("--eval")
    cmd += extra_args

    _run(cmd, cwd=repo_dir, env=env)


if __name__ == "__main__":
    main()
