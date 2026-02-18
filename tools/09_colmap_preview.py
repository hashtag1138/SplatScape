#!/usr/bin/env python3
"""
tools/colmap_preview.py

- Skips steps if outputs already exist (unless --force).
- --view opens a viewer:
    * If "model_viewer" exists in this COLMAP build: uses it.
    * Otherwise falls back to "colmap gui" (standard COLMAP viewer).
- Optional --export-ply exports sparse points to PLY (view with CloudCompare/MeshLab).

Usage:
  python tools/colmap_preview.py --config test_colmap.yaml
  python tools/colmap_preview.py --config test_colmap.yaml --view
  python tools/colmap_preview.py --config test_colmap.yaml --view --export-ply
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get(d, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _get_paths(cfg: dict) -> dict:
    """Support both legacy root-level 'paths' and newer 'colmap.paths'."""
    paths = cfg.get("paths")
    if isinstance(paths, dict) and paths:
        return paths

    colmap = cfg.get("colmap")
    if isinstance(colmap, dict):
        p2 = colmap.get("paths")
        if isinstance(p2, dict) and p2:
            return p2

    return {}


def _repo_root_from_this_file() -> Path:
    # tools/colmap_preview.py -> repo_root
    return Path(__file__).resolve().parent.parent



try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"[colmap_preview] {msg}")


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        _die("PyYAML is required. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        _die("Config root must be a mapping/dict.")
    return cfg


def _resolve_path(cfg_path: Path, p: str, repo_root: Path | None = None) -> Path:
    """Resolve a path from config.

    - Absolute paths are returned as-is.
    - Relative paths are resolved against repo_root (preferred) when provided,
      otherwise against the config file directory.
    """
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    base = repo_root if repo_root is not None else cfg_path.parent
    # Prefer repo_root for typical repo-relative paths like work/... or works2/...
    return (base / pp).resolve()


def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".third_party").exists():
            return p
    return start


def _autodetect_colmap_bin(cfg_path: Path, colmap_bin: str) -> str:
    if colmap_bin and colmap_bin != "colmap":
        return str(_resolve_path(cfg_path, colmap_bin))
    which = shutil.which("colmap")
    if which:
        return which
    repo = _find_repo_root(cfg_path.parent)
    candidates = [
        repo / ".third_party" / "colmap_cuda" / "install" / "bin" / "colmap",
        repo / ".third_party" / "colmap" / "install" / "bin" / "colmap",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "colmap"


def _ensure_colmap_exists(colmap_bin: str, cfg_path: Path) -> None:
    if colmap_bin == "colmap":
        if shutil.which("colmap") is not None:
            return
        repo = _find_repo_root(cfg_path.parent)
        hint = repo / ".third_party" / "colmap_cuda" / "install" / "bin" / "colmap"
        _die(f"COLMAP binary not found. Set in YAML:\n  colmap_bin: {hint}")
    p = Path(colmap_bin)
    if not p.exists():
        _die(f"COLMAP binary path does not exist: {p}")


def _run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            _die(f"Command failed (exit={ret}). See log: {log_path}", code=ret)


def _has_features(database_path: Path) -> bool:
    return database_path.exists() and database_path.stat().st_size > 0


def _has_sparse_model(sparse_dir: Path) -> Optional[Path]:
    if not sparse_dir.exists():
        return None

    def is_model_dir(d: Path) -> bool:
        cams = (d / "cameras.bin").exists() or (d / "cameras.txt").exists()
        imgs = (d / "images.bin").exists() or (d / "images.txt").exists()
        pts = (d / "points3D.bin").exists() or (d / "points3D.txt").exists()
        return cams and imgs and pts

    d0 = sparse_dir / "0"
    if d0.exists() and is_model_dir(d0):
        return d0
    if is_model_dir(sparse_dir):
        return sparse_dir
    for child in sorted(sparse_dir.iterdir()):
        if child.is_dir() and is_model_dir(child):
            return child
    return None


def _colmap_has_command(colmap_bin: str, command: str) -> bool:
    # `colmap help` lists commands; robust across builds.
    try:
        out = subprocess.check_output([colmap_bin, "help"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return False
    return f" {command} " in out or f"\n{command}\n" in out or f"\n{command} " in out


def _export_ply(colmap_bin: str, model_dir: Path, ply_path: Path) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(ply_path),
        "--output_type", "PLY",
    ]
    print(f"[colmap_preview] Exporting PLY: {ply_path}")
    subprocess.run(cmd, check=False)


def _open_view(colmap_bin: str, model_dir: Path, prefer: str) -> None:
    # Prefer model_viewer if it exists, else GUI.
    if prefer == "model_viewer" and _colmap_has_command(colmap_bin, "model_viewer"):
        cmd = [colmap_bin, "model_viewer", "--input_path", str(model_dir)]
        print(f"[colmap_preview] Opening model_viewer on: {model_dir}")
        subprocess.run(cmd, check=False)
        return

    # GUI exists in normal COLMAP builds.
    if _colmap_has_command(colmap_bin, "gui"):
        print("[colmap_preview] Opening COLMAP GUI (viewer is inside).")
        print("[colmap_preview] In the GUI: File -> Import model... then select the sparse model directory.")
        subprocess.run([colmap_bin, "gui"], check=False)
        return

    _die("No viewer command available in this COLMAP build (no gui, no model_viewer).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--force", action="store_true", help="Force re-run steps even if outputs exist")
    ap.add_argument("--view", action="store_true", help="Open viewer if sparse model exists")
    ap.add_argument("--viewer", choices=["gui", "model_viewer"], default="gui")
    ap.add_argument("--export-ply", action="store_true", help="Export sparse model to PLY for external viewers")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = _repo_root_from_this_file()
    if not cfg_path.exists():
        _die(f"Config not found: {cfg_path}")

    cfg = _load_config(cfg_path)

    colmap_bin_raw = str(cfg.get("colmap_bin", "colmap"))
    colmap_bin = _autodetect_colmap_bin(cfg_path, colmap_bin_raw)
    _ensure_colmap_exists(colmap_bin, cfg_path)

    paths = _get_paths(cfg)
    if not isinstance(paths, dict):
        _die("paths must be a dict.")

    colmap_dir = _resolve_path(cfg_path, str(paths.get("colmap_dir", "colmap_out")), repo_root)
    images_dir = _resolve_path(cfg_path, str(paths.get("images_dir", "images")), repo_root)
    database_path = _resolve_path(cfg_path, str(paths.get("database_path", colmap_dir / "database.db")), repo_root)
    sparse_dir = _resolve_path(cfg_path, str(paths.get("sparse_dir", colmap_dir / "sparse")), repo_root)
    logs_dir = _resolve_path(cfg_path, str(paths.get("logs_dir", colmap_dir / "logs")), repo_root)

    images_dir_p = Path(images_dir)
    if not images_dir or not images_dir_p.exists():
        _die(
            "Images directory does not exist:\n"
            f"  images_dir: {images_dir}\n"
            "Fix your YAML paths.images_dir or colmap.paths.images_dir (or create the folder)."
        )

    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    fe = cfg.get("feature_extractor", {}) or {}
    em = cfg.get("exhaustive_matcher", {}) or {}
    mapper = cfg.get("mapper", {}) or {}

    have_db = _has_features(database_path)
    model_dir = _has_sparse_model(sparse_dir)

    print(f"[colmap_preview] Using colmap_bin: {colmap_bin}")
    print(f"[colmap_preview] images_dir:   {images_dir}")
    print(f"[colmap_preview] database:     {database_path}   (exists={database_path.exists()})")
    print(f"[colmap_preview] sparse_dir:    {sparse_dir}      (model={'yes' if model_dir else 'no'})")

    # If just view and already have model, don’t compute.
    if args.view and model_dir is not None and not args.force:
        if args.export_ply:
            _export_ply(colmap_bin, model_dir, model_dir / "sparse_points.ply")
        _open_view(colmap_bin, model_dir, args.viewer)
        print("[colmap_preview] done.")
        return

    # Step 1
    if args.force or not have_db:
        cmd1 = [colmap_bin, "feature_extractor", "--database_path", str(database_path), "--image_path", str(images_dir)]
        for k, v in fe.items():
            cmd1 += [f"--{k}", str(v)]
        _run_cmd(cmd1, logs_dir / "01_feature_extractor.log")
        have_db = _has_features(database_path)
    else:
        print("[colmap_preview] Skip feature_extractor (database already exists).")

    # Step 2 (only on --force by default; otherwise we don’t touch DB)
    if args.force:
        cmd2 = [colmap_bin, "exhaustive_matcher", "--database_path", str(database_path)]
        for k, v in em.items():
            cmd2 += [f"--{k}", str(v)]
        _run_cmd(cmd2, logs_dir / "02_exhaustive_matcher.log")
    else:
        print("[colmap_preview] Skip exhaustive_matcher (no --force).")

    # Step 3
    model_dir = _has_sparse_model(sparse_dir)
    if args.force or model_dir is None:
        cmd3 = [colmap_bin, "mapper", "--database_path", str(database_path), "--image_path", str(images_dir), "--output_path", str(sparse_dir)]
        for k, v in mapper.items():
            cmd3 += [f"--{k}", str(v)]
        _run_cmd(cmd3, logs_dir / "03_mapper.log")
        model_dir = _has_sparse_model(sparse_dir)
    else:
        print("[colmap_preview] Skip mapper (sparse model already exists).")

    if args.export_ply:
        if model_dir is None:
            _die("No sparse model found to export. Check logs/03_mapper.log.")
        _export_ply(colmap_bin, model_dir, model_dir / "sparse_points.ply")

    if args.view:
        if model_dir is None:
            _die("No sparse model found to view. Check logs/03_mapper.log.")
        _open_view(colmap_bin, model_dir, args.viewer)

    print("[colmap_preview] done.")


if __name__ == "__main__":
    main()

