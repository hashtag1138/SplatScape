#!/usr/bin/env python3
"""
tools/colmap_clear.py

Delete COLMAP outputs referenced by a YAML config, to keep tests clean.

Usage:
  python tools/colmap_clear.py --config configs/colmap.yaml
  python tools/colmap_clear.py --config configs/colmap.yaml --yes

It removes (if present):
- database.db
- sparse/  (all reconstructions)
- undistorted/
- dense/   (if created later)
- logs/

It DOES NOT delete:
- images_dir
- the config file itself
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"[colmap_clear] {msg}")


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        _die("PyYAML is required. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        _die("Config root must be a mapping/dict.")
    return cfg


def _resolve_path(cfg_path: Path, p: str, repo_root: Path | None = None) -> Path:
    """Resolve paths from config.

    - Absolute paths are kept as-is.
    - Relative paths are resolved against repo_root if provided, else config directory.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    base = repo_root if repo_root is not None else cfg_path.parent
    return (base / pp).resolve()


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"[colmap_clear] removed dir: {path}")
    else:
        path.unlink()
        print(f"[colmap_clear] removed file: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        _die(f"Config not found: {cfg_path}")

    cfg = _load_config(cfg_path)
    # Optional: if config defines a work_dir, we assume config lives in repo root and
    # use that folder as a signal to resolve relative paths against the repo root.
    repo_root = cfg_path.parent
    wd = cfg.get('work_dir')
    if isinstance(wd, str) and wd:
        # keep repo_root as config dir; work_dir itself may be relative/absolute elsewhere
        pass
    # Support both legacy 'paths' at root and the newer 'colmap.paths' layout
    paths = cfg.get("paths", {})
    if (not isinstance(paths, dict)) or (not paths):
        colmap_cfg = cfg.get("colmap", {})
        if isinstance(colmap_cfg, dict):
            paths = colmap_cfg.get("paths", {})

    if not isinstance(paths, dict):
        _die("paths must be a dict (either at root 'paths' or under 'colmap.paths').")

    colmap_dir = _resolve_path(cfg_path, str(paths.get("colmap_dir", "")), repo_root)
    database_path = _resolve_path(cfg_path, str(paths.get("database_path", colmap_dir / "database.db")), repo_root)
    database_shm_path = Path(str(database_path) + "-shm")
    database_wal_path = Path(str(database_path) + "-wal")
    sparse_dir = _resolve_path(cfg_path, str(paths.get("sparse_dir", colmap_dir / "sparse")), repo_root)
    undistorted_dir = _resolve_path(cfg_path, str(paths.get("undistorted_dir", colmap_dir / "undistorted")), repo_root)
    logs_dir = _resolve_path(cfg_path, str(paths.get("logs_dir", colmap_dir / "logs")), repo_root)
    dense_dir = colmap_dir / "dense"
    ply_path = colmap_dir / "sparse0.ply"

    targets = [database_path, database_shm_path, database_wal_path, sparse_dir, undistorted_dir, dense_dir, logs_dir, ply_path]

    if not args.yes:
        print("About to remove:")
        for t in targets:
            if t.exists():
                print(f"  - {t}")
        resp = input("Proceed? (y/N): ").strip().lower()
        if resp not in ("y", "yes"):
            print("[colmap_clear] cancelled.")
            return

    removed_any = False
    for t in targets:
        if t.exists():
            removed_any = True
        _rm(t)

    if not removed_any:
        print("[colmap_clear] nothing to remove (paths may not match your config).")

    # Keep colmap_dir itself (nice for retaining the folder)
    print("[colmap_clear] done.")


if __name__ == "__main__":
    main()
