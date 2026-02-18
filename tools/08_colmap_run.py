#!/usr/bin/env python3
"""
tools/colmap_run.py

Run a standard COLMAP pipeline from a YAML config:
1) feature_extractor
2) matcher (exhaustive / sequential / spatial / vocab_tree)
3) mapper (sparse)
4) image_undistorter

Usage:
  python tools/colmap_run.py --config test_colmap.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"[colmap_run] {msg}")


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        _die("PyYAML is required. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and 'colmap' in cfg and isinstance(cfg.get('colmap'), dict):
        cfg = cfg['colmap']
    if not isinstance(cfg, dict):
        _die("Config root must be a mapping/dict.")
    return cfg


def _resolve_path(cfg_path: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (cfg_path.parent / pp).resolve()


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
        repo / ".third_party" / "colmap_cuda" / "bin" / "colmap",
        repo / ".third_party" / "colmap" / "bin" / "colmap",
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
        _die(
            "COLMAP binary not found.\n"
            "Either install 'colmap' in PATH, or set in your YAML:\n"
            f"  colmap_bin: {hint}\n"
            f"Current colmap_bin value resolved to: {colmap_bin}"
        )

    p = Path(colmap_bin)
    if not p.exists():
        _die(f"COLMAP binary path does not exist: {p}")
    if p.is_dir():
        _die(f"COLMAP binary path is a directory, expected an executable file: {p}")


def _ensure_dir(path: Path, what: str) -> None:
    if path.exists() and not path.is_dir():
        _die(
            f"{what} must be a directory but is not: {path}\n"
            "Tip: remove/rename that path (or run your colmap_clear.py) and retry."
        )
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path, what: str) -> None:
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        _die(f"Parent directory for {what} is not a directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)


def _run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n[colmap_run] Command failed to start. "
                "Is colmap_bin correct in your YAML?"
            ) from e

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            _die(f"Command failed (exit={ret}). See log: {log_path}", code=ret)


def _build_matcher_cmd(cfg: Dict[str, Any], colmap_bin: str, database_path: Path) -> tuple[list[str], str]:
    """Build COLMAP matcher command from config.

    Supported config styles:
      - New style:
          matcher:
            type: sequential|exhaustive|spatial|vocab_tree
            options:
              <COLMAPFlag>: <value>
      - Backward compatible:
          exhaustive_matcher:
            <COLMAPFlag>: <value>
        (treated as matcher.type=exhaustive)
    """
    matcher_cfg = cfg.get("matcher", None)
    legacy_exhaustive = cfg.get("exhaustive_matcher", {}) or {}

    mtype = "exhaustive"
    options: Dict[str, Any] = {}

    if isinstance(matcher_cfg, dict) and matcher_cfg:
        mtype = str(matcher_cfg.get("type", mtype)).strip().lower()
        # options can be nested under 'options' or provided directly (excluding 'type')
        if isinstance(matcher_cfg.get("options"), dict):
            options = dict(matcher_cfg.get("options") or {})
        else:
            options = {k: v for k, v in matcher_cfg.items() if k != "type"}
    elif isinstance(legacy_exhaustive, dict) and legacy_exhaustive:
        # old configs
        mtype = "exhaustive"
        options = dict(legacy_exhaustive)

    cmd_map = {
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
        "spatial": "spatial_matcher",
        "vocab_tree": "vocab_tree_matcher",
        "vocabtree": "vocab_tree_matcher",
    }
    if mtype not in cmd_map:
        _die(
            "Unknown matcher.type: %r. Supported: exhaustive, sequential, spatial, vocab_tree" % mtype
        )

    cmd = [colmap_bin, cmd_map[mtype], "--database_path", str(database_path)]
    for k, v in options.items():
        cmd += [f"--{k}", str(v)]

    return cmd, mtype

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        _die(f"Config not found: {cfg_path}")

    cfg = _load_config(cfg_path)

    # ---- colmap binary
    colmap_bin_raw = str(cfg.get("colmap_bin", "colmap"))
    colmap_bin = _autodetect_colmap_bin(cfg_path, colmap_bin_raw)
    _ensure_colmap_exists(colmap_bin, cfg_path)

    # ---- paths
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        _die("paths must be a dict.")

    colmap_dir = _resolve_path(cfg_path, str(paths.get("colmap_dir", "colmap_out")))
    images_dir = _resolve_path(cfg_path, str(paths.get("images_dir", "images")))
    database_path = _resolve_path(cfg_path, str(paths.get("database_path", colmap_dir / "database.db")))
    sparse_dir = _resolve_path(cfg_path, str(paths.get("sparse_dir", colmap_dir / "sparse")))
    undistorted_dir = _resolve_path(cfg_path, str(paths.get("undistorted_dir", colmap_dir / "undistorted")))
    logs_dir = _resolve_path(cfg_path, str(paths.get("logs_dir", colmap_dir / "logs")))

    # ---- sanity checks + required dirs
    if not images_dir.exists() or not images_dir.is_dir():
        _die(
            f"images_dir does not exist or is not a directory: {images_dir}\n"
            "Fix your YAML (paths.images_dir) or create the folder."
        )

    _ensure_dir(colmap_dir, "colmap_dir")
    _ensure_dir(logs_dir, "logs_dir")

    # Important: mapper output_path must be a directory
    _ensure_dir(sparse_dir, "sparse_dir")

    # undistorter output dir
    _ensure_dir(undistorted_dir, "undistorted_dir")

    # database parent dir
    _ensure_parent_dir(database_path, "database_path")

    # ---- options
    fe = cfg.get("feature_extractor", {}) or {}
    mapper = cfg.get("mapper", {}) or {}
    und = cfg.get("image_undistorter", {}) or {}

    # Use absolute paths for stability
    cmd1 = [
        colmap_bin,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
    ]
    for k, v in fe.items():
        cmd1 += [f"--{k}", str(v)]

        cmd2, matcher_type = _build_matcher_cmd(cfg, colmap_bin, database_path)
    cmd3 = [
        colmap_bin,
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ]
    for k, v in mapper.items():
        cmd3 += [f"--{k}", str(v)]

    sparse0 = sparse_dir / "0"
    cmd4 = [
        colmap_bin,
        "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse0),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP",
    ]
    for k, v in und.items():
        cmd4 += [f"--{k}", str(v)]

    print(f"[colmap_run] Using colmap_bin: {colmap_bin}")
    print(f"[colmap_run] images_dir: {images_dir}")
    print(f"[colmap_run] database: {database_path}")
    print(f"[colmap_run] sparse_dir: {sparse_dir}")
    print(f"[colmap_run] undistorted_dir: {undistorted_dir}")
    print(f"[colmap_run] logs_dir: {logs_dir}")

    _run_cmd(cmd1, logs_dir / "01_feature_extractor.log")
    _run_cmd(cmd2, logs_dir / f"02_{matcher_type}_matcher.log")
    _run_cmd(cmd3, logs_dir / "03_mapper.log")
    _run_cmd(cmd4, logs_dir / "04_image_undistorter.log")

    print("[colmap_run] done.")


if __name__ == "__main__":
    main()

