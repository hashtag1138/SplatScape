#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3dgs_prepare_fixed4.py

Prepare a 3D Gaussian Splatting "scene" directory from a COLMAP output, using a YAML config.

It will:
- choose images source (undistorted/images if available, else original images_dir)
- auto-flatten nested "images/images/*" (if your images_dir points to a dataset root that contains an images/ folder)
- create scene_dir layout:
    scene_dir/
      images/            <-- actual image files live here
      sparse/0/          <-- COLMAP model (BIN preferred)
- place images/ and sparse/0/ via symlink or copy
- convert COLMAP model to BIN if needed (if only *.txt exists)

Usage:
  python tools/3dgs_prepare_fixed4.py --config test2.yaml --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _die(msg: str, code: int = 2) -> None:
    print(f"[3dgs_prepare] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _info(msg: str) -> None:
    print(f"[3dgs_prepare] {msg}")


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        _die(f"Failed to read YAML (install PyYAML). Error: {e}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rm_if_exists(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        _rm_if_exists(dst)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        _die(f"Unknown link_mode: {mode}")


def _looks_like_bin_model(model0: Path) -> bool:
    return (model0 / "cameras.bin").exists() and (model0 / "images.bin").exists()


def _looks_like_txt_model(model0: Path) -> bool:
    return (model0 / "cameras.txt").exists() and (model0 / "images.txt").exists()


def _convert_to_bin(colmap_bin: str, model0: Path) -> None:
    _info(f"Converting model to BIN: {model0}")
    _run(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model0),
            "--output_path",
            str(model0),
            "--output_type",
            "BIN",
        ]
    )


def _has_images_here(p: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    try:
        for it in p.iterdir():
            if it.is_file() and it.suffix.lower() in exts:
                return True
    except Exception:
        return False
    return False


def _flatten_images_dir(src_images: Path) -> Path:
    """
    If src_images is a dataset root like:
        src_images/
          images/
            frame0001.jpg
    then return src_images/images.
    Otherwise return src_images unchanged.
    """
    if src_images.is_dir() and not _has_images_here(src_images):
        nested = src_images / "images"
        if nested.is_dir() and _has_images_here(nested):
            _info(f"Detected nested images/ folder; using: {nested}")
            return nested
    return src_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite existing scene links/copies")
    args = ap.parse_args()

    _info(f"config: {args.config}")

    cfg = _load_yaml(Path(args.config).expanduser())
    col = cfg.get("colmap", cfg)
    t = cfg.get("three_dgs") or cfg.get("3dgs") or {}
    if not t:
        _die("Missing 'three_dgs' section in YAML.")

    colmap_bin = str(col.get("colmap_bin") or "colmap")
    paths = col.get("paths", {})
    images_dir = Path(paths.get("images_dir", "")).expanduser()
    sparse_dir = Path(paths.get("sparse_dir", "")).expanduser()
    undistorted_dir = Path(paths.get("undistorted_dir", "")).expanduser()

    if not images_dir.is_dir():
        _die(f"images_dir does not exist: {images_dir}")

    # Determine COLMAP model folder
    model0 = sparse_dir / "0"
    if not model0.exists():
        if (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists():
            model0 = sparse_dir
        else:
            _die(f"COLMAP sparse model not found under: {sparse_dir} (expected sparse/0 or model files directly)")

    # Choose images source
    source_images_mode = str(t.get("source_images", "undistorted")).strip().lower()
    und_images = undistorted_dir / "images"
    if source_images_mode == "undistorted" and und_images.is_dir():
        src_images = und_images
        _info(f"Using undistorted images: {src_images}")
    else:
        src_images = images_dir
        _info(f"Using original images: {src_images}")

    # Auto-flatten nested images/
    src_images = _flatten_images_dir(src_images)

    scene_dir = Path(t.get("scene_dir", "")).expanduser()
    if not str(scene_dir):
        _die("three_dgs.scene_dir is empty")
    _ensure_dir(scene_dir)

    link_mode = str(t.get("link_mode", "symlink")).strip().lower()
    if link_mode not in ("symlink", "copy"):
        _die("three_dgs.link_mode must be 'symlink' or 'copy'")

    scene_images = scene_dir / "images"
    scene_sparse0 = scene_dir / "sparse" / "0"
    _ensure_dir(scene_dir / "sparse")

    if (scene_images.exists() or scene_images.is_symlink()) and not args.force:
        _die(f"Scene images already exist: {scene_images} (use --force)")
    if (scene_sparse0.exists() or scene_sparse0.is_symlink()) and not args.force:
        _die(f"Scene sparse already exist: {scene_sparse0} (use --force)")

    _info(f"Preparing scene_dir: {scene_dir}")
    _link_or_copy(src_images, scene_images, link_mode)
    _link_or_copy(model0, scene_sparse0, link_mode)

    # Convert model to BIN if needed (do it on the scene copy/link target)
    if bool(t.get("convert_model_to_bin", True)):
        target_model0 = scene_sparse0
        if _looks_like_txt_model(target_model0) and not _looks_like_bin_model(target_model0):
            _convert_to_bin(colmap_bin, target_model0)
        else:
            _info("Model already BIN (or both BIN+TXT present).")

    _info("Done.")
    print(f"[3dgs_prepare] scene_dir: {scene_dir}")
    print(f"[3dgs_prepare] images:    {scene_images}")
    print(f"[3dgs_prepare] sparse/0:   {scene_sparse0}")


if __name__ == "__main__":
    main()
