#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colmap_3dgs_config_wizard.py

Wizard that writes a single YAML config containing:
- COLMAP config (like colmap_config_wizard_autocomplete_v313_fixed.py)
- 3D Gaussian Splatting (3DGS) config (repo path, scene preparation, training args)

Usage:
  python tools/colmap_3dgs_config_wizard.py --out config.yaml --gpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------
# YAML load (optional)
# ---------------------------


def _load_yaml_if_possible(path: Path) -> Optional[Dict[str, Any]]:
    """Load YAML into a dict if PyYAML is available.

    Returns None if the file doesn't exist, can't be parsed, or PyYAML isn't installed.
    """
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update mapping dst with src (src wins)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _have_prompt_toolkit() -> bool:
    try:
        import prompt_toolkit  # noqa: F401
        return True
    except Exception:
        return False


def _pt_prompt_path(msg: str, default: Optional[str], *, only_directories: bool = True) -> str:
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import PathCompleter
    except Exception:
        if default is None or default == "":
            return input(f"{msg}: ").strip()
        s = input(f"{msg} [{default}]: ").strip()
        return s if s else default

    completer = PathCompleter(expanduser=True, only_directories=only_directories)
    if default is None or default == "":
        s = prompt(f"{msg}: ", completer=completer).strip()
        return s
    s = prompt(f"{msg} [{default}]: ", default=default, completer=completer).strip()
    return s if s else default


def _pt_prompt_text(msg: str, default: Optional[str]) -> str:
    if default is None or default == "":
        return input(f"{msg}: ").strip()
    s = input(f"{msg} [{default}]: ").strip()
    return s if s else default


def _dump_yaml(data: Any) -> str:
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=120)
    except Exception:
        def esc(s: str) -> str:
            if s == "" or any(c in s for c in [":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"]):
                return "'" + s.replace("'", "''") + "'"
            return s

        def dump(obj: Any, indent: int = 0) -> str:
            sp = "  " * indent
            if isinstance(obj, dict):
                out = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        out.append(f"{sp}{k}:")
                        out.append(dump(v, indent + 1).rstrip("\n"))
                    else:
                        out.append(f"{sp}{k}: {dump(v, 0).strip()}")
                return "\n".join(out) + "\n"
            if isinstance(obj, list):
                out = []
                for v in obj:
                    if isinstance(v, (dict, list)):
                        out.append(f"{sp}-")
                        out.append(dump(v, indent + 1).rstrip("\n"))
                    else:
                        out.append(f"{sp}- {dump(v, 0).strip()}")
                return "\n".join(out) + "\n"
            if isinstance(obj, bool):
                return ("true" if obj else "false") + "\n"
            if isinstance(obj, (int, float)):
                return str(obj) + "\n"
            if obj is None:
                return "null\n"
            return esc(str(obj)) + "\n"

        return dump(data)


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


def _prompt(msg: str, default: Optional[str] = None) -> str:
    return _pt_prompt_text(msg, default)


def _prompt_dir(msg: str, default: Optional[str] = None) -> str:
    return _pt_prompt_path(msg, default, only_directories=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _bool_prompt(msg: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = input(f"{msg} [y/n] [{d}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true", "t")


def _int_prompt(msg: str, default: int) -> int:
    while True:
        s = _prompt(msg, str(default))
        try:
            return int(s)
        except ValueError:
            print("  -> please enter an integer")



def _extract_flag_value(tokens: list[str], flag: str) -> Optional[str]:
    """Return the value following `flag` in tokens, or None."""
    try:
        i = tokens.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(tokens):
        return None
    return str(tokens[i + 1])

def _remove_flag_and_value(tokens: list[str], flag: str) -> list[str]:
    """Remove all occurrences of `flag` and its following value (if any)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == flag:
            i += 1
            if i < len(tokens):
                i += 1
            continue
        out.append(tokens[i])
        i += 1
    return out



def _first_existing_dir(repo_root: Path, rel_candidates: list[str]) -> Optional[Path]:
    for rel in rel_candidates:
        p = (repo_root / rel).resolve()
        if p.is_dir():
            return p
    return None


def _detect_colmap_bin(repo_root: Path) -> str:
    cand = repo_root / ".third_party" / "colmap_cuda" / "install" / "bin" / "colmap"
    return str(cand) if cand.exists() else "colmap"


def _detect_3dgs_repo(repo_root: Path) -> str:
    cand = repo_root / ".third_party" / "3dgs" / "gaussian-splatting"
    return str(cand)

def _detect_3dgs_venv(repo_root: Path) -> Optional[Path]:
    # Prefer dedicated 3DGS venv, then common project venvs.
    for rel in [".venv_3dgs", ".venv-3dgs", "venv_3dgs", ".venv"]:
        p = (repo_root / rel).resolve()
        if (p / "bin" / "python").exists():
            return p
    return None


def _prompt_file(msg: str, default: Optional[str] = None) -> str:
    # Path prompt that allows files (for python executable).
    return _pt_prompt_path(msg, default, only_directories=False)


def _path_from_user_input(s: str, repo_root: Path) -> Path:
    p = Path(s).expanduser()
    return (repo_root / p).resolve() if not p.is_absolute() else p.resolve()



def _choose_work_subdir(base_work: Path, rel: str) -> Path:
    """Choose between <base_work>/<rel> and <base_work>/work/<rel>.

    Many repos use either:
      works2/00_frames_raw
    or:
      works2/work/00_frames_raw

    If the '/work' variant exists already, prefer it.
    """
    a = (base_work / rel).resolve()
    b = (base_work / "work" / rel).resolve()
    if b.exists():
        return b
    return a


def build_default_config(repo_root: Path, gpu: bool) -> Dict[str, Any]:
    work_dir_default = "work"
    images_default = _first_existing_dir(
        repo_root,
        ["works2/work/00_frames_raw", "works/work/00_frames_raw", "works/images"],
    ) or (repo_root / "works" / "images").resolve()

    colmap_dir_default = _first_existing_dir(
        repo_root,
        [
            f"{work_dir_default}/colmap_out",
            "works2/colmap_out",
            "works/colmap_out",
        ],
    ) or (repo_root / work_dir_default / "colmap_out").resolve()
    scene_dir_default = (repo_root / work_dir_default / "3dgs_scene").resolve()

    return {
        "work_dir": "work",
        "extract_frames": {
            "in_dir": "inputs/videos",
            "out_dir": "work/00_frames_raw",
            "fps": 2.0,
            "scale": None,
            "ext": "jpg",
            "video_exts": "mp4,mov,mkv,webm",
        },
        "filter_frames": {
            "in_dir": "work/00_frames_raw",
            "out_scored": "work/10_frames_scored",
            "out_keep": "work/11_frames_keep",
            "out_reject": "work/12_frames_reject",
            "min_brightness": 35,
            "max_brightness": 215,
            "max_dark_ratio": 0.60,
            "max_bright_ratio": 0.60,
            "min_contrast": 18,
            "min_sharpness": 120,
            "window_size": 0,
            "keep_per_window": 3,
            "dedupe": True,
            "dedupe_phash_dist": 6,
            "move": False,
            "max_images": 0,
        },
        "colmap": {
            "colmap_bin": _detect_colmap_bin(repo_root),
            "paths": {
                "images_dir": str(images_default),
                "colmap_dir": str(colmap_dir_default),
                "database_path": str(colmap_dir_default / "database.db"),
                "sparse_dir": str(colmap_dir_default / "sparse"),
                "dense_dir": str(colmap_dir_default / "dense"),
                "undistorted_dir": str(colmap_dir_default / "undistorted"),
                "logs_dir": str(colmap_dir_default / "logs"),
            },
            "feature_extractor": {
                "ImageReader.single_camera": 1,
                "ImageReader.camera_model": "OPENCV",
                "FeatureExtraction.use_gpu": 1 if gpu else 0,
                "FeatureExtraction.gpu_index": 0 if gpu else -1,
                "SiftExtraction.max_image_size": 3200,
                "SiftExtraction.max_num_features": 8192,
                "SiftExtraction.first_octave": -1,
            },
            "matcher": {
                "type": "exhaustive",
                "FeatureMatching.use_gpu": 1 if gpu else 0,
                "FeatureMatching.gpu_index": 0 if gpu else -1,
                "SiftMatching.max_ratio": 0.8,
                "SiftMatching.max_distance": 0.7,
                "SiftMatching.cross_check": 1,
                "SequentialMatching.overlap": 10,
                "SequentialMatching.loop_detection": 0,
            },
            "mapper": {
                "Mapper.min_num_matches": 15,
                "Mapper.ignore_watermarks": 1,
                "Mapper.ba_refine_focal_length": 1,
                "Mapper.ba_refine_principal_point": 0,
                "Mapper.ba_refine_extra_params": 1,
            },
            "post": {"run_undistort": True, "export_ply": True, "run_model_analyzer": False},
        },
        "three_dgs": {
            "repo_dir": _detect_3dgs_repo(repo_root),
            "venv_dir": str((_detect_3dgs_venv(repo_root) or (repo_root / ".venv_3dgs").resolve())),
            "python": None,
            "scene_dir": str(scene_dir_default),
            "work_dir": str((scene_dir_default.parent).resolve()),
            "output_dir": str(((scene_dir_default.parent) / "3dgs" / "output").resolve()),

            "link_mode": "symlink",
            "source_images": "undistorted",
            "convert_model_to_bin": True,
            "train": {
                # Convenience: pick a HW profile in the wizard (safe for downstream to ignore).
                "preset": "t4" if gpu else "1060",
                "iterations": 30000,
                "eval": True,
                # Optional: extra args passed to train.py by your runner (if you implement it).
                "extra_args": [],
                # Optional knobs (safe for downstream to ignore):
                "resolution": 1,
                "num_workers": 4,
            },
        },
    }


def _three_dgs_presets() -> Dict[str, Dict[str, Any]]:
    """Reasonable *starting points* for the wizard.

    Keep them generic so they don't depend on a specific gaussian-splatting fork/version.
    """
    return {
        "1060": {
            "train": {
                "preset": "1060",
                # 3GB VRAM class GPUs: keep things conservative by default.
                # - Use strong downscale to avoid the built-in 1.6K auto-resize and reduce VRAM.
                # - Stop densification earlier to avoid late-iteration VRAM spikes (optimizer state grows).
                "iterations": 5000,
                "resolution": 4,
                "num_workers": 2,
                "extra_args": [
                    "--densify_from_iter", "500",
                    "--densify_until_iter", "3000",
                    "--densification_interval", "200",
                    "--percent_dense", "0.005",
                ],
            }
        },
        "t4": {
            "train": {
                "preset": "t4",
                "iterations": 30000,
                "resolution": 1,
                "num_workers": 4,
                "extra_args": [],
            }
        },
        "l4": {
            "train": {
                "preset": "l4",
                "iterations": 35000,
                "resolution": 1,
                "num_workers": 6,
                "extra_args": [],
            }
        },
    }


def interactive_edit(cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    print("\n== COLMAP + 3DGS config wizard ==")
    if _have_prompt_toolkit():
        print("Tip: directory inputs support TAB completion.\n")

    c = cfg["colmap"]
    t = cfg["three_dgs"]

    # ---------------------------
    # Work directory + pre-COLMAP steps (extract/filter)
    # ---------------------------
    print("\n== Work directory / Preprocessing ==")

    # Base work dir (relative to repo root allowed). Default is "work".
    work_default = str(cfg.get("work_dir") or "work")
    work_s = _prompt_dir("Work directory (base for extracted/scored frames)", work_default)
    work_p = _path_from_user_input(work_s, repo_root)
    _ensure_dir(work_p)
    cfg["work_dir"] = str(work_p)

    # If COLMAP/3DGS sections still have legacy defaults, update them to use this work_dir.
    try:
        # COLMAP defaults
        c_paths = c.get("paths", {}) if isinstance(c, dict) else {}
        if isinstance(c_paths, dict):
            # If images_dir not explicitly set, prefer filtered keep dir, else extracted raw
            if not c_paths.get("images_dir"):
                c_paths["images_dir"] = str(_path_from_user_input(str(Path(cfg["work_dir"]) / "11_frames_keep"), repo_root))
            # If colmap_dir missing or points to old "works/colmap_out", reset to <work_dir>/colmap_out
            colmap_dir_cur = str(c_paths.get("colmap_dir") or "")
            if (not colmap_dir_cur) or ("/works/colmap_out" in colmap_dir_cur) or ("/works2/colmap_out" in colmap_dir_cur):
                c_paths["colmap_dir"] = str(_path_from_user_input(str(Path(cfg["work_dir"]) / "colmap_out"), repo_root))
            c["paths"] = c_paths

        # 3DGS defaults
        if isinstance(t, dict):
            # default work_dir for 3DGS uses the global cfg work_dir
            if not t.get("work_dir"):
                t["work_dir"] = str(_path_from_user_input(cfg["work_dir"], repo_root))
            if not t.get("output_dir"):
                t["output_dir"] = str(_path_from_user_input(str(Path(cfg["work_dir"]) / "3dgs" / "output"), repo_root))
            if not t.get("scene_dir"):
                t["scene_dir"] = str(_path_from_user_input(str(Path(cfg.get("work_dir") or "work") / "3dgs_scene"), repo_root))
    except Exception:
        pass

    ef = cfg.get("extract_frames", {}) or {}
    ff = cfg.get("filter_frames", {}) or {}
    cfg["extract_frames"] = ef
    cfg["filter_frames"] = ff

    # If we just changed work_dir, refresh pre-processing path defaults that still point to the old "work/..." layout.
    base_work = Path(cfg["work_dir"])

    # videos input default: <work_dir>/inputs/videos if it exists, else keep "inputs/videos"
    default_videos_dir = (base_work / "inputs" / "videos").resolve()
    if (not ef.get("in_dir")) or str(ef.get("in_dir")) in ("inputs/videos", "inputs\\videos"):
        if default_videos_dir.exists():
            ef["in_dir"] = str(default_videos_dir)

    # extracted frames output default: prefer <work_dir>/work/00_frames_raw if it exists, else <work_dir>/00_frames_raw
    if (not ef.get("out_dir")) or str(ef.get("out_dir")).startswith("work/") or str(ef.get("out_dir")).startswith("work\\"):
        ef["out_dir"] = str(_choose_work_subdir(base_work, "00_frames_raw"))

    # filter outputs default similarly
    if (not ff.get("in_dir")) or str(ff.get("in_dir")).startswith("work/") or str(ff.get("in_dir")).startswith("work\\"):
        ff["in_dir"] = ef["out_dir"]

    for key, rel in [
        ("out_scored", "10_frames_scored"),
        ("out_keep", "11_frames_keep"),
        ("out_reject", "12_frames_reject"),
    ]:
        v = ff.get(key)
        if (not v) or str(v).startswith("work/") or str(v).startswith("work\\"):
            ff[key] = str(_choose_work_subdir(base_work, rel))


    # ---- Extract frames config ----
    print("\n-- Extract frames --")
    in_default = ef.get("in_dir") or "inputs/videos"
    out_default = ef.get("out_dir") or str(Path(cfg["work_dir"]) / "00_frames_raw")
    ef["in_dir"] = _prompt_dir("Videos input directory (folder containing mp4/mov/...)", in_default)
    ef["out_dir"] = _prompt_dir("Extracted frames output directory", out_default)

    # numeric / simple params
    try:
        ef["fps"] = float(_prompt("Extract FPS (frames per second)", str(ef.get("fps", 2.0))))
    except Exception:
        ef["fps"] = 2.0
    scale_def = ef.get("scale", None)
    scale_s = _prompt("Extract scale (optional, e.g. 1920:-2) (empty = no scale)", "" if scale_def in (None, "None") else str(scale_def)).strip()
    ef["scale"] = scale_s if scale_s else None
    ef["ext"] = (_prompt("Extract image format (jpg|png)", str(ef.get("ext", "jpg"))).strip().lower() or "jpg")
    ef["video_exts"] = _prompt("Allowed video extensions (comma-separated)", str(ef.get("video_exts", "mp4,mov,mkv,webm")))

    # Ensure dirs exist
    _ensure_dir(_path_from_user_input(ef["in_dir"], repo_root))
    _ensure_dir(_path_from_user_input(ef["out_dir"], repo_root))

    # ---- Filter frames config ----
    print("\n-- Filter frames --")
    ff["in_dir"] = _prompt_dir("Filter input frames directory", ff.get("in_dir") or ef["out_dir"])
    ff["out_scored"] = _prompt_dir("Scored report directory", ff.get("out_scored") or str(Path(cfg["work_dir"]) / "10_frames_scored"))
    ff["out_keep"] = _prompt_dir("Keep output directory", ff.get("out_keep") or str(Path(cfg["work_dir"]) / "11_frames_keep"))
    ff["out_reject"] = _prompt_dir("Reject output directory", ff.get("out_reject") or str(Path(cfg["work_dir"]) / "12_frames_reject"))

    # Thresholds
    def _float_prompt(msg: str, default: float) -> float:
        s = _prompt(msg, str(default)).strip()
        try:
            return float(s)
        except Exception:
            return default

    ff["min_brightness"] = _float_prompt("min_brightness", float(ff.get("min_brightness", 35)))
    ff["max_brightness"] = _float_prompt("max_brightness", float(ff.get("max_brightness", 215)))
    ff["max_dark_ratio"] = _float_prompt("max_dark_ratio", float(ff.get("max_dark_ratio", 0.60)))
    ff["max_bright_ratio"] = _float_prompt("max_bright_ratio", float(ff.get("max_bright_ratio", 0.60)))
    ff["min_contrast"] = _float_prompt("min_contrast", float(ff.get("min_contrast", 18)))
    ff["min_sharpness"] = _float_prompt("min_sharpness", float(ff.get("min_sharpness", 120)))

    # Selection controls
    ff["window_size"] = _int_prompt("window_size (0 = disabled)", int(ff.get("window_size", 0)))
    ff["keep_per_window"] = _int_prompt("keep_per_window", int(ff.get("keep_per_window", 3)))

    ff["dedupe"] = _bool_prompt("Enable dedupe (pHash)?", bool(ff.get("dedupe", True)))
    ff["dedupe_phash_dist"] = _int_prompt("dedupe_phash_dist (<= => near-duplicate)", int(ff.get("dedupe_phash_dist", 6)))
    ff["move"] = _bool_prompt("Move images instead of copy?", bool(ff.get("move", False)))
    ff["max_images"] = _int_prompt("max_images (0 = no limit)", int(ff.get("max_images", 0)))

    _ensure_dir(_path_from_user_input(ff["in_dir"], repo_root))
    _ensure_dir(_path_from_user_input(ff["out_scored"], repo_root))
    _ensure_dir(_path_from_user_input(ff["out_keep"], repo_root))
    _ensure_dir(_path_from_user_input(ff["out_reject"], repo_root))


    c["colmap_bin"] = _prompt("COLMAP binary", c["colmap_bin"])

    while True:
        images_s = _prompt_dir("Images directory (must exist)", c["paths"]["images_dir"])
        images_p = _path_from_user_input(images_s, repo_root)
        if images_p.is_dir():
            c["paths"]["images_dir"] = str(images_p)
            break
        print(f"[wizard] ERROR: directory does not exist: {images_p}\n")

    colmap_dir_s = _prompt_dir("COLMAP output directory (will be created)", c["paths"]["colmap_dir"])
    colmap_dir_p = _path_from_user_input(colmap_dir_s, repo_root)
    _ensure_dir(colmap_dir_p)

    c["paths"]["colmap_dir"] = str(colmap_dir_p)
    c["paths"]["database_path"] = str(colmap_dir_p / "database.db")
    c["paths"]["sparse_dir"] = str(colmap_dir_p / "sparse")
    c["paths"]["dense_dir"] = str(colmap_dir_p / "dense")
    c["paths"]["undistorted_dir"] = str(colmap_dir_p / "undistorted")
    c["paths"]["logs_dir"] = str(colmap_dir_p / "logs")

    use_gpu = _bool_prompt("Use GPU for SIFT feature extraction / matching?", bool(c["feature_extractor"].get("FeatureExtraction.use_gpu", 0)))
    c["feature_extractor"]["FeatureExtraction.use_gpu"] = 1 if use_gpu else 0
    c["matcher"]["FeatureMatching.use_gpu"] = 1 if use_gpu else 0

    gpu_index = _int_prompt("GPU index (0 = first GPU)", int(c["feature_extractor"].get("FeatureExtraction.gpu_index", 0))) if use_gpu else -1
    c["feature_extractor"]["FeatureExtraction.gpu_index"] = gpu_index
    c["matcher"]["FeatureMatching.gpu_index"] = gpu_index

    c["feature_extractor"]["ImageReader.single_camera"] = 1 if _bool_prompt("Assume a single camera (recommended for video frames)?", True) else 0
    c["feature_extractor"]["ImageReader.camera_model"] = _prompt(
        "Camera model (PINHOLE | OPENCV | OPENCV_FISHEYE | SIMPLE_RADIAL ...)",
        c["feature_extractor"].get("ImageReader.camera_model", "OPENCV"),
    )

    c["feature_extractor"]["SiftExtraction.max_image_size"] = _int_prompt("SiftExtraction.max_image_size (px; reduce if OOM)", int(c["feature_extractor"].get("SiftExtraction.max_image_size", 3200)))
    c["feature_extractor"]["SiftExtraction.max_num_features"] = _int_prompt("SiftExtraction.max_num_features", int(c["feature_extractor"].get("SiftExtraction.max_num_features", 8192)))

    mtype = _prompt("Matcher type (exhaustive | sequential)", c["matcher"].get("type", "exhaustive")).lower().strip()
    if mtype not in ("exhaustive", "sequential"):
        print("[wizard] Unknown matcher type; using 'exhaustive'")
        mtype = "exhaustive"
    c["matcher"]["type"] = mtype

    if mtype == "sequential":
        c["matcher"]["SequentialMatching.overlap"] = _int_prompt("SequentialMatching.overlap (frames overlap window)", int(c["matcher"].get("SequentialMatching.overlap", 10)))
        c["matcher"]["SequentialMatching.loop_detection"] = 1 if _bool_prompt("SequentialMatching.loop_detection?", bool(c["matcher"].get("SequentialMatching.loop_detection", 0))) else 0

    c["mapper"]["Mapper.min_num_matches"] = _int_prompt("Mapper.min_num_matches", int(c["mapper"].get("Mapper.min_num_matches", 15)))
    c["mapper"]["Mapper.ba_refine_focal_length"] = 1 if _bool_prompt("Refine focal length (BA)?", bool(c["mapper"].get("Mapper.ba_refine_focal_length", 1))) else 0
    c["mapper"]["Mapper.ba_refine_principal_point"] = 1 if _bool_prompt("Refine principal point (BA)?", bool(c["mapper"].get("Mapper.ba_refine_principal_point", 0))) else 0
    c["mapper"]["Mapper.ba_refine_extra_params"] = 1 if _bool_prompt("Refine extra params (BA)?", bool(c["mapper"].get("Mapper.ba_refine_extra_params", 1))) else 0

    c["post"]["run_undistort"] = _bool_prompt("Run image_undistorter (dense prep)?", bool(c["post"].get("run_undistort", True)))
    c["post"]["export_ply"] = _bool_prompt("Export PLY (sparse cloud)?", bool(c["post"].get("export_ply", True)))
    c["post"]["run_model_analyzer"] = _bool_prompt("Run model_analyzer?", bool(c["post"].get("run_model_analyzer", False)))

    print("\n== 3D Gaussian Splatting ==")

    # Preset selection (1060 / T4 / L4) to seed defaults.
    presets = _three_dgs_presets()
    cur_preset = (cfg.get("three_dgs", {}).get("train", {}) or {}).get("preset")
    cur_preset_s = str(cur_preset or "t4").lower()
    if cur_preset_s not in presets:
        cur_preset_s = "t4"
    preset = _prompt("3DGS preset (1060 | t4 | l4)", cur_preset_s).strip().lower() or cur_preset_s
    if preset not in presets:
        print("[wizard] Unknown preset; using 't4'")
        preset = "t4"

    # Apply preset seed, then overlay whatever is already in cfg (so existing YAML wins).
    seeded = {"train": {}}
    _deep_update(seeded, presets[preset])
    seeded.setdefault("train", {})["preset"] = preset
    cfg["three_dgs"] = _deep_update(seeded, cfg.get("three_dgs", {}))
    t = cfg["three_dgs"]
    t["repo_dir"] = _prompt_dir("3DGS repo directory (gaussian-splatting)", t.get("repo_dir"))

    # Optional: venv/python used by 3dgs_train to run train.py (useful when main project venv differs).
    venv_default = t.get("venv_dir") or str((_detect_3dgs_venv(repo_root) or (repo_root / ".venv_3dgs").resolve()))
    venv_s = _prompt_dir("3DGS venv directory (optional, used by 3dgs_train)", venv_default)
    venv_p = _path_from_user_input(venv_s, repo_root)
    t["venv_dir"] = str(venv_p)

    py_default = t.get("python") or (str((venv_p / "bin" / "python").resolve()) if (venv_p / "bin" / "python").exists() else "")
    py_s = _prompt_file("3DGS python executable (optional; leave empty to use venv/bin/python)", py_default).strip()
    if py_s:
        py_p = _path_from_user_input(py_s, repo_root)
        t["python"] = str(py_p)
    else:
        t["python"] = None

    # Base working directory for 3DGS artifacts (recommended: your project work folder).
    work_default = t.get("work_dir")
    if not work_default:
        sd = t.get("scene_dir")
        if sd:
            try:
                work_default = str(Path(sd).expanduser().resolve().parent)
            except Exception:
                work_default = None
    if not work_default:
        work_default = str(_path_from_user_input(str(cfg.get("work_dir") or "work"), repo_root))
    # Use the global work_dir chosen at the beginning of the wizard.
    # (If you need a different base for 3DGS only, set three_dgs.work_dir manually in YAML.)
    work_p = _path_from_user_input(str(cfg.get("work_dir") or "work"), repo_root)
    _ensure_dir(work_p)
    t["work_dir"] = str(work_p)

    out_default = t.get("output_dir") or str((work_p / "3dgs" / "output").resolve())
    out_s = _prompt_dir("3DGS output directory (will be created)", out_default)
    out_p = _path_from_user_input(out_s, repo_root)
    _ensure_dir(out_p)
    t["output_dir"] = str(out_p)

    if not t.get("scene_dir"):
        t["scene_dir"] = str(_path_from_user_input(str(Path(cfg.get("work_dir") or "work") / "3dgs_scene"), repo_root))

    scene_s = _prompt_dir("3DGS scene directory (will be created)", t.get("scene_dir"))
    scene_p = _path_from_user_input(scene_s, repo_root)
    _ensure_dir(scene_p)
    t["scene_dir"] = str(scene_p)

    t["link_mode"] = (_prompt("Scene prepare mode (symlink | copy)", t.get("link_mode", "symlink")) or "symlink").strip().lower()
    if t["link_mode"] not in ("symlink", "copy"):
        print("[wizard] Unknown link_mode, using 'symlink'")
        t["link_mode"] = "symlink"

    t["source_images"] = (_prompt("Use images from (undistorted | original)", t.get("source_images", "undistorted")) or "undistorted").strip().lower()
    if t["source_images"] not in ("undistorted", "original"):
        print("[wizard] Unknown source_images, using 'undistorted'")
        t["source_images"] = "undistorted"

    t["convert_model_to_bin"] = _bool_prompt("Convert COLMAP model to BIN if needed?", bool(t.get("convert_model_to_bin", True)))

    tr = t.get("train", {}) or {}

    tr["preset"] = preset
    tr["iterations"] = _int_prompt("3DGS training iterations", int(tr.get("iterations", 30000)))
    tr["eval"] = _bool_prompt("3DGS --eval?", bool(tr.get("eval", True)))

    # Optional generic knobs (safe for downstream to ignore)
    tr["resolution"] = _int_prompt(
        "3DGS resolution downscale (1 = full, 2 = half, ...)",
        int(tr.get("resolution", 1)),
    )
    tr["num_workers"] = _int_prompt("3DGS data loader workers", int(tr.get("num_workers", 4)))

    # Densification controls (important for low VRAM GPUs like GTX 1060 3GB).
    # We store them into extra_args to stay compatible with vanilla gaussian-splatting.
    extra_tokens = [str(x) for x in (tr.get("extra_args") or [])]
    
    # If the existing YAML already contains these flags inside extra_args, use them as defaults.
    densify_from_default = _extract_flag_value(extra_tokens, "--densify_from_iter")
    densify_until_default = _extract_flag_value(extra_tokens, "--densify_until_iter")
    densify_interval_default = _extract_flag_value(extra_tokens, "--densification_interval")
    percent_dense_default = _extract_flag_value(extra_tokens, "--percent_dense")
    
    # Remove them from "additional" args so we don't duplicate them.
    extra_tokens_clean = _remove_flag_and_value(extra_tokens, "--densify_from_iter")
    extra_tokens_clean = _remove_flag_and_value(extra_tokens_clean, "--densify_until_iter")
    extra_tokens_clean = _remove_flag_and_value(extra_tokens_clean, "--densification_interval")
    extra_tokens_clean = _remove_flag_and_value(extra_tokens_clean, "--percent_dense")
    
    densify_from = _int_prompt("3DGS densify_from_iter (start densification)", int(densify_from_default or 500))
    densify_until = _int_prompt("3DGS densify_until_iter (stop densification)", int(densify_until_default or 3000))
    densify_interval = _int_prompt("3DGS densification_interval", int(densify_interval_default or 200))
    # percent_dense is a float in many forks; we keep it as string to avoid locale issues.
    percent_dense = _prompt("3DGS percent_dense (e.g. 0.005)", percent_dense_default or "0.005").strip() or "0.005"
    
    # Now ask for any additional args (space-separated), excluding the densify flags above.
    extra_default = " ".join([str(x) for x in extra_tokens_clean])
    extra_s = _prompt("3DGS extra args (additional, space-separated; optional)", extra_default).strip()
    extra_additional = extra_s.split() if extra_s else extra_tokens_clean
    
    # Rebuild extra_args with densify flags first, then additional.
    tr["extra_args"] = [
    "--densify_from_iter", str(densify_from),
    "--densify_until_iter", str(densify_until),
    "--densification_interval", str(densify_interval),
    "--percent_dense", str(percent_dense),
    *[str(x) for x in extra_additional],
    ]
    
    t["train"] = tr

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colmap_3dgs.yaml", help="Output YAML path")
    ap.add_argument("--gpu", action="store_true", help="Prefer GPU defaults for COLMAP")
    ap.add_argument("--noninteractive", action="store_true", help="Do not prompt; just write defaults")
    args = ap.parse_args()

    repo_root = _repo_root_from_this_file()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    # If the YAML already exists, we use it as defaults and only fill missing keys.
    cfg = build_default_config(repo_root, gpu=args.gpu)
    existing = _load_yaml_if_possible(out_path)
    if existing:
        cfg = _deep_update(cfg, existing)
    if not args.noninteractive:
        cfg = interactive_edit(cfg, repo_root)
    else:
        # Ensure work/extract/filter directories exist (noninteractive defaults)
        work_dir = _path_from_user_input(str(cfg.get("work_dir") or "work"), repo_root)
        _ensure_dir(work_dir)
        cfg["work_dir"] = str(work_dir)

        ef = cfg.get("extract_frames", {}) or {}
        ff = cfg.get("filter_frames", {}) or {}
        # Resolve default paths relative to repo root if needed
        if ef.get("out_dir"):
            _ensure_dir(_path_from_user_input(str(ef["out_dir"]), repo_root))
        if ff.get("out_scored"):
            _ensure_dir(_path_from_user_input(str(ff["out_scored"]), repo_root))
        if ff.get("out_keep"):
            _ensure_dir(_path_from_user_input(str(ff["out_keep"]), repo_root))
        if ff.get("out_reject"):
            _ensure_dir(_path_from_user_input(str(ff["out_reject"]), repo_root))

        colmap_dir = _path_from_user_input(cfg["colmap"]["paths"]["colmap_dir"], repo_root)
        _ensure_dir(colmap_dir)
        cfg["colmap"]["paths"]["colmap_dir"] = str(colmap_dir)
        cfg["colmap"]["paths"]["database_path"] = str(colmap_dir / "database.db")
        cfg["colmap"]["paths"]["sparse_dir"] = str(colmap_dir / "sparse")
        cfg["colmap"]["paths"]["dense_dir"] = str(colmap_dir / "dense")
        cfg["colmap"]["paths"]["undistorted_dir"] = str(colmap_dir / "undistorted")
        cfg["colmap"]["paths"]["logs_dir"] = str(colmap_dir / "logs")
        scene_dir = _path_from_user_input(cfg["three_dgs"]["scene_dir"], repo_root)
        _ensure_dir(scene_dir)
        cfg["three_dgs"]["scene_dir"] = str(scene_dir)

        venv_dir = _path_from_user_input(cfg["three_dgs"].get("venv_dir") or str((repo_root / ".venv_3dgs").resolve()), repo_root)
        cfg["three_dgs"]["venv_dir"] = str(venv_dir)
        # keep cfg["three_dgs"]["python"] as-is (None by default)

    # Final write choice: overwrite or pick a new name.
    if not args.noninteractive and out_path.exists():
        ow = input(f"[wizard] '{out_path.name}' already exists. Overwrite? [y/N] ").strip().lower()
        if ow not in ("y", "yes"):
            new_name = input("[wizard] New filename (relative or absolute path): ").strip()
            if new_name:
                new_path = Path(new_name).expanduser()
                out_path = new_path if new_path.is_absolute() else (repo_root / new_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_dump_yaml(cfg), encoding="utf-8")

    print(f"[wizard] wrote: {out_path}")
    print(f"[wizard] colmap.images_dir: {cfg['colmap']['paths']['images_dir']}")
    print(f"[wizard] colmap.colmap_dir:  {cfg['colmap']['paths']['colmap_dir']}")
    print(f"[wizard] 3dgs.scene_dir:     {cfg['three_dgs']['scene_dir']}")
    print(f"[wizard] 3dgs.repo_dir:      {cfg['three_dgs']['repo_dir']}")
    
if __name__ == "__main__":
    main()
