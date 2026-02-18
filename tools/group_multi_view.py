#!/usr/bin/env python3
"""
group_multi_view.py
Reorder multi-angle extracted frames into time-synchronized sequences for COLMAP.

Typical use-case:
- You have N videos (0.mp4, 1.mp4, 2.mp4, 3.mp4, ...) that are time-synced and same duration.
- extract_frames.py creates:
    work/00_frames_raw/0/0_000001.jpg ...
    work/00_frames_raw/1/1_000001.jpg ...
  (i.e., one subfolder per view / clip)

This script creates a single flat folder with filenames ordered by (time_index, view_index), e.g.:
  000000_00.jpg, 000000_01.jpg, 000000_02.jpg, 000000_03.jpg,
  000001_00.jpg, 000001_01.jpg, ...

That way COLMAP sees interleaved images from all views at each timestep.

Example:
  python tools/group_multi_view.py --config config.yaml

Then point filter_frames.in_dir to the grouped output, e.g.:
  filter_frames:
    in_dir: work/01_frames_grouped

Notes / Design choices:
- We match frames by extracting the *last* integer group from the filename stem.
  Examples:
    0_000123.jpg  -> 123
    frame_0012.png -> 12
- By default we require the frame index to exist in *all* views (intersection),
  so each timestep has N images. You can relax this with --allow_missing.
- The output is flat by default (best for COLMAP images_dir). Optional --out_layout=per_frame
  creates subfolders per time index if you prefer debugging, but COLMAP typically wants flat.
"""

import argparse
import json
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time as now_time
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# ---------------------------
# Config helpers (kept consistent with your other tools)
# ---------------------------

def _get(d, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml") from e
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Config did not parse to a dict: {path}")
    return obj


def resolve_path(p: str | None, repo_root: Path, config_dir: Path) -> Path:
    if not p:
        return Path()
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    # Prefer repo_root resolution; fall back to config_dir.
    cand = (repo_root / pp).resolve()
    return cand if cand.exists() or str(pp).startswith("work") or str(pp).startswith("inputs") else (config_dir / pp).resolve()


# ---------------------------
# Core logic
# ---------------------------

INT_RE = re.compile(r"(\d+)(?!.*\d)")  # last integer group in string


def parse_frame_index(path: Path) -> Optional[int]:
    """Extract frame index from a filename by taking the last integer group in the stem."""
    m = INT_RE.search(path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


@dataclass
class ViewInfo:
    name: str            # subfolder name
    files: List[Path]    # sorted image paths
    index_map: Dict[int, Path]  # frame_idx -> path


def list_views(in_dir: Path, explicit_views: Optional[List[str]] = None) -> List[Path]:
    if explicit_views:
        out = []
        for v in explicit_views:
            p = (in_dir / v)
            if not p.is_dir():
                raise SystemExit(f"[group_multi_view] View folder not found: {p}")
            out.append(p)
        return out

    # auto-detect: direct subdirectories that contain images
    subs = sorted([p for p in in_dir.iterdir() if p.is_dir()])
    # If in_dir itself already contains images (no subdirs), treat it as single view.
    exts = {".jpg", ".jpeg", ".png"}
    has_images_at_root = any((p.is_file() and p.suffix.lower() in exts) for p in in_dir.iterdir())
    if has_images_at_root:
        return [in_dir]
    return subs


def collect_view(view_dir: Path, exts: Tuple[str, ...]) -> ViewInfo:
    files = sorted([p for p in view_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    index_map: Dict[int, Path] = {}
    dropped = 0
    for p in files:
        idx = parse_frame_index(p)
        if idx is None:
            dropped += 1
            continue
        # Keep the first occurrence if duplicates exist
        index_map.setdefault(idx, p)
    if not index_map:
        raise SystemExit(f"[group_multi_view] No usable images found in {view_dir}")
    if dropped:
        print(f"[group_multi_view] WARN: skipped {dropped} file(s) without frame index in {view_dir}")
    # Rebuild files in time order
    ordered = [index_map[k] for k in sorted(index_map.keys())]
    return ViewInfo(name=view_dir.name, files=ordered, index_map=index_map)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, do_move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path; uses group_multi_view.* keys")
    ap.add_argument("--in_dir", required=False, help="Input frames root (expects one subfolder per view)")
    ap.add_argument("--out_dir", required=False, help="Output directory")
    ap.add_argument("--views", default=None,
                    help="Comma-separated subfolder names to use (optional). Default: auto-detect all subfolders.")
    ap.add_argument("--allow_missing", action="store_true",
                    help="If set: include frame indices even if some views are missing (missing ones are skipped).")
    ap.add_argument("--move", action="store_true",
                    help="Move images instead of copy (default copy).")
    ap.add_argument("--exts", default="jpg,jpeg,png", help="Comma-separated allowed image extensions")
    ap.add_argument("--frame_digits", type=int, default=6, help="Zero-padding digits for frame index in output filenames")
    ap.add_argument("--view_digits", type=int, default=2, help="Zero-padding digits for view index in output filenames")
    ap.add_argument("--out_layout", choices=["flat", "per_frame"], default="flat",
                    help="flat => all images in out_dir; per_frame => out_dir/<frame_idx>/...")
    ap.add_argument("--dry_run", action="store_true", help="Plan only; don't copy/move files.")
    args = ap.parse_args()

    # Load defaults from --config if provided
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg = load_yaml_config(cfg_path)
        repo_root = _repo_root_from_this_file()
        config_dir = cfg_path.parent

        work_dir = _get(cfg, "work_dir") or "work"

        gm = _get(cfg, "group_multi_view") or {}

        def _p(v: str) -> str:
            return str(resolve_path(v, repo_root, config_dir))

        if not args.in_dir:
            # Default: extracted frames output dir
            ef_out = _get(cfg, "extract_frames.out_dir") or str(Path(str(work_dir)) / "00_frames_raw")
            args.in_dir = _p(str(gm.get("in_dir") or ef_out))

        if not args.out_dir:
            # Default: separate stage between extract and filter
            args.out_dir = _p(str(gm.get("out_dir") or (Path(str(work_dir)) / "01_frames_grouped")))

        if args.views is None and gm.get("views") is not None:
            v = gm.get("views")
            if isinstance(v, list):
                args.views = ",".join(str(x) for x in v)
            elif isinstance(v, str):
                args.views = v

        if gm.get("allow_missing") is not None:
            args.allow_missing = bool(gm.get("allow_missing"))
        if gm.get("move") is not None:
            args.move = bool(gm.get("move"))
        if gm.get("exts") is not None:
            args.exts = str(gm.get("exts"))
        if gm.get("frame_digits") is not None:
            args.frame_digits = int(gm.get("frame_digits"))
        if gm.get("view_digits") is not None:
            args.view_digits = int(gm.get("view_digits"))
        if gm.get("out_layout") is not None:
            args.out_layout = str(gm.get("out_layout"))
        if gm.get("dry_run") is not None:
            args.dry_run = bool(gm.get("dry_run"))

    if not args.in_dir or not args.out_dir:
        raise SystemExit("You must provide --config or both --in_dir and --out_dir.")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    exts = tuple("." + e.strip().lower() for e in args.exts.split(",") if e.strip())
    explicit_views = [v.strip() for v in args.views.split(",")] if args.views else None

    view_dirs = list_views(in_dir, explicit_views=explicit_views)
    if len(view_dirs) < 2:
        print(f"[group_multi_view] WARN: detected {len(view_dirs)} view(s). This tool is most useful with 2+ views.")

    # Collect view maps
    t0 = perf_counter()
    views: List[ViewInfo] = []
    for vd in view_dirs:
        views.append(collect_view(vd, exts=exts))

    # Compute target frame indices
    key_sets = [set(v.index_map.keys()) for v in views]
    if args.allow_missing:
        all_keys = sorted(set().union(*key_sets))
    else:
        all_keys = sorted(set.intersection(*key_sets)) if key_sets else []

    if not all_keys:
        raise SystemExit("[group_multi_view] No common frame indices found across views. "
                         "If your views are slightly misaligned, try --allow_missing.")

    # Build copy plan
    plan: List[Tuple[Path, Path, int, int]] = []  # (src, dst, frame_idx, view_idx)
    for frame_idx in all_keys:
        for view_idx, v in enumerate(views):
            src = v.index_map.get(frame_idx)
            if src is None:
                if args.allow_missing:
                    continue
                # Shouldn't happen due to intersection, but keep safe.
                raise SystemExit(f"[group_multi_view] Internal error: missing frame {frame_idx} for view {v.name}")

            suffix = src.suffix.lower()
            fname = f"{frame_idx:0{args.frame_digits}d}_{view_idx:0{args.view_digits}d}{suffix}"

            if args.out_layout == "flat":
                dst = out_dir / fname
            else:
                dst = out_dir / f"{frame_idx:0{args.frame_digits}d}" / fname

            plan.append((src, dst, frame_idx, view_idx))

    # Execute
    io_t0 = perf_counter()
    manifest = []
    for src, dst, frame_idx, view_idx in tqdm(plan, desc="Grouping", unit="img"):
        manifest.append({
            "dst": str(dst),
            "src": str(src),
            "frame_idx": int(frame_idx),
            "view_idx": int(view_idx),
            "view_name": views[view_idx].name if view_idx < len(views) else str(view_idx),
        })
        if args.dry_run:
            continue
        copy_or_move(src, dst, do_move=args.move)

    io_dt = perf_counter() - io_t0
    total_dt = perf_counter() - t0

    # Write manifest + perf
    manifest_path = out_dir / "manifest.jsonl"
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in manifest:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    perf = {
        "script": "group_multi_view.py",
        "timestamp": now_time(),
        "total_time_sec": total_dt,
        "io_time_sec": io_dt,
        "count_views": len(views),
        "view_names": [v.name for v in views],
        "count_frames": len(all_keys),
        "count_images_out": len(plan),
        "mode": "move" if args.move else "copy",
        "allow_missing": bool(args.allow_missing),
        "out_layout": args.out_layout,
        "args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    perf_path = out_dir / "perf.json"
    if not args.dry_run:
        perf_path.write_text(json.dumps(perf, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] views={len(views)} frames={len(all_keys)} images_out={len(plan)} total_time_sec={total_dt:.2f}")
    if args.dry_run:
        print("[DRY_RUN] nothing was written.")
    else:
        print(f"[OUT]  grouped images:  {out_dir}")
        print(f"[OUT]  manifest:       {manifest_path}")
        print(f"[PERF] wrote:         {perf_path}")
        print("\nNext step:")
        print("  - Set filter_frames.in_dir to this out_dir in your YAML, then run filter_frames.py.")
        print(f"    Example:\n      filter_frames:\n        in_dir: {out_dir}")

if __name__ == "__main__":
    main()
