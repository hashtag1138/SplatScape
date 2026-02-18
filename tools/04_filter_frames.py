#!/usr/bin/env python3
"""
filter_frames.py
Frames -> (scored report) + keep/reject folders + manifests
Progress: scoring + selection + file copy, with total time, plus perf.json.

Key features:
- dark/bright/contrast/sharpness scoring
- optional dedupe using perceptual hash (pHash)
- optional "best-of per window" selection (keep top-K every N frames per folder)
- writes:
  - work/10_frames_scored/scored.csv + scored.jsonl
  - work/11_frames_keep/images + manifest.jsonl
  - work/12_frames_reject/images + manifest.jsonl
  - perf.json in scored dir

Example:
  python tools/filter_frames.py \
    --in_dir work/00_frames_raw \
    --out_scored work/10_frames_scored \
    --out_keep work/11_frames_keep \
    --out_reject work/12_frames_reject \
    --min_sharpness 120 \
    --dedupe --dedupe_phash_dist 6 \
    --window_size 20 --keep_per_window 3
"""

def _apply_sharpness_percentile(good, top_percent: float):
    """Keep only the top_percent sharpest items from `good` (list of dicts).

    Returns (kept, (threshold, dropped)) where dropped are the remaining items.
    If top_percent >= 100, returns (good, None).
    """
    p = float(top_percent)
    if p <= 0:
        return [], (0.0, good)
    if p >= 100:
        return good, None
    good_sorted = sorted(good, key=lambda r: float(r.get("sharpness", 0.0)), reverse=True)
    k = max(1, int(round(len(good_sorted) * (p / 100.0))))
    kept = good_sorted[:k]
    thr = float(kept[-1].get("sharpness", 0.0)) if kept else 0.0
    dropped = good_sorted[k:]
    return kept, (thr, dropped)

import argparse
import csv
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm


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
    # Prefer repo_root resolution (matches rest of your tools); fall back to config_dir.
    cand = (repo_root / pp).resolve()
    return cand if cand.exists() or str(pp).startswith("work") or str(pp).startswith("inputs") else (config_dir / pp).resolve()



@dataclass
class Score:
    brightness: float
    contrast: float
    sharpness: float
    dark_ratio: float
    bright_ratio: float
    phash: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_scores(img_path: Path) -> Score:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imread failed")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())

    dark_ratio = float((gray < 20).mean())
    bright_ratio = float((gray > 235).mean())

    pil = Image.open(img_path).convert("RGB")
    ph = str(imagehash.phash(pil))  # hex string
    return Score(brightness, contrast, sharpness, dark_ratio, bright_ratio, ph)


def phash_dist(a: str, b: str) -> int:
    x = int(a, 16) ^ int(b, 16)
    return x.bit_count()


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def flatten_relpath(rel: Path) -> str:
    # clip01/frame_0001.jpg -> clip01_frame_0001.jpg
    return "_".join(rel.parts)


def copy_or_move(src: Path, dst: Path, do_move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path; uses filter_frames.* keys")

    ap.add_argument("--in_dir", required=False, help="Root folder containing extracted frames (recursive)")
    ap.add_argument("--out_scored", required=False, help="Where to write scored.csv/scored.jsonl and perf.json")
    ap.add_argument("--out_keep", required=False, help="Keep output folder (will create images/ + manifest.jsonl)")
    ap.add_argument("--out_reject", required=False, help="Reject output folder (will create images/ + manifest.jsonl)")

    # Thresholds
    ap.add_argument("--min_brightness", type=float, default=35)
    ap.add_argument("--max_brightness", type=float, default=215)
    ap.add_argument("--max_dark_ratio", type=float, default=0.60)
    ap.add_argument("--max_bright_ratio", type=float, default=0.60)
    ap.add_argument("--min_contrast", type=float, default=18)
    ap.add_argument("--min_sharpness", type=float, default=120)
    ap.add_argument("--sharpness_top_percent", type=float, default=None,
                    help="Keep only the top N% sharpest images after other thresholds. Example: 30 keeps top 30%. Overrides min_sharpness if set.")

    # Selection controls
    ap.add_argument("--window_size", type=int, default=0,
                    help="If >0: group consecutive frames per subfolder, keep only best N per window.")
    ap.add_argument("--keep_per_window", type=int, default=3)

    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--dedupe_phash_dist", type=int, default=6, help="<= this => near-duplicate (greedy)")

    ap.add_argument("--move", action="store_true", help="Move images instead of copy (default copy).")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, limit number of images for quick tests.")
    args = ap.parse_args()

    # Load defaults from --config if provided
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg = load_yaml_config(cfg_path)
        repo_root = _repo_root_from_this_file()
        config_dir = cfg_path.parent

        work_dir = _get(cfg, "work_dir") or "work"
        ff = _get(cfg, "filter_frames") or {}

        # Resolve paths (relative allowed)
        def _p(key: str, default: str) -> str:
            v = ff.get(key, default)
            return str(resolve_path(str(v), repo_root, config_dir))

        if not args.in_dir:
            args.in_dir = _p("in_dir", str(Path(str(work_dir)) / "00_frames_raw"))
        if not args.out_scored:
            args.out_scored = _p("out_scored", str(Path(str(work_dir)) / "10_frames_scored"))
        if not args.out_keep:
            args.out_keep = _p("out_keep", str(Path(str(work_dir)) / "11_frames_keep"))
        if not args.out_reject:
            args.out_reject = _p("out_reject", str(Path(str(work_dir)) / "12_frames_reject"))

        # Thresholds
        for k in ["min_brightness","max_brightness","max_dark_ratio","max_bright_ratio","min_contrast","min_sharpness","sharpness_top_percent"]:
            if k in ff:
                if ff.get(k) is not None:
                    setattr(args, k, float(ff[k]))
        # Selection/dedupe/IO
        for k in ["window_size","keep_per_window","dedupe_phash_dist","max_images"]:
            if k in ff:
                if ff.get(k) is not None:
                    setattr(args, k, int(ff[k]))
        for k in ["dedupe","move"]:
            if k in ff:
                if ff.get(k) is not None:
                    setattr(args, k, bool(ff[k]))
    if not args.in_dir or not args.out_scored or not args.out_keep or not args.out_reject:
        raise SystemExit("You must provide --config or --in_dir/--out_scored/--out_keep/--out_reject.")

    t0 = perf_counter()

    in_dir = Path(args.in_dir)
    out_scored = Path(args.out_scored)
    keep_dir = Path(args.out_keep) / "images"
    rej_dir = Path(args.out_reject) / "images"

    ensure_dir(out_scored)
    ensure_dir(keep_dir)
    ensure_dir(rej_dir)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    if args.max_images and args.max_images > 0:
        images = images[:args.max_images]

    if not images:
        raise SystemExit(f"No images found under {in_dir}")

    # --- Scoring ---
    score_t0 = perf_counter()
    rows = []
    for p in tqdm(images, desc="Scoring", unit="img"):
        img_t0 = perf_counter()
        try:
            s = compute_scores(p)
            row = {
                "src": str(p),
                "brightness": s.brightness,
                "contrast": s.contrast,
                "sharpness": s.sharpness,
                "dark_ratio": s.dark_ratio,
                "bright_ratio": s.bright_ratio,
                "phash": s.phash,
                "error": None,
            }
        except Exception as e:
            row = {
                "src": str(p),
                "brightness": None,
                "contrast": None,
                "sharpness": None,
                "dark_ratio": None,
                "bright_ratio": None,
                "phash": None,
                "error": str(e),
            }
        row["score_time_ms"] = (perf_counter() - img_t0) * 1000.0
        rows.append(row)

    score_dt = perf_counter() - score_t0
    avg_score_ms = 1000.0 * score_dt / max(1, len(images))

    scored_csv = out_scored / "scored.csv"
    scored_jsonl = out_scored / "scored.jsonl"

    with open(scored_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_jsonl(scored_jsonl, rows)

    print(f"[STATS] scored={len(images)} avg_score_ms={avg_score_ms:.1f} score_time_sec={score_dt:.2f}")
    print(f"[OUT]   {scored_csv}")
    print(f"[OUT]   {scored_jsonl}")

    # --- Threshold filtering ---
    thresh_t0 = perf_counter()
    good = []
    bad = []  # list[(row, reasons)]

    for r in rows:
        if r["error"]:
            bad.append((r, ["read_fail"]))
            continue

        reasons = []
        if r["brightness"] < args.min_brightness or r["dark_ratio"] > args.max_dark_ratio:
            reasons.append("too_dark")
        if r["brightness"] > args.max_brightness or r["bright_ratio"] > args.max_bright_ratio:
            reasons.append("too_bright")
        if r["contrast"] < args.min_contrast:
            reasons.append("low_contrast")
        if (args.sharpness_top_percent is None) and (r["sharpness"] < args.min_sharpness):
            reasons.append("blurry")

        if reasons:
            bad.append((r, reasons))
        else:
            good.append(r)

    thresh_dt = perf_counter() - thresh_t0
    sharp_thr = None
    if args.sharpness_top_percent is not None:
        kept, info = _apply_sharpness_percentile(good, args.sharpness_top_percent)
        if info is not None:
            sharp_thr, dropped = info
            for r in dropped:
                bad.append((r, ["sharpness_percentile_drop"]))
        good = kept
    print(f"[STATS] after_threshold keep={len(good)} reject={len(bad)} thresh_time_sec={thresh_dt:.2f}"
          + (f" sharp_top_percent={args.sharpness_top_percent:g}% sharp_thr={sharp_thr:.3f}" if args.sharpness_top_percent is not None else ""))

    # --- Window selection (best-of per chunk per folder) ---
    if args.window_size and args.window_size > 0:
        win_t0 = perf_counter()
        grouped = {}
        for r in good:
            p = Path(r["src"])
            grouped.setdefault(str(p.parent), []).append(r)

        selected = []
        dropped = []
        for _, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x["src"])
            w = args.window_size
            k = args.keep_per_window
            for i in range(0, len(items_sorted), w):
                chunk = items_sorted[i:i + w]
                # Heuristic: sharpness * contrast
                chunk_sorted = sorted(chunk, key=lambda x: (x["sharpness"] * x["contrast"]), reverse=True)
                selected.extend(chunk_sorted[:k])
                dropped.extend(chunk_sorted[k:])

        good = selected
        for r in dropped:
            bad.append((r, ["window_drop"]))

        win_dt = perf_counter() - win_t0
        print(f"[STATS] window_select keep={len(good)} (dropped {len(dropped)}) window_time_sec={win_dt:.2f}")

    # --- Dedupe (greedy by file order) ---
    if args.dedupe:
        dd_t0 = perf_counter()
        good_sorted = sorted(good, key=lambda x: x["src"])
        deduped = []
        dupes = []

        for r in good_sorted:
            if not deduped:
                deduped.append(r)
                continue
            last = deduped[-1]
            if r["phash"] and last["phash"] and phash_dist(r["phash"], last["phash"]) <= args.dedupe_phash_dist:
                dupes.append(r)
            else:
                deduped.append(r)

        good = deduped
        for r in dupes:
            bad.append((r, ["near_duplicate"]))

        dd_dt = perf_counter() - dd_t0
        print(f"[STATS] dedupe keep={len(good)} dupes={len(dupes)} dedupe_time_sec={dd_dt:.2f}")

    # --- Copy/move outputs ---
    io_t0 = perf_counter()
    keep_manifest = []
    reject_manifest = []

    # Keep set for quick lookup
    keep_src_set = {r["src"] for r in good}

    # Copy KEEP
    for r in tqdm(good, desc="Copy KEEP", unit="img"):
        src = Path(r["src"])
        rel = src.relative_to(in_dir)
        dst_name = flatten_relpath(rel)
        dst = keep_dir / dst_name
        copy_or_move(src, dst, do_move=args.move)

        keep_manifest.append({
            "path": str(dst),
            "src": r["src"],
            "keep": True,
            "brightness": r["brightness"],
            "contrast": r["contrast"],
            "sharpness": r["sharpness"],
            "dark_ratio": r["dark_ratio"],
            "bright_ratio": r["bright_ratio"],
            "phash": r["phash"],
        })

    # Copy REJECT
    for r, reasons in tqdm(bad, desc="Copy REJECT", unit="img"):
        src = Path(r["src"])
        if not src.exists():
            # If moved already and this was re-classified, skip safely
            continue
        rel = src.relative_to(in_dir)
        dst_name = flatten_relpath(rel)
        dst = rej_dir / dst_name
        # If move=true and src already moved to KEEP, skip
        if args.move and r["src"] in keep_src_set:
            continue
        copy_or_move(src, dst, do_move=args.move)

        reject_manifest.append({
            "path": str(dst),
            "src": r["src"],
            "keep": False,
            "reasons": reasons,
            "brightness": r.get("brightness"),
            "contrast": r.get("contrast"),
            "sharpness": r.get("sharpness"),
            "dark_ratio": r.get("dark_ratio"),
            "bright_ratio": r.get("bright_ratio"),
            "phash": r.get("phash"),
            "error": r.get("error"),
        })

    io_dt = perf_counter() - io_t0

    keep_manifest_path = Path(args.out_keep) / "manifest.jsonl"
    reject_manifest_path = Path(args.out_reject) / "manifest.jsonl"
    write_jsonl(keep_manifest_path, keep_manifest)
    write_jsonl(reject_manifest_path, reject_manifest)

    total_dt = perf_counter() - t0
    print(f"[DONE] keep={len(keep_manifest)} reject={len(reject_manifest)} io_time_sec={io_dt:.2f} total_time_sec={total_dt:.2f}")
    print(f"[OUT]  keep images:   {keep_dir}")
    print(f"[OUT]  keep manifest: {keep_manifest_path}")
    print(f"[OUT]  reject images: {rej_dir}")
    print(f"[OUT]  reject manifest:{reject_manifest_path}")

    perf = {
        "script": "filter_frames.py",
        "timestamp": time(),
        "total_time_sec": total_dt,
        "score_time_sec": score_dt,
        "threshold_time_sec": thresh_dt,
        "io_time_sec": io_dt,
        "avg_score_ms": avg_score_ms,
        "count_images_input": len(images),
        "count_keep": len(keep_manifest),
        "count_reject": len(reject_manifest),
        "args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    perf_path = out_scored / "perf.json"
    perf_path.write_text(json.dumps(perf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PERF] wrote {perf_path}")


if __name__ == "__main__":
    main()

