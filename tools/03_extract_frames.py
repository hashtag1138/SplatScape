#!/usr/bin/env python3
"""
extract_frames.py
Videos -> extracted frames (organized per clip folder)
Progress: real-time per video (based on ffmpeg -progress) + total time.

Example:
  python tools/extract_frames.py \
    --in_dir inputs/videos \
    --out_dir work/00_frames_raw \
    --fps 2 \
    --scale 1920:-2
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter, time
from typing import Optional

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



def run_cmd_check(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration_sec(video_path: Path) -> Optional[float]:
    """Return duration in seconds, or None if ffprobe fails."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def run_ffmpeg_extract_with_progress(
    video_path: Path,
    out_pattern: Path,
    fps: float,
    scale: Optional[str],
    ext: str,
    desc: str
) -> dict:
    """
    Run ffmpeg extraction and track progress with -progress pipe:1
    Returns stats dict.
    """
    duration = probe_duration_sec(video_path)
    # Build filter chain
    vf = f"fps={fps}"
    if scale:
        vf += f",scale={scale}"

    # jpg quality: -q:v 2 (good), png: -compression_level 3 (fast-ish)
    base_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
    ]
    if ext == "jpg":
        base_cmd += ["-q:v", "2"]
    else:
        base_cmd += ["-compression_level", "3"]

    # Add progress output to stdout
    cmd = base_cmd + ["-progress", "pipe:1", "-nostats", str(out_pattern)]

    t0 = perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    if duration and duration > 0:
        bar = tqdm(total=duration, desc=desc, unit="s", leave=True)
    else:
        # Fallback: unknown duration -> indeterminate bar
        bar = tqdm(total=0, desc=desc, unit="lines", leave=True)

    last_sec = 0.0
    last_lines = 0
    out_time_sec = 0.0

    try:
        assert p.stdout is not None
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue

            # progress keys include: out_time_ms=..., progress=continue/end
            if line.startswith("out_time_ms="):
                try:
                    ms = int(line.split("=", 1)[1])
                    out_time_sec = ms / 1_000_000.0
                except Exception:
                    continue

                if duration and duration > 0:
                    if out_time_sec > last_sec:
                        bar.update(out_time_sec - last_sec)
                        last_sec = out_time_sec
            else:
                # unknown duration fallback
                if not (duration and duration > 0):
                    last_lines += 1
                    if last_lines % 10 == 0:
                        bar.update(10)

        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with code {rc}")
    finally:
        bar.close()

    dt = perf_counter() - t0
    return {
        "video": str(video_path),
        "duration_sec": duration,
        "processed_time_sec": out_time_sec,
        "elapsed_sec": dt,
        "fps_extract": fps,
        "scale": scale,
        "ext": ext,
        "out_pattern": str(out_pattern),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path; uses extract_frames.* keys")
    ap.add_argument("--in_dir", required=False, help="Folder containing videos")
    ap.add_argument("--out_dir", required=False, help="Output root folder")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    ap.add_argument("--scale", default=None, help="Optional: e.g. 1920:-2 or 1280:-2")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--video_exts", default="mp4,mov,mkv,webm", help="Comma-separated allowed video extensions")
    args = ap.parse_args()

    # Load defaults from --config if provided
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg = load_yaml_config(cfg_path)
        repo_root = _repo_root_from_this_file()
        config_dir = cfg_path.parent

        work_dir = _get(cfg, "work_dir") or "work"
        ef = _get(cfg, "extract_frames") or {}

        # Resolve paths (relative allowed)
        if not args.in_dir:
            args.in_dir = str(resolve_path(str(ef.get("in_dir") or "inputs/videos"), repo_root, config_dir))
        if not args.out_dir:
            args.out_dir = str(resolve_path(str(ef.get("out_dir") or (Path(str(work_dir)) / "00_frames_raw")), repo_root, config_dir))

        if args.fps is None or ("fps" in ef):
            args.fps = float(ef.get("fps", args.fps if args.fps is not None else 2.0))
        if ef.get("scale", None) is not None:
            args.scale = ef.get("scale", None)
        if ef.get("ext", None):
            args.ext = str(ef.get("ext"))
        if ef.get("video_exts", None):
            args.video_exts = str(ef.get("video_exts"))

    if not args.in_dir or not args.out_dir:
        raise SystemExit("You must provide --config or both --in_dir and --out_dir.")

    t0 = perf_counter()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {("." + e.strip().lower()) for e in args.video_exts.split(",") if e.strip()}
    videos = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not videos:
        raise SystemExit(f"No videos found in {in_dir} with extensions {sorted(exts)}")

    stats = []
    for v in videos:
        clip_out = out_dir / v.stem
        clip_out.mkdir(parents=True, exist_ok=True)
        out_pat = clip_out / f"{v.stem}_%06d.{args.ext}"

        s = run_ffmpeg_extract_with_progress(
            video_path=v,
            out_pattern=out_pat,
            fps=args.fps,
            scale=args.scale,
            ext=args.ext,
            desc=f"Extract {v.name}"
        )
        stats.append(s)

    total_dt = perf_counter() - t0
    print(f"[DONE] videos={len(videos)} total_time_sec={total_dt:.2f}")

    perf = {
        "script": "extract_frames.py",
        "timestamp": time(),
        "total_time_sec": total_dt,
        "count_videos": len(videos),
        "args": vars(args),
        "per_video": stats,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    perf_path = out_dir / "perf.json"
    perf_path.write_text(json.dumps(perf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PERF] wrote {perf_path}")


if __name__ == "__main__":
    main()

