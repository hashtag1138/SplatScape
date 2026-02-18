#!/usr/bin/env python3
"""
preview_gallery.py
Generate an HTML gallery + a contact sheet from a manifest.jsonl.

Progress: thumbnail generation bar + total time + perf.json.

Example:
  python tools/preview_gallery.py \
    --manifest work/11_frames_keep/manifest.jsonl \
    --out_dir work/19_previews/keep \
    --title "KEEP (for COLMAP)"
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter, time

from PIL import Image
from tqdm import tqdm


def _get(d, path: str):
    cur = d
    for part in path.split('.'):
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
    obj = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError(f"Config did not parse to a dict: {path}")
    return obj


def resolve_path(p: str | None, repo_root: Path, config_dir: Path) -> Path:
    if not p:
        return Path()
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    cand = (repo_root / pp).resolve()
    # Keep same heuristic as other tools
    return cand if cand.exists() or str(pp).startswith('work') or str(pp).startswith('inputs') else (config_dir / pp).resolve()


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:sans-serif;margin:16px;background:#111;color:#eee}}
h1{{font-size:20px}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{width:{w}px}}
img{{width:{w}px;height:auto;border-radius:10px;border:1px solid #333}}
.small{{color:#aaa;font-size:12px;word-break:break-all;line-height:1.35}}
a{{color:#8cf}}
</style></head><body>
<h1>{title}</h1>
<p class="small">Items shown: {count} (max_items cap may apply)</p>
<div class="grid">
{cards}
</div></body></html>
"""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config path; uses filter_frames.* + work_dir")
    ap.add_argument("--kind", choices=["keep","reject"], default="keep", help="Which manifest to preview (default: keep)")
    ap.add_argument("--manifest", required=False, help="manifest.jsonl (overrides --config)")
    ap.add_argument("--out_dir", required=False, help="Output directory for gallery (overrides --config)")
    ap.add_argument("--title", default="Preview")
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--thumb_w", type=int, default=240)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    # Load defaults from --config if provided
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg = load_yaml_config(cfg_path)
        repo_root = _repo_root_from_this_file()
        config_dir = cfg_path.parent

        work_dir = _get(cfg, 'work_dir') or 'work'
        ff = _get(cfg, 'filter_frames') or {}

        # Select manifest based on kind, unless --manifest overrides
        if not args.manifest:
            if args.kind == 'keep':
                m = ff.get('out_keep') or str(Path(str(work_dir)) / '11_frames_keep')
            else:
                m = ff.get('out_reject') or str(Path(str(work_dir)) / '12_frames_reject')
            mdir = resolve_path(str(m), repo_root, config_dir)
            args.manifest = str(mdir / 'manifest.jsonl')

        # Default output dir: <work_dir>/19_previews/<kind>
        if not args.out_dir:
            out_base = Path(str(work_dir)) / '19_previews' / args.kind
            args.out_dir = str(resolve_path(str(out_base), repo_root, config_dir))

        # Default title if not explicitly set
        if args.title == 'Preview':
            args.title = f"{args.kind.upper()} preview"

    if not args.manifest or not args.out_dir:
        raise SystemExit('You must provide --config or both --manifest and --out_dir.')

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")


    t0 = perf_counter()

    out = Path(args.out_dir)
    ensure_dir(out)

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_items:
                break
            items.append(json.loads(line))

    thumbs = []
    cards = []

    for i, it in enumerate(tqdm(items, desc="Thumbnails", unit="img")):
        p = Path(it.get("path", ""))
        if not p.exists():
            continue

        img = Image.open(p).convert("RGB")
        w = args.thumb_w
        h = int(img.height * (w / img.width))
        img_t = img.resize((w, h))

        thumb_path = out / f"thumb_{i:04d}.jpg"
        img_t.save(thumb_path, quality=args.quality)

        thumbs.append(img_t)

        meta = []
        for k in ("sharpness", "contrast", "brightness"):
            v = it.get(k, None)
            if isinstance(v, (int, float)):
                meta.append(f"{k}:{v:.1f}")

        cards.append(
            f'<div class="card">'
            f'<a href="{p.as_posix()}"><img src="{thumb_path.name}"></a>'
            f'<div class="small">{p.name}<br>{" | ".join(meta)}</div>'
            f'</div>'
        )

    html = HTML.format(title=args.title, cards="\n".join(cards), w=args.thumb_w, count=len(thumbs))
    html_path = out / "gallery.html"
    html_path.write_text(html, encoding="utf-8")

    # contact sheet
    cols = args.cols
    rows = args.rows
    n = min(len(thumbs), cols * rows)
    sheet_path = out / "contactsheet.jpg"
    if n:
        tw, th = thumbs[0].size
        sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
        for idx in range(n):
            x = (idx % cols) * tw
            y = (idx // cols) * th
            sheet.paste(thumbs[idx], (x, y))
        sheet.save(sheet_path, quality=args.quality)

    total_dt = perf_counter() - t0
    print(f"[DONE] thumbs={len(thumbs)} total_time_sec={total_dt:.2f}")
    print(f"[OUT]  {html_path}")
    if sheet_path.exists():
        print(f"[OUT]  {sheet_path}")

    perf = {
        "script": "preview_gallery.py",
        "timestamp": time(),
        "total_time_sec": total_dt,
        "count_items": len(items),
        "count_thumbs": len(thumbs),
        "args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    perf_path = out / "perf.json"
    perf_path.write_text(json.dumps(perf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PERF] wrote {perf_path}")


if __name__ == "__main__":
    main()

