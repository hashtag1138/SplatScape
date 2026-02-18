#!/usr/bin/env python3
"""
3dgs_output_preview.py

What it does:
- Reads your pipeline YAML via --config
- Deduces the 3DGS output directory from the config (three_dgs.output_dir or three_dgs.work_dir etc.)
- Finds the latest point_cloud.ply under:
    <output_dir>/point_cloud/iteration_*/point_cloud.ply
- Ensures antimatter15/splat exists under .third_party/ (clones if missing)
- Optionally converts PLY -> SPLAT using splat/convert.py
- Serves the viewer with a local HTTP server (so the browser can fetch the file)

Notes:
- The splat viewer can load a 3DGS PLY via drag&drop.
- Auto-load via ?url= expects a .splat file, so conversion is the easiest way to auto-open.

Usage examples:
  python3 tools/3dgs_output_preview.py --config test2.yaml --open
  python3 tools/3dgs_output_preview.py --config test2.yaml --no-convert --open   # drag&drop mode
"""

from __future__ import annotations

import argparse
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Optional


SPLAT_REPO_URL = "https://github.com/antimatter15/splat"

def ensure_pip_package(module_name: str, pip_name: str | None = None) -> None:
    """Best-effort install of a missing Python dependency into the current interpreter."""
    pip_name = pip_name or module_name
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    print(f"[deps] Missing '{module_name}'. Installing '{pip_name}' into {sys.executable} ...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", pip_name])
    # Re-check
    __import__(module_name)



def _get(d: Any, path: str) -> Any:
    """Get nested dict value using dot path; returns None if missing."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for --config parsing. Install with:\n"
            "  pip install pyyaml\n"
            "or add it to your venv requirements."
        ) from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a dict.")
    return data


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_project_root(start: Path) -> Path:
    """Try to find a folder containing .third_party by walking up a few levels."""
    cur = start.resolve()
    for _ in range(8):
        if (cur / ".third_party").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def ensure_repo(third_party_dir: Path) -> Path:
    """Returns <third_party_dir>/splat, cloning if missing."""
    third_party_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = third_party_dir / "splat"

    if repo_dir.exists() and (repo_dir / "index.html").exists():
        return repo_dir

    if repo_dir.exists():
        raise RuntimeError(
            f"Expected repo at {repo_dir}, but it already exists and doesn't look like antimatter15/splat."
        )

    run(["git", "clone", "--depth", "1", SPLAT_REPO_URL, str(repo_dir)])
    return repo_dir


def resolve_path(maybe_path: Any, *, base_dir: Path) -> Optional[Path]:
    if not maybe_path:
        return None
    p = Path(str(maybe_path)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def deduce_3dgs_output_dir(cfg: dict, config_path: Path) -> Path:
    """
    Priority:
    1) three_dgs.output_dir
    2) three_dgs.work_dir + /3dgs/output
    3) paths.work_dir + /3dgs/output
    4) work_dir (root) + /3dgs/output
    Else: error
    """
    base_dir = config_path.parent.resolve()

    out_dir = resolve_path(_get(cfg, "three_dgs.output_dir"), base_dir=base_dir)
    if out_dir:
        return out_dir

    work_dir = (
        resolve_path(_get(cfg, "three_dgs.work_dir"), base_dir=base_dir)
        or resolve_path(_get(cfg, "paths.work_dir"), base_dir=base_dir)
        or resolve_path(_get(cfg, "work_dir"), base_dir=base_dir)
    )
    if work_dir:
        return (work_dir / "3dgs" / "output").resolve()

    raise RuntimeError(
        "Couldn't deduce 3DGS output dir from config. Add one of:\n"
        "  three_dgs.output_dir: /abs/path/to/3dgs/output\n"
        "or\n"
        "  three_dgs.work_dir: /abs/path/to/work_dir\n"
        "(then output becomes <work_dir>/3dgs/output)"
    )


def find_latest_point_cloud_ply(output_dir: Path) -> Path:
    """Find latest iteration_*/point_cloud.ply inside output_dir/point_cloud."""
    pc_root = output_dir / "point_cloud"
    if not pc_root.exists():
        raise FileNotFoundError(f"Not found: {pc_root} (is your output_dir correct?)")

    candidates: list[tuple[int, Path]] = []
    for it_dir in pc_root.glob("iteration_*"):
        if not it_dir.is_dir():
            continue
        # iteration_30000 -> 30000
        try:
            it_num = int(it_dir.name.split("_")[-1])
        except Exception:
            continue
        ply = it_dir / "point_cloud.ply"
        if ply.exists():
            candidates.append((it_num, ply))

    if not candidates:
        raise FileNotFoundError(f"No point_cloud.ply found under {pc_root}/iteration_*/point_cloud.ply")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def convert_ply_to_splat(repo_dir: Path, ply_path: Path, out_splat: Path) -> Path:
    """Convert a 3DGS .ply to .splat using splat/convert.py (1-arg mode).

    Your version of antimatter15/splat/convert.py writes its output in 1-arg mode
    (e.g. 'output.splat' in repo_dir). We therefore skip the 2-arg attempt and
    directly run the working mode.

    Returns the produced .splat path.
    """
    convert_py = repo_dir / "convert.py"
    if not convert_py.exists():
        raise RuntimeError(f"convert.py not found in {repo_dir}")

    out_splat.parent.mkdir(parents=True, exist_ok=True)

    # convert.py requires plyfile
    ensure_pip_package("plyfile", "plyfile")

    # Run the working CLI mode: python convert.py input.ply
    run([sys.executable, str(convert_py), str(ply_path)], cwd=repo_dir)

    # Common outputs:
    #  - repo_dir/output.splat
    #  - input.ply.splat (next to input)
    #  - input.splat (same stem)
    candidates = [
        repo_dir / "output.splat",
        Path(str(ply_path) + ".splat"),
        ply_path.with_suffix(".splat"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c

    raise RuntimeError(f"Conversion produced no output. Looked for: {', '.join(map(str, candidates))}")


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve_and_open(repo_dir: Path, url_value: Optional[str], port: int, open_browser: bool) -> None:
    """Serve repo_dir over HTTP and optionally open viewer with ?url="""
    os.chdir(repo_dir)

    server = ThreadingHTTPServer(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler)
    actual_port = server.server_address[1]

    if url_value:
        viewer_url = f"http://127.0.0.1:{actual_port}/?url={url_value}"
    else:
        viewer_url = f"http://127.0.0.1:{actual_port}/"

    print(f"[server] Serving {repo_dir} on http://127.0.0.1:{actual_port}/")
    print(f"[viewer]  {viewer_url}")
    if not url_value:
        print("[hint] Drag&drop your point_cloud.ply into the page to load it.")
    print("[hint] Press Ctrl+C to stop the server.")

    if open_browser:
        webbrowser.open(viewer_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Stopping...")
    finally:
        server.server_close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to your pipeline YAML config.")
    ap.add_argument(
        "--ply",
        type=str,
        default="",
        help="Optional override path to point_cloud.ply. If omitted, script finds latest from config output_dir.",
    )
    ap.add_argument("--port", type=int, default=8080, help="Port for the local server (default: 8080).")

    ap.add_argument(
        "--third-party",
        type=str,
        default="",
        help="Optional path to the .third_party directory. If omitted, script searches upward from cwd.",
    )
    ap.add_argument(
        "--no-convert",
        action="store_true",
        help="Do not convert PLY->SPLAT. (You can still drag&drop the PLY in the browser.)",
    )
    ap.add_argument("--open", action="store_true", help="Open the browser automatically.")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .splat path (default: <repo_dir>/local_model.splat).",
    )
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = load_yaml(config_path)

    # Determine output_dir and choose ply
    if args.ply:
        ply_path = Path(args.ply).expanduser().resolve()
    else:
        output_dir = deduce_3dgs_output_dir(cfg, config_path)
        ply_path = find_latest_point_cloud_ply(output_dir)

    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    print(f"[ply] {ply_path}")

    project_root = find_project_root(Path.cwd())
    third_party_dir = Path(args.third_party).expanduser().resolve() if args.third_party else (project_root / ".third_party")
    repo_dir = ensure_repo(third_party_dir)

    if args.no_convert:
        # Stock viewer cannot auto-load a .ply via ?url=, but supports drag&drop of 3DGS PLY files.
        serve_and_open(repo_dir, url_value=None, port=args.port, open_browser=args.open)
        return 0

    out_splat = Path(args.out).expanduser().resolve() if args.out else (repo_dir / "local_model.splat")
    produced = convert_ply_to_splat(repo_dir, ply_path, out_splat)

    # Ensure it's inside repo dir so SimpleHTTPRequestHandler can serve it
    served_path = produced
    if served_path.parent != repo_dir:
        # copy under repo so the HTTP server can access it
        dst = repo_dir / (Path(args.out).name if args.out else "local_model.splat")
        shutil.copy2(served_path, dst)
        served_path = dst

    serve_and_open(repo_dir, url_value=f"http://127.0.0.1:{args.port}/{served_path.name}", port=args.port, open_browser=args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
