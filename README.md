# SplatScape — From video to 3D Gaussian Splats
COLMAP → 3D Gaussian Splatting pipeline (with YAML config)

This repo is a **pragmatic pipeline** to go from **video → frames → filtered frames → COLMAP reconstruction → 3D Gaussian Splatting (3DGS)**, with small helper scripts to keep paths consistent via a single `*.yaml` config.

The goal is to make the workflow repeatable:
- one **work directory** (`work_dir`) per dataset/run
- one **config file** (`--config your_run.yaml`)
- scripts that agree on where inputs/outputs live

---

## What you get

### Main steps
1. **Install dependencies**
   - COLMAP (optionally CUDA-enabled)
   - 3DGS (gaussian-splatting repo)
2. **Configure a run (wizard)**
   - generates a YAML config (`project.yaml`, etc.)
3. **Extract frames from video**
4. **Filter frames**
   - quality scoring, dedupe, keep/reject sets
   - optional HTML gallery preview
5. **Run COLMAP**
   - feature extraction + matching + mapping + undistortion
6. **Prepare 3DGS scene**
7. **Train 3DGS**
8. **Preview result**
   - quick preview via `antimatter15/splat` web viewer

---

## Repository layout (typical)

You can choose any `work_dir` (default: `work`). A common layout looks like:

```
<work_dir>/
  inputs/videos/              # your source videos (optional convention)
  00_frames_raw/              # extracted frames
  10_frames_scored/           # scoring reports
  11_frames_keep/             # filtered frames kept
  12_frames_reject/           # rejected frames
  19_previews/keep/           # HTML gallery previews
  colmap_out/                 # COLMAP outputs (db, sparse, undistorted, logs)
  3dgs_scene/                 # 3DGS prepared scene
  3dgs/output/                # 3DGS training outputs (point_cloud/iteration_*/point_cloud.ply)
```

Your exact structure can be different; the **YAML config is the source of truth**.

---

## Installation

### 1) Install Python deps (venv + requirements.txt)

`00_colmap_install.py` creates (or reuses) the project's venv and installs the Python dependencies from `requirements.txt` (it also upgrades `pip`).

That includes the libraries used by:
- frame extraction / filtering (`opencv-python`, `numpy`, `Pillow`, `tqdm`, …)
- YAML config handling (`PyYAML`)
- 3DGS preview conversion (`plyfile`) — **included in `requirements.txt`**

So in the common case, you don't need to `pip install ...` manually: just run the installer script.

### 2) Install COLMAP
Run:
```bash
python3 tools/00_colmap_install.py
```

If you already have a COLMAP binary installed, you can point to it via your YAML config.

### 3) Install 3DGS (gaussian-splatting)
Run:
```bash
python3 tools/01_3DGS_install.py --auto --allow-sudo
```

This usually clones/sets up the gaussian-splatting repo under `.third_party/`.

---

## Quickstart (end-to-end)

If you installed via `00_colmap_install.py`, your venv lives at `.venv/`. You can either activate it or run scripts with `.venv/bin/python3`.

### 1) Generate a config
Wizard (GPU example):
```bash
python3 tools/02_colmap_3dgs_config_wizard.py --gpu --out project.yaml
```

It will ask for a `work_dir` early. That value is then used as the default base for the rest of the paths.

### 2) Extract frames
```bash
python3 tools/03_extract_frames.py --config project.yaml
```

### 3) Filter frames
```bash
python3 tools/04_filter_frames.py --config project.yaml
```

### 4) (Optional) Preview the keep/reject gallery
```bash
python3 tools/05_preview_gallery.py --config project.yaml --kind keep
python3 tools/05_preview_gallery.py --config project.yaml --kind reject
```

### 5) Run COLMAP
```bash
python3 tools/08_colmap_run.py --config project.yaml
```

### 6) Preview COLMAP sparse model
```bash
python3 tools/09_colmap_preview.py --config project.yaml
```

### 7) Prepare scene for 3DGS
```bash
python3 tools/10_3dgs_prepare_scene.py --config project.yaml
```

### 8) Train 3DGS
```bash
python3 tools/11_3dgs_train.py --config project.yaml
```

### 9) Preview 3DGS output
```bash
python3 tools/12_3dgs_preview.py --config project.yaml --open
```

This downloads `antimatter15/splat` under `.third_party/` if missing, starts a local server, and opens the viewer.

---

## Clearing outputs (reset a run)

To delete COLMAP outputs for the current config:
```bash
python3 tools/06_colmap_clear_fixed.py --config project.yaml --yes
```

Notes:
- this removes `database.db`, `sparse/`, `undistorted/`, `logs/` (based on your YAML paths)
- it keeps the parent folder (so your directory structure remains)

---

## YAML configuration

The wizard generates a YAML file containing (at minimum):

- `work_dir`
- `extract_frames.*`
- `filter_frames.*`
- `colmap.paths.*`
- `three_dgs.work_dir`, `three_dgs.output_dir`, `three_dgs.scene_dir`

All scripts should accept:
```bash
--config your_run.yaml
```

…and derive their input/output folders from this file.

---

## Configuration reference (project.yaml)

The config file is the **contract** between all scripts. If something is slow, low-quality, or OOMs, the first place to look is `project.yaml`.

Below is a practical guide to the key parameters and the trade-offs they control: **quality ↔ time ↔ memory (VRAM/RAM) ↔ OOM risk**.

> Tip: treat configuration like “budgeting”.  
> You spend budget on (A) number of images, (B) image resolution, (C) feature count, (D) densification.  
> Running out of budget = OOM or unstable reconstructions.

---

### `work_dir`
**What:** Base folder for the whole run (inputs/frames/COLMAP/3DGS outputs).

**Impact:**
- **Quality:** none (purely paths)
- **Time:** none
- **Risk:** low (but wrong paths = “file not found” errors)

---

### `extract_frames.*` (video → images)
**Goal:** choose how many frames you feed into the pipeline.

| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `in_dir` | Where your videos live | none | none | none |
| `out_dir` | Where extracted frames go | none | none | disk usage |
| `fps` | Extract rate (frames per second) | higher fps = more viewpoints (good)… until it becomes redundant | **↑ a lot** (all later steps scale with image count) | **↑** (COLMAP DB + matching) |
| `scale` | Optional resize at extraction (e.g. `1920:-2`) | lower scale = less detail, but often still enough for stable geometry | **↓** (faster everywhere) | **↓** (GPU features + 3DGS) |
| `ext` | `jpg`/`png` | PNG keeps more detail but huge; JPG is usually fine | PNG slower I/O | PNG uses more disk/RAM |
| `video_exts` | Video extensions whitelist | none | none | none |

**Rules of thumb**
- If you get a slow pipeline or huge matching time: reduce `fps` first.
- If you get COLMAP GPU OOM: reduce `scale` or later `SiftExtraction.max_image_size`.

---

### `filter_frames.*` (score + dedupe + keep/reject)
**Goal:** keep frames that are sharp, well exposed, and not too redundant.

| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `min_brightness` / `max_brightness` | Filters too dark/bright images | helps stability; too strict can remove useful views | slight | reduces dataset size ⇒ **↓** |
| `max_dark_ratio` / `max_bright_ratio` | Reject frames with too much dark/white area | removes “blown highlights” / “underexposed” frames | slight | **↓** |
| `min_contrast` | Reject flat frames | improves feature matching | slight | **↓** |
| `min_sharpness` | Reject blurry frames | often **huge** quality win (COLMAP + 3DGS) | slight | **↓** |
| `window_size` | Sliding window selection (0 disables) | preserves temporal coverage | **↓** (fewer images) | **↓** |
| `keep_per_window` | Keep N images per window | too low can hurt coverage; too high increases redundancy | tunes speed/quality | tunes OOM risk |
| `dedupe` | Enable perceptual hash dedupe | removes near duplicates | **↓** (faster) | **↓** |
| `dedupe_phash_dist` | Lower = stricter dedupe | too strict can remove legitimate small changes | **↓** | **↓** |
| `max_images` | Hard cap (0 = no limit) | too low can reduce coverage | **↓↓** | **↓↓** |

**Practical presets**
- **Fast test:** `max_images: 64`, `window_size: 10`, `keep_per_window: 2`
- **Balanced:** `max_images: 128`, `keep_per_window: 3`
- **Quality:** `max_images: 0` (no cap) + careful dedupe

---

### `colmap.*` (reconstruction)
COLMAP quality is mostly controlled by:
1) number of images  
2) image resolution  
3) number of extracted features  
4) matching strategy parameters  
5) mapper thresholds

#### `colmap.paths.*`
**What:** where COLMAP reads/writes.
- `images_dir` should point to your **kept frames** folder (often `.../11_frames_keep`).
- `colmap_dir`, `database_path`, `sparse_dir`, `undistorted_dir`, `logs_dir` are outputs.

**Impact:** purely filesystem; wrong paths = missing folders, “no images”, etc.

#### `colmap.feature_extractor.*`
| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `FeatureExtraction.use_gpu` | GPU SIFT extraction | same quality; GPU is faster | **↓** | **↑ VRAM**, OOM possible |
| `FeatureExtraction.gpu_index` | Which GPU | none | none | none |
| `SiftExtraction.max_image_size` | Max image edge used for SIFT | higher can improve details | **↑** | **↑ VRAM** (OOM risk) |
| `SiftExtraction.max_num_features` | Cap features per image | higher can help difficult scenes | **↑** | **↑ VRAM/RAM/DB size** |
| `ImageReader.single_camera` | Force one intrinsics model | good for fixed focal/phone; can help stability | slight | none |
| `ImageReader.camera_model` | e.g. `PINHOLE` | correct model matters | none | none |

**OOM checklist (feature extractor)**
- If you see CUDA “out of memory” during feature extraction:
  1) set `SiftExtraction.max_image_size` lower (e.g. 1600 → 1200 → 1024)
  2) reduce `SiftExtraction.max_num_features` (8192 → 4096)
  3) set `FeatureExtraction.use_gpu: 0` (CPU fallback)

#### `colmap.matcher.*`
| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `type: sequential` | Matches frames near each other in time | good for video | **↓** vs exhaustive | **↓** |
| `SequentialMatching.overlap` | How many neighbors to match | more overlap = more constraints | **↑** | **↑** (more matches) |
| `SiftMatching.max_ratio` / `max_distance` | Match strictness | too strict loses matches; too loose adds outliers | depends | more matches ⇒ more RAM |
| `SiftMatching.cross_check` | Symmetric match check | improves robustness | slight ↑ | slight |

**Rule of thumb:** if COLMAP is unstable, increase overlap a bit; if it is too slow, reduce overlap.

#### `colmap.mapper.*`
| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `Mapper.min_num_matches` | Minimum matches for registering | lower = easier to register but risk bad poses | affects stability | none |
| `Mapper.ba_refine_focal_length` | Refine intrinsics | can improve accuracy | slight ↑ | none |
| `Mapper.ba_refine_principal_point` | Often keep 0 for stability | sometimes needed | slight ↑ | none |
| `Mapper.ba_refine_extra_params` | Lens model params | can help with distortion | slight ↑ | none |

---

### `three_dgs.*` (Gaussian Splatting)
#### Paths & scene wiring
| Key | What it does | Notes |
|---|---|---|
| `repo_dir` | gaussian-splatting checkout | under `.third_party/` |
| `venv_dir` / `python` | training env | depends on your install script |
| `scene_dir` | prepared 3DGS scene | produced by `3dgs_prepare_scene` |
| `output_dir` | training outputs | expect `point_cloud/iteration_*/point_cloud.ply` |
| `source_images` | which COLMAP images feed 3DGS | usually `undistorted` |
| `convert_model_to_bin` | converts COLMAP model for 3DGS | convenience |

#### Training controls: `three_dgs.train.*`
| Key | What it does | Quality impact | Time impact | Memory/OOM impact |
|---|---|---|---|---|
| `preset` | hardware preset (e.g. `1060`) | selects safe defaults | tunes speed | tunes OOM risk |
| `iterations` | training length | more = better convergence (to a point) | **↑** | slight |
| `resolution` | downscale factor (common in 3DGS forks) | lower factor (1) = sharper | **↑** | **↑ VRAM** |
| `num_workers` | data loader workers | small effect | can help throughput | **↑ RAM** |
| `eval` | run eval/metrics | none for final render | slight ↑ | slight |
| `extra_args` | advanced flags (densify schedule etc.) | can improve detail / fill holes | varies | densify can **↑ VRAM** |

**Densification knobs (common culprits for OOM)**
- `--percent_dense`: higher values densify more aggressively → better detail but **much higher VRAM**
- `--densify_until_iter` and `--densification_interval`: more frequent densification increases VRAM spikes

**Safe tuning path (avoid OOM)**
1) Increase `resolution` (i.e., more downscale) before touching anything else (e.g. `2` → `4`)
2) Reduce densification aggressiveness:
   - lower `--percent_dense`
   - increase `--densification_interval`
   - reduce densify range (`densify_until_iter`)
3) If still OOM, reduce input image resolution (COLMAP undistorted resolution or extraction `scale`)

---

## Performance & stability cheat sheet

**Want faster runs?**
- Lower `extract_frames.fps`
- Enable/strengthen filtering: `window_size`, `max_images`, `dedupe`
- Reduce COLMAP matching overlap

**Want better quality?**
- Ensure `filter_frames.min_sharpness` is not too low (blur kills everything)
- Don’t over-filter: keep enough viewpoints (raise `max_images` or `keep_per_window`)
- Increase 3DGS `iterations` after you have a stable COLMAP model

**Getting OOM (GPU) most often comes from:**
- COLMAP GPU SIFT at high resolution (`SiftExtraction.max_image_size`)
- Too many images (matching + DB)
- 3DGS densification parameters (`--percent_dense`, frequency)

---


## Troubleshooting

### COLMAP GPU “out of memory”
If COLMAP fails during SIFT GPU extraction with `out of memory`, you have 3 practical options:

1) **Switch SIFT to CPU** (slow but robust)
```yaml
colmap:
  feature_extractor:
    SiftExtraction.use_gpu: 0
```

2) **Reduce max image size** (often enough)
```yaml
colmap:
  feature_extractor:
    SiftExtraction.max_image_size: 1600   # try 1200 or 1024 if needed
```

3) **Use fewer images**
Filter harder (or set a cap in `filter_frames.max_images`).

### “Images directory does not exist” in preview
This means `images_dir` in the YAML points to a folder that doesn’t exist.
Fix the path in `colmap.paths.images_dir` (or regenerate config with the wizard).

### 3DGS preview conversion issues
The splat converter has multiple CLI variants. This repo uses the working mode for the upstream script:
- `python convert.py input.ply`
and detects `output.splat`.

`plyfile` is installed via `requirements.txt` (through `00_colmap_install.py`). If preview still fails, you can run with `--no-convert` (drag&drop the PLY in the viewer) if your viewer version supports it.

---

## Notes / roadmap ideas

- Add presets for different GPUs (VRAM limits)
- Better automatic fallbacks (COLMAP GPU → CPU retry on OOM)
- One “run_pipeline.py” to chain all steps

---

## License
Choose a license for your project (MIT/Apache-2.0/etc.) and add it here.
