# LiDAR Surrogate-Safety Pilot

Pilot pipeline: 3D detection on LiDAR point clouds → multi-object tracking →
kinematics → surrogate safety metrics (TTC, PET, hard-braking events).

```
point clouds ──► CenterPoint (MMDetection3D, pretrained nuScenes)
                    │  3D boxes + velocity head
                    ▼
              Kalman tracker (AB3DMOT-style, this repo)
                    │  per-object trajectories
                    ▼
              kinematics: speed, heading, acceleration
                    │
                    ▼
              surrogate safety metrics:
                TTC  (predictive, closing conflicts)
                PET  (measured, path-crossing conflicts)
                HBE  (hard braking < -3 m/s², leading indicator)
```

## Layout

- `src/lidar_pilot/tracking/` — constant-velocity Kalman filter per box +
  Hungarian association tracker (implemented from scratch, no filterpy)
- `src/lidar_pilot/kinematics.py` — speed / heading / acceleration from
  trajectories
- `src/lidar_pilot/metrics/` — TTC, PET, hard-braking event extraction;
  each module's docstring states the exact formulation used
- `tests/` — closed-form synthetic scenarios (head-on TTC, orthogonal PET
  crossings, braking profiles, ID-stability under occlusion)
- `scripts/` — the nuScenes pipeline runner (GPU), conflict mining, and
  figure generation

## Setup (local, metrics development)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Detection runs on a CUDA box. On Blackwell GPUs (RTX 50-series, sm_120)
prebuilt wheels do not exist for the detection stack; the working recipe is:
torch 2.7.1 from the cu128 index, mmcv 2.1.0 built from source with
`MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0"`, then
mmdet 3.3.0 and mmdetection3d v1.4.0 installed with
`--no-build-isolation`. Pretrained CenterPoint-pillar nuScenes weights
come from the MMDetection3D model zoo (checkpoint URLs are in
`configs/centerpoint/metafile.yml`).

## Data

- **nuScenes-mini** (ego-vehicle, bring-up + validation against GT
  velocities) — nuscenes.org, non-commercial ToU
- **LUMPI** (roadside multi-LiDAR intersection, the target study) —
  data.uni-hannover.de/dataset/lumpi, CC BY-NC 3.0
