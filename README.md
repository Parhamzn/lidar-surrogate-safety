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
- `PILOT_PLAN.md` — full project plan, dataset survey, and the verified
  RTX 5090 (Blackwell) toolchain recipe

## Setup (local, metrics development)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Detection runs remotely (RTX 5090) on the env described in
`PILOT_PLAN.md` §6: torch 2.7.1+cu128, source-built mmcv 2.1.0,
mmdetection3d v1.4.0.

## Data

- **nuScenes-mini** (ego-vehicle, bring-up + validation against GT
  velocities) — nuscenes.org, non-commercial ToU
- **LUMPI** (roadside multi-LiDAR intersection, the target study) —
  data.uni-hannover.de/dataset/lumpi, CC BY-NC 3.0
