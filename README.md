# TinyAutoJEPA: Self-Supervised World Model Pilot

TinyAutoJEPA is an autonomous driving system based on a Joint Embedding Predictive Architecture (JEPA).  
It learns a **world model** from observations and predicts future states under actions, instead of directly imitating actions.

This repository is organized around a `python/` workspace with training, data tooling, and runtime scripts.


## Repository Layout

```
python/
  drive_mpc.py               # Main MPC driving loop
  models/                    # Model checkpoints (.pth)
  train/                     # Training scripts + networks.py
  debug/                     # Debug/visualization utilities
  data/                      # Data collection + conversion utilities
media/                       # Videos, gifs, diagrams, visuals
requirements.txt
```


## Quick Start (Run MPC)

From repo root:

```
python python/drive_mpc.py
```

`drive_mpc.py` writes a video to `media/run_mpc.mp4` (if `media/` exists) and opens a visualization window.


## Data Collection

Data tools live in `python/data/`.

```
# Collect data (random/expert/recover via arguments)
python python/data/collect_data.py

# Convert npz -> npy for fast seek
python python/data/convert_data.py

# Check actions or dataset sanity
python python/data/check_actions.py
python python/data/count_frames.py
```

Notes:
- `collect_data.py` outputs to `python/data_<mode>` by default.
- Collection uses stable-baselines3 if expert policies are enabled.


## Training Pipeline

All training scripts live under `python/train/` and share `python/train/networks.py`.

1) **Encoder (VICReg)**
```
python python/train/train_encoder.py
```

2) **Decoder (visual verification)**
```
python python/train/train_decoder.py
```

3) **Predictor (multi-step latent dynamics)**
```
python python/train/train_predictor.py
```

4) **Latent Heads (road / xoff / speed)**
```
python python/train/train_latent_heads.py
```

5) **Cost Model (optional)**
```
python python/train/train_cost_model.py
```

Outputs are saved under `python/models/` and visuals under `media/`.


## Debug & Visualization

Debug tools live in `python/debug/`.

```
python python/debug/visualise_dream.py
python python/debug/debug_live_vision.py
python python/debug/verify_decoder.py
python python/debug/verify_predictor.py
python python/debug/verify_data.py
python python/debug/manual_drive.py
python python/debug/watch_raceline.py
```


## Model Components (High Level)

**TinyEncoder**  
Transforms 64x64 RGB frame stacks into a spatial latent map (B, 512, 8, 8).

**Predictor**  
Autoregressive latent dynamics model. Predicts future latent states over a horizon given actions and speed.

**TinyDecoder**  
Reconstructs latent states to images for human inspection.

**Latent Heads**  
Predict road confidence, speed, and x-offset directly from the latent.


## Environment

Primary environment: `gymnasium` **CarRacing-v3**.

Dependencies are listed in `requirements.txt`. Some tools require optional packages
(e.g., `stable_baselines3` for expert data collection).


## Notes

- Most scripts accept environment variables for configuration.  
- Paths are resolved relative to `python/` and `media/` automatically.
- If you move folders, update the `_ROOT` helpers in each script accordingly.

