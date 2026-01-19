#!/usr/bin/env python3
"""
collect_data.py — CarRacing-v3 data collection for world models (JEPA-style)

Outputs (one .npz per episode):
  - obs:    uint8  (T, H, W, 3)      RGB frames (default 64x64)
  - action: float32(T-1, 3)          [steer, gas, brake] applied to reach obs[t+1]
  - reward: float32(T-1,)
  - done:   uint8  (T-1,)            1 if terminal/truncated at that transition
  - info:   bytes                    JSON metadata blob

Modes:
  - random:  correlated random policy + optional corner-bias turn-bursts
  - expert:  PPO policy (SB3) driving
  - recover: PPO with occasional sabotage bursts (use sparingly; guarded)

Safety/quality gates:
  - Off-road guard: terminate the episode if "road_score" stays low too long
  - Stuck guard: terminate if speed stays below a threshold too long
These are designed to prevent long, low-value grass spin-out tails from polluting the dataset.

Notes:
  - Safe to re-run into the same out_dir: filenames include timestamp (no overwrite).
  - Compression is slower: use --no_compress to speed up capture; compress later if desired.
"""

import argparse
import os
import time
import json
import cv2
import numpy as np
import multiprocessing as mp

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["random", "expert", "recover"])
    p.add_argument("--out_dir", default=None, help="Default: ./data_<mode>")
    p.add_argument("--workers", type=int, default=8)

    # Backward-compatible: episodes is PER WORKER unless episodes_total is set.
    p.add_argument("--episodes", type=int, default=50, help="Episodes per worker (unless --episodes_total is set)")
    p.add_argument("--episodes_total", type=int, default=None, help="Total episodes across all workers (overrides --episodes)")

    p.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--model", type=str, default="ppo_carracing_v3_perfected.zip")
    p.add_argument("--seed", type=int, default=0, help="Base seed (worker/episode derived)")
    p.add_argument("--visualize", action="store_true", help="Show render (forces 1 worker)")
    p.add_argument("--img_size", type=int, default=64)

    # Save options
    p.add_argument("--compress", action="store_true", default=True, help="Use np.savez_compressed (default)")
    p.add_argument("--no_compress", action="store_false", dest="compress", help="Use np.savez (faster, larger files)")
    p.add_argument("--write_json", action="store_true", help="Also write a .json sidecar next to each npz (optional)")
    p.add_argument("--save_metrics", action="store_true",
                   help="Also save speed/road_score arrays in the npz (off by default for compatibility)")

    # -----------------
    # Random corner-bias knobs (recommended for hairpins)
    # -----------------
    p.add_argument("--corner_bias", type=float, default=0.35,
                   help="Chance to start a short turn-burst when on-road (0..1).")
    p.add_argument("--turn_burst_min", type=int, default=3)
    p.add_argument("--turn_burst_max", type=int, default=8)
    p.add_argument("--turn_mag_min", type=float, default=0.55)
    p.add_argument("--turn_mag_max", type=float, default=1.00)
    p.add_argument("--turn_gas_min", type=float, default=0.35)
    p.add_argument("--turn_gas_max", type=float, default=0.80)
    p.add_argument("--turn_brake_prob", type=float, default=0.20)
    p.add_argument("--turn_brake_max", type=float, default=0.35)

    # -----------------
    # Off-road / stuck guards (prevents junk tails)
    # -----------------
    p.add_argument("--road_thr", type=float, default=0.20,
                   help="If road_score < road_thr, we consider it off-road-ish.")
    p.add_argument("--offroad_max_steps", type=int, default=80,
                   help="Terminate episode if off-road-ish this many consecutive steps.")
    p.add_argument("--stuck_speed_thr", type=float, default=1.5,
                   help="Speed below this counts as 'stuck'.")
    p.add_argument("--stuck_max_steps", type=int, default=80,
                   help="Terminate episode if stuck this many consecutive steps.")
    p.add_argument("--guard_warmup", type=int, default=40,
                   help="Do not apply guards for the first N steps of each episode.")

    # -----------------
    # recover-mode sabotage knobs (use low values)
    # -----------------
    p.add_argument("--sabotage_prob", type=float, default=0.003, help="Chance per step to start sabotage burst")
    p.add_argument("--sabotage_len_min", type=int, default=2)
    p.add_argument("--sabotage_len_max", type=int, default=5)
    p.add_argument("--sabotage_cooldown", type=int, default=220)
    p.add_argument("--sabotage_gas_min", type=float, default=0.2)
    p.add_argument("--sabotage_gas_max", type=float, default=0.5)
    return p.parse_args()


def _counter_path(out_dir: str, worker_id: int) -> str:
    return os.path.join(out_dir, f".transitions_w{worker_id:02d}.txt")

def _atomic_write_int(path: str, value: int):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(value)) + "\n")
    os.replace(tmp, path)

def _bump_worker_counter(out_dir: str, worker_id: int, delta: int):
    path = _counter_path(out_dir, worker_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cur = int(f.read().strip() or "0")
    except FileNotFoundError:
        cur = 0
    _atomic_write_int(path, cur + int(delta))

# -----------------------------
# Frame processing
# -----------------------------
def process_frame(frame_rgb: np.ndarray, img_size: int) -> np.ndarray:
    """Ensure uint8 RGB (img_size,img_size,3)."""
    if frame_rgb is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    frame = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


# -----------------------------
# Heuristics: speed + road score
# -----------------------------
def get_speed(base_env) -> float:
    """Approximate speed from physics; returns 0.0 if unavailable."""
    try:
        car = base_env.unwrapped.car
        v = car.hull.linearVelocity
        return float(np.sqrt(v[0] * v[0] + v[1] * v[1]))
    except Exception:
        return 0.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def road_score_from_img64(img64_u8: np.ndarray) -> float:
    """
    Soft road score in [0,1] from a 64x64 RGB uint8 image.
    This is deliberately simple and robust; it's only for *guarding* against junk tails.
    """
    # ROI near the bottom where road/grass is informative
    y1, y2, x1, x2 = 32, 62, 10, 54
    roi = img64_u8[y1:y2, x1:x2, :].astype(np.float32) / 255.0
    r = roi[..., 0]
    g = roi[..., 1]
    b = roi[..., 2]
    m = (r + g + b) / 3.0

    # grassness via green dominance
    green_index = g - 0.5 * (r + b)
    grass = sigmoid((green_index - 0.12) / 0.06)  # bias/scale tuned to CarRacing-ish palette

    # brightness gate (ignore very dark/very bright)
    gate_lo = sigmoid((m - 0.10) / 0.05)
    gate_hi = sigmoid((0.90 - m) / 0.05)
    gate = gate_lo * gate_hi

    road_w = (1.0 - grass) * gate
    return float(np.clip(road_w.mean(), 0.0, 1.0))


# -----------------------------
# Policies: random with corner bias
# -----------------------------
class RandomTurnPolicy:
    def __init__(self, rng: np.random.RandomState, args: argparse.Namespace):
        self.rng = rng
        self.args = args
        self.prev_steer = 0.0

        # turn burst state
        self.burst_left = 0
        self.burst_sign = 1.0
        self.burst_mag = 0.8

    def step(self, on_road: bool) -> np.ndarray:
        """
        If on-road, occasionally start a short high-steer burst to increase corner coverage.
        Otherwise fall back to correlated random steering.
        """
        a = self.args

        # Maintain burst if active
        if self.burst_left > 0:
            steer = float(np.clip(self.burst_sign * self.burst_mag, -1.0, 1.0))
            gas = float(self.rng.uniform(a.turn_gas_min, a.turn_gas_max))
            brake = 0.0
            if self.rng.rand() < a.turn_brake_prob:
                brake = float(self.rng.uniform(0.05, a.turn_brake_max))
                gas = min(gas, 0.55)  # avoid full throttle + brake simultaneously

            self.burst_left -= 1
            self.prev_steer = steer
            return np.array([steer, gas, brake], dtype=np.float32)

        # Possibly start a new burst (only if on-road)
        if on_road and (self.rng.rand() < a.corner_bias):
            self.burst_left = int(self.rng.randint(a.turn_burst_min, a.turn_burst_max + 1))
            self.burst_sign = 1.0 if self.rng.rand() > 0.5 else -1.0
            self.burst_mag = float(self.rng.uniform(a.turn_mag_min, a.turn_mag_max))
            # execute first burst step immediately
            return self.step(on_road=True)

        # Base correlated random
        noise = self.rng.uniform(-1.0, 1.0)
        steer = float(np.clip(0.4 * self.prev_steer + 0.6 * noise, -1.0, 1.0))

        mode = self.rng.choice(["accel", "brake", "coast"], p=[0.70, 0.10, 0.20])
        if mode == "accel":
            gas, brake = float(self.rng.uniform(0.3, 1.0)), 0.0
        elif mode == "brake":
            gas, brake = 0.0, float(self.rng.uniform(0.1, 0.8))
        else:
            gas, brake = 0.0, 0.0

        self.prev_steer = steer
        return np.array([steer, gas, brake], dtype=np.float32)


# -----------------------------
# Helpers: env access
# -----------------------------
def _get_base_env(env):
    """
    Return the underlying gymnasium env instance for rendering.
    Works for:
      - DummyVecEnv
      - VecFrameStack(DummyVecEnv)
    """
    if hasattr(env, "venv") and hasattr(env.venv, "envs"):
        return env.venv.envs[0]
    if hasattr(env, "envs"):
        return env.envs[0]
    if hasattr(env, "env") and hasattr(env.env, "envs"):
        return env.env.envs[0]
    raise RuntimeError("Could not locate base env for rendering.")


def _vec_reset(env, seed: int):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


# -----------------------------
# Worker
# -----------------------------
def run_worker(worker_id: int, args: argparse.Namespace):
    mode = args.mode
    out_dir = args.out_dir or f"./data_{mode}"
    os.makedirs(out_dir, exist_ok=True)

    # Reduce CPU oversubscription in multi-proc runs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    def make_env():
        return gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=args.steps)

    vec_env = DummyVecEnv([make_env])
    env = VecFrameStack(vec_env, n_stack=4) if mode in ("expert", "recover") else vec_env

    model = None
    if mode in ("expert", "recover"):
        model = PPO.load(args.model, device="cpu")
        if worker_id == 0:
            print(f"[worker {worker_id}] Loaded PPO: {args.model}")

    seed_worker = int(args.seed) + worker_id * 1000003
    rng = np.random.RandomState(seed_worker)

    # Episodes allocation
    if args.episodes_total is None:
        eps_this_worker = args.episodes
        total_eps = args.workers * args.episodes
    else:
        total_eps = int(args.episodes_total)
        base = total_eps // args.workers
        extra = total_eps % args.workers
        eps_this_worker = base + (1 if worker_id < extra else 0)

    if worker_id == 0:
        approx_frames = total_eps * args.steps
        print(f"--- Collecting {mode.upper()} data ---")
        print(f"out_dir={out_dir} (one .npz per episode)")
        print(f"workers={args.workers} total_episodes={total_eps} steps={args.steps} ~{approx_frames} frames")
        print(f"save={'np.savez_compressed' if args.compress else 'np.savez'}  img={args.img_size}x{args.img_size}")
        if mode == "random":
            print(f"random corner_bias={args.corner_bias} burst=[{args.turn_burst_min},{args.turn_burst_max}] "
                  f"turn_mag=[{args.turn_mag_min},{args.turn_mag_max}]")

    win_name = f"collect_data [{mode}]"
    if args.visualize:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    base_env = _get_base_env(env)

    # random policy instance (per worker)
    rand_pol = RandomTurnPolicy(rng, args)

    try:
        ep_local = 0
        while True:
            if not args.visualize and ep_local >= eps_this_worker:
                break

            ep_seed = int(seed_worker + ep_local * 99991)
            obs = _vec_reset(env, seed=ep_seed)

            frame0 = base_env.render()
            f0_64 = process_frame(frame0, args.img_size)

            obs_seq = [f0_64]
            act_seq = []
            rew_seq = []
            done_seq = []

            speed_seq = [get_speed(base_env)]
            road_seq = [road_score_from_img64(f0_64)]

            sabotage_left = 0
            cooldown = 0
            sabotage_action = np.zeros((1, 3), dtype=np.float32)

            offroad_run = 0
            stuck_run = 0

            for t in range(args.steps):
                # Guard signals computed from the latest saved obs frame
                rs = road_seq[-1]
                spd = speed_seq[-1]
                on_road = (rs >= args.road_thr)

                # Choose action
                if mode == "random":
                    a_flat = rand_pol.step(on_road=on_road)
                    a_env = a_flat[None, :]

                elif mode == "expert":
                    a_env, _ = model.predict(obs, deterministic=True)

                else:  # recover
                    if sabotage_left > 0:
                        a_env = sabotage_action
                        sabotage_left -= 1
                    else:
                        a_env, _ = model.predict(obs, deterministic=True)
                        if cooldown > 0:
                            cooldown -= 1
                        elif rng.rand() < args.sabotage_prob:
                            sabotage_left = int(rng.randint(args.sabotage_len_min, args.sabotage_len_max + 1))
                            cooldown = int(args.sabotage_cooldown)
                            force_steer = 1.0 if rng.rand() > 0.5 else -1.0
                            force_gas = float(rng.uniform(args.sabotage_gas_min, args.sabotage_gas_max))
                            sabotage_action = np.array([[force_steer, force_gas, 0.0]], dtype=np.float32)

                obs, reward, dones, infos = env.step(a_env)

                frame = base_env.render()
                f64 = process_frame(frame, args.img_size)

                # append transition
                obs_seq.append(f64)
                act_seq.append(a_env[0].astype(np.float32))
                r = float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward)
                rew_seq.append(r)
                done_flag = int(dones[0]) if isinstance(dones, (list, np.ndarray)) else int(dones)
                done_seq.append(done_flag)

                spd2 = get_speed(base_env)
                rs2 = road_score_from_img64(f64)
                speed_seq.append(spd2)
                road_seq.append(rs2)

                # Update guards (after warmup)
                if t >= args.guard_warmup:
                    if rs2 < args.road_thr:
                        offroad_run += 1
                    else:
                        offroad_run = 0

                    if spd2 < args.stuck_speed_thr:
                        stuck_run += 1
                    else:
                        stuck_run = 0

                    # Terminate episode early if we are just producing junk
                    if offroad_run >= args.offroad_max_steps or stuck_run >= args.stuck_max_steps:
                        # We do NOT mark done=1 artificially; we just end logging this episode.
                        break

                if args.visualize:
                    disp = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_NEAREST)
                    status = mode.upper()
                    if mode == "recover":
                        status = "SABOTAGE" if sabotage_left > 0 else "RECOVER"
                    cv2.putText(disp, f"{status}  w={worker_id} ep={ep_local} t={t}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(disp, f"road={rs2:.3f} speed={spd2:.2f} offrun={offroad_run} stuck={stuck_run}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    cv2.imshow(win_name, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return

                if done_flag:
                    break

            obs_arr = np.asarray(obs_seq, dtype=np.uint8)        # (T,H,W,3)
            act_arr = np.asarray(act_seq, dtype=np.float32)     # (T-1,3)
            rew_arr = np.asarray(rew_seq, dtype=np.float32)     # (T-1,)
            done_arr = np.asarray(done_seq, dtype=np.uint8)     # (T-1,)

            if obs_arr.shape[0] < 2 or act_arr.shape[0] < 1:
                ep_local += 1
                continue

            meta = {
                "mode": mode,
                "worker": int(worker_id),
                "episode": int(ep_local),
                "seed_worker": int(seed_worker),
                "seed_episode": int(ep_seed),
                "steps": int(act_arr.shape[0]),
                "timestamp": int(time.time()),
                "img_size": int(args.img_size),
                "compressed": bool(args.compress),
                "guards": {
                    "road_thr": float(args.road_thr),
                    "offroad_max_steps": int(args.offroad_max_steps),
                    "stuck_speed_thr": float(args.stuck_speed_thr),
                    "stuck_max_steps": int(args.stuck_max_steps),
                    "guard_warmup": int(args.guard_warmup),
                },
            }
            if mode == "random":
                meta["random_corner_bias"] = {
                    "corner_bias": float(args.corner_bias),
                    "turn_burst_min": int(args.turn_burst_min),
                    "turn_burst_max": int(args.turn_burst_max),
                    "turn_mag_min": float(args.turn_mag_min),
                    "turn_mag_max": float(args.turn_mag_max),
                }

            ts = int(time.time() * 1000)
            fname = os.path.join(out_dir, f"{mode}_w{worker_id:02d}_ep{ep_local:05d}_{ts}.npz")
            info_blob = np.bytes_(json.dumps(meta).encode("utf-8"))

            save_kwargs = dict(
                obs=obs_arr,
                action=act_arr,
                reward=rew_arr,
                done=done_arr,
                info=info_blob,
            )

            # Optional extras (off by default for compatibility)
            if args.save_metrics:
                # Align metrics to obs length (T)
                save_kwargs["speed"] = np.asarray(speed_seq[:obs_arr.shape[0]], dtype=np.float32)
                save_kwargs["road_score"] = np.asarray(road_seq[:obs_arr.shape[0]], dtype=np.float32)

            if args.compress:
                np.savez_compressed(fname, **save_kwargs)
                _bump_worker_counter(out_dir, worker_id, act_arr.shape[0])
            else:
                np.savez(fname, **save_kwargs)
                _bump_worker_counter(out_dir, worker_id, act_arr.shape[0])

            if args.write_json:
                with open(fname.replace(".npz", ".json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            if worker_id == 0 and (ep_local % 5 == 0):
                print(f"[worker {worker_id}] saved ep {ep_local} -> {os.path.basename(fname)}  T={obs_arr.shape[0]}")

            ep_local += 1

    finally:
        env.close()
        if args.visualize:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.visualize:
        args.workers = 1

    mp.set_start_method("spawn", force=True)

    if args.workers == 1:
        run_worker(0, args)
    else:
        with mp.Pool(args.workers) as pool:
            pool.starmap(run_worker, [(i, args) for i in range(args.workers)])


if __name__ == "__main__":
    main()
