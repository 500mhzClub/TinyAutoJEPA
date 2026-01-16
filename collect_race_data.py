
import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import time
import warnings
from typing import Optional, Tuple

# --- CONFIGURATION ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(min(32, mp.cpu_count()))))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "80"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "600"))

DATA_DIR = os.getenv("DATA_DIR", "data_raceline")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))
CHUNK_FRAMES = int(os.getenv("CHUNK_FRAMES", "20000"))  # flush periodically to avoid huge RAM

# Pure Pursuit / controller tuning
LOOKAHEAD_BASE = float(os.getenv("LOOKAHEAD_BASE", "10.0"))   # base lookahead in "track points"
LOOKAHEAD_GAIN = float(os.getenv("LOOKAHEAD_GAIN", "1.2"))    # additional lookahead per m/s
STEER_GAIN = float(os.getenv("STEER_GAIN", "1.2"))            # overall steering gain

# Speed control (units are Box2D-ish; treat as relative tuning)
V_MAX = float(os.getenv("V_MAX", "32.0"))     # top target speed on straights
V_MIN = float(os.getenv("V_MIN", "12.0"))     # minimum target speed for sharp turns
CURV_GAIN = float(os.getenv("CURV_GAIN", "55.0"))  # how aggressively curvature lowers speed
BRAKE_GAIN = float(os.getenv("BRAKE_GAIN", "0.8"))  # brake strength when overspeed
GAS_GAIN = float(os.getenv("GAS_GAIN", "0.045"))    # throttle proportional gain

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # CarRacing obs is ~96x96; crop top HUD-ish area a bit (keeps road + horizon)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def _wrap_idx(i: int, n: int) -> int:
    # Always returns 0..n-1
    return int(i % n)

def _extract_track_xy(track_item) -> Optional[Tuple[float, float]]:
    """
    Gymnasium CarRacing track items have varied formats across forks/versions.
    Common legacy format: (alpha, beta, x, y)
    We attempt to robustly extract (x,y).
    """
    if track_item is None:
        return None

    # If it's a dict-like
    if isinstance(track_item, dict):
        if "x" in track_item and "y" in track_item:
            return float(track_item["x"]), float(track_item["y"])
        return None

    # If it's a tuple/list
    if isinstance(track_item, (tuple, list)):
        L = len(track_item)
        if L == 4:
            # (alpha, beta, x, y) is the most common
            x, y = track_item[2], track_item[3]
            return float(x), float(y)
        if L == 3:
            a, b, c = track_item[0], track_item[1], track_item[2]
            # Heuristic: angles are ~[-pi, pi], coordinates are typically much larger magnitude.
            def is_angle(v: float) -> bool:
                return abs(float(v)) <= 3.6

            # (x, y, angle)
            if (not is_angle(a)) and (not is_angle(b)) and is_angle(c):
                return float(a), float(b)
            # (angle, x, y)
            if is_angle(a) and (not is_angle(b)) and (not is_angle(c)):
                return float(b), float(c)
            # (x, angle, y) or other weirdness: pick the two least angle-like
            vals = [float(a), float(b), float(c)]
            angle_flags = [is_angle(v) for v in vals]
            xy = [v for v, is_a in zip(vals, angle_flags) if not is_a]
            if len(xy) >= 2:
                return xy[0], xy[1]
            return None
        # If longer, try last two as x,y
        if L >= 2:
            x, y = track_item[-2], track_item[-1]
            try:
                return float(x), float(y)
            except Exception:
                return None

    return None

def get_centerline(env) -> np.ndarray:
    """
    Returns Nx2 array of track center points in world coordinates.
    """
    track = getattr(env.unwrapped, "track", None)
    if track is None or len(track) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for item in track:
        xy = _extract_track_xy(item)
        if xy is not None:
            pts.append(xy)

    if len(pts) < 10:
        return np.zeros((0, 2), dtype=np.float32)

    # Remove obvious duplicates (track sometimes repeats points)
    arr = np.asarray(pts, dtype=np.float32)
    # Keep points that move at least epsilon
    keep = [0]
    eps = 1e-3
    for i in range(1, len(arr)):
        if np.linalg.norm(arr[i] - arr[keep[-1]]) > eps:
            keep.append(i)
    arr = arr[keep]
    return arr

def car_state(env) -> Tuple[np.ndarray, float, float]:
    """
    Returns (pos_xy, heading_angle, speed).
    """
    car = env.unwrapped.car
    hull = car.hull
    pos = np.array([hull.position.x, hull.position.y], dtype=np.float32)
    heading = float(hull.angle)  # radians
    v = hull.linearVelocity
    speed = float(np.sqrt(v.x * v.x + v.y * v.y))
    return pos, heading, speed

def nearest_index(path_xy: np.ndarray, pos_xy: np.ndarray, prev_idx: Optional[int], window: int = 80) -> int:
    """
    Windowed nearest-point search around prev_idx for speed.
    Falls back to global nearest if prev_idx is None.
    """
    n = len(path_xy)
    if n == 0:
        return 0

    if prev_idx is None:
        d2 = np.sum((path_xy - pos_xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    # search in [prev_idx - window, prev_idx + window]
    idxs = np.arange(prev_idx - window, prev_idx + window + 1)
    idxs = np.array([_wrap_idx(i, n) for i in idxs], dtype=np.int32)
    cand = path_xy[idxs]
    d2 = np.sum((cand - pos_xy[None, :]) ** 2, axis=1)
    return int(idxs[int(np.argmin(d2))])

def curvature_ahead(path_xy: np.ndarray, idx: int, look: int = 18) -> float:
    """
    Simple curvature proxy using turning angle over a short horizon.
    Returns a nonnegative scalar; higher => tighter turn.
    """
    n = len(path_xy)
    if n < (look + 3):
        return 0.0

    i0 = _wrap_idx(idx, n)
    i1 = _wrap_idx(idx + look, n)
    i2 = _wrap_idx(idx + 2 * look, n)

    p0 = path_xy[i0]
    p1 = path_xy[i1]
    p2 = path_xy[i2]

    v1 = p1 - p0
    v2 = p2 - p1

    n1 = np.linalg.norm(v1) + 1e-8
    n2 = np.linalg.norm(v2) + 1e-8

    v1 /= n1
    v2 /= n2

    # angle between segments
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    ang = float(np.arccos(dot))  # [0, pi]
    # normalize by segment length to get a curvature-like proxy
    return ang / float(look)

def pure_pursuit_steer(pos_xy: np.ndarray, heading: float, target_xy: np.ndarray, steer_gain: float) -> float:
    """
    Pure pursuit steering: transform target into car frame and compute curvature.
    """
    # rotation into car frame (car x-axis forward)
    dx, dy = float(target_xy[0] - pos_xy[0]), float(target_xy[1] - pos_xy[1])
    ch, sh = float(np.cos(heading)), float(np.sin(heading))

    # world -> car frame
    x_car =  ch * dx + sh * dy
    y_car = -sh * dx + ch * dy

    # If target is behind, damp steering to avoid flip
    if x_car < 1e-3:
        return 0.0

    Ld2 = x_car * x_car + y_car * y_car + 1e-8
    curvature = 2.0 * y_car / Ld2  # signed
    steer = steer_gain * curvature
    return float(np.clip(steer, -1.0, 1.0))

def raceline_controller(path_xy: np.ndarray, env, prev_idx: Optional[int], prev_steer: float) -> Tuple[np.ndarray, int, float]:
    """
    Returns (action, new_prev_idx, new_prev_steer)
    action = [steer, gas, brake]
    """
    pos, heading, speed = car_state(env)

    if len(path_xy) < 10:
        # Fallback: drive straight-ish if no path
        steer = 0.0
        gas = 0.6
        brake = 0.0
        return np.array([steer, gas, brake], dtype=np.float32), 0, float(steer)

    idx = nearest_index(path_xy, pos, prev_idx, window=90)

    # Speed-adaptive lookahead (in index-steps along discrete centerline points)
    lookahead = int(np.clip(LOOKAHEAD_BASE + LOOKAHEAD_GAIN * speed, 6.0, 55.0))
    target_idx = _wrap_idx(idx + lookahead, len(path_xy))
    target = path_xy[target_idx]

    steer_raw = pure_pursuit_steer(pos, heading, target, STEER_GAIN)

    # Mild temporal smoothing to reduce oscillations
    steer = float(np.clip(0.25 * prev_steer + 0.75 * steer_raw, -1.0, 1.0))

    # Curvature-based target speed (look a bit ahead so you slow before the bend)
    curv = curvature_ahead(path_xy, idx, look=16)
    v_target = float(np.clip(V_MAX - CURV_GAIN * curv, V_MIN, V_MAX))

    # Throttle/brake: proportional control on speed error with overspeed braking
    err = v_target - speed
    gas = float(np.clip(GAS_GAIN * err, 0.0, 1.0))

    # If overspeed or tight curvature, brake more
    overspeed = speed - v_target
    brake = 0.0
    if overspeed > 1.0:
        brake = float(np.clip(BRAKE_GAIN * (overspeed / max(v_target, 1.0)), 0.0, 1.0))
        gas = 0.0

    # Extra caution: in very sharp turns, bias toward coasting if steering is large
    if abs(steer) > 0.75 and speed > (v_target + 2.0):
        gas = 0.0
        brake = float(max(brake, 0.2))

    action = np.array([steer, gas, brake], dtype=np.float32)
    return action, idx, steer

def flush_chunk(data_dir: str, worker_id: int, chunk_id: int,
                states, actions, next_states) -> None:
    if len(states) == 0:
        return
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"raceline_chunk_w{worker_id:02d}_{chunk_id:05d}.npz")
    np.savez(
        filename,
        states=np.asarray(states, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.uint8),
    )
    print(f"[Worker {worker_id}] wrote {len(states):,} frames -> {filename}")

def worker_func(worker_id: int) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    seed = int(time.time()) + worker_id * 10000
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    try:
        env = gym.make("CarRacing-v3", render_mode=None, max_episode_steps=MAX_STEPS)
    except Exception:
        env = gym.make("CarRacing-v2", render_mode=None)

    states, actions, next_states = [], [], []
    prev_idx: Optional[int] = None
    prev_steer = 0.0
    chunk_id = 0

    try:
        for ep in range(EPISODES_PER_WORKER):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            # Track is generated on reset; extract centerline once per episode
            path_xy = get_centerline(env)

            prev_idx = None
            prev_steer = 0.0

            for _ in range(MAX_STEPS):
                s = process_frame(obs)

                action, prev_idx, prev_steer = raceline_controller(path_xy, env, prev_idx, prev_steer)

                obs2, _, terminated, truncated, _ = env.step(action)
                ns = process_frame(obs2)

                states.append(s)
                actions.append(action)
                next_states.append(ns)

                obs = obs2

                # Periodic flush to keep memory bounded
                if len(states) >= CHUNK_FRAMES:
                    flush_chunk(DATA_DIR, worker_id, chunk_id, states, actions, next_states)
                    chunk_id += 1
                    states, actions, next_states = [], [], []

                if terminated or truncated:
                    break
    finally:
        env.close()

    # Flush remainder
    flush_chunk(DATA_DIR, worker_id, chunk_id, states, actions, next_states)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Racing Line Data Collection (Pure Pursuit + Curvature Speed Control) ===")
    print(f"workers={NUM_WORKERS} episodes/worker={EPISODES_PER_WORKER} max_steps={MAX_STEPS} chunk_frames={CHUNK_FRAMES}")
    print(f"data_dir={DATA_DIR}")

    mp.set_start_method("spawn", force=True)
    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(worker_func, range(NUM_WORKERS))
