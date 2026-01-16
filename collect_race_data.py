import gymnasium as gym
import numpy as np
import multiprocessing as mp
import os
import cv2
import math
import time

# --- CONFIGURATION ---
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "32"))
EPISODES_PER_WORKER = int(os.getenv("EPISODES_PER_WORKER", "50"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1000"))
DATA_DIR = os.getenv("DATA_DIR", "data_race")
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))

# --- Controller tuning (safer defaults) ---
BASE_LOOKAHEAD = float(os.getenv("BASE_LOOKAHEAD", "10.0"))
LOOKAHEAD_SPEED = float(os.getenv("LOOKAHEAD_SPEED", "0.20"))

STANLEY_K = float(os.getenv("STANLEY_K", "0.7"))
HEADING_GAIN = float(os.getenv("HEADING_GAIN", "1.0"))

V_MAX = float(os.getenv("V_MAX", "35.0"))
V_MIN = float(os.getenv("V_MIN", "10.0"))
CURVATURE_SLOWDOWN = float(os.getenv("CURVATURE_SLOWDOWN", "20.0"))

GAS_MAX = float(os.getenv("GAS_MAX", "0.65"))
BRAKE_MAX = float(os.getenv("BRAKE_MAX", "0.55"))

# Off-road detection hysteresis
OFFROAD_GRACE_STEPS = int(os.getenv("OFFROAD_GRACE_STEPS", "60"))
OFFROAD_STREAK_KILL = int(os.getenv("OFFROAD_STREAK_KILL", "25"))

# Segment acceptance
MIN_FRAMES_TO_KEEP = int(os.getenv("MIN_FRAMES_TO_KEEP", "120"))

# Steering rate limit (per-step)
MAX_STEER_DELTA = float(os.getenv("MAX_STEER_DELTA", "0.12"))


def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = frame[:84, :, :]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def _wrap_idx(i: int, n: int) -> int:
    return int(i % n)


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _track_half_width(u) -> float:
    """
    Gym CarRacing uses TRACK_WIDTH as HALF road width in world units (used with +/- normal offsets).
    If not present, fall back to a conservative guess.
    """
    for name in ("TRACK_WIDTH", "track_width", "ROAD_WIDTH", "road_width"):
        v = getattr(u, name, None)
        if isinstance(v, (int, float)) and v > 0:
            # If someone exposed ROAD_WIDTH as full width, we still treat as half-width
            # only when it looks like the classic TRACK_WIDTH scale. Heuristic:
            vv = float(v)
            if name.lower().endswith("width") and vv > 12.0:
                return vv * 0.5
            return vv
    return 7.0  # conservative half-width fallback


def _closest_idx(track_xy: np.ndarray, car_pos: np.ndarray, prev_idx: int | None) -> int:
    n = int(len(track_xy))
    if n == 0:
        return 0

    if prev_idx is None:
        d = np.linalg.norm(track_xy - car_pos[None, :], axis=1)
        return int(np.argmin(d))

    # Bidirectional local search around previous index (handles overshoot / spins)
    back = 25
    fwd = 60
    offsets = np.arange(-back, fwd, dtype=np.int32)
    idxs = (prev_idx + offsets) % n
    d = np.linalg.norm(track_xy[idxs] - car_pos[None, :], axis=1)
    return int(idxs[int(np.argmin(d))])


def _local_path_yaw(track_xy: np.ndarray, idx: int) -> float:
    n = int(len(track_xy))
    p0 = track_xy[idx]
    p1 = track_xy[_wrap_idx(idx + 3, n)]
    v = p1 - p0
    if float(np.linalg.norm(v)) < 1e-3:
        p1 = track_xy[_wrap_idx(idx + 8, n)]
        v = p1 - p0
    return math.atan2(float(v[1]), float(v[0]))


def _cross_track_error(track_xy: np.ndarray, idx: int, car_pos: np.ndarray) -> tuple[float, float]:
    """
    Returns (cte_signed, path_yaw) computed at the LOCAL closest waypoint.
    """
    path_yaw = _local_path_yaw(track_xy, idx)
    p0 = track_xy[idx]
    dx = float(car_pos[0] - p0[0])
    dy = float(car_pos[1] - p0[1])
    left_n = np.array([-math.sin(path_yaw), math.cos(path_yaw)], dtype=np.float32)
    cte = float(dx * left_n[0] + dy * left_n[1])
    return cte, path_yaw


def expert_controller(env, prev_idx: int | None, prev_steer: float, step_count: int):
    """
    Stable pursuit + Stanley-like correction, with geometric offroad detection.
    Returns: action, new_prev_idx, new_prev_steer, offroad_bool
    """
    u = env.unwrapped
    car = getattr(u, "car", None)
    track = getattr(u, "track", None)

    if car is None or track is None or len(track) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), prev_idx, prev_steer, False

    car_pos = np.array(car.hull.position, dtype=np.float32)
    car_angle = float(car.hull.angle)
    car_vel = np.array(car.hull.linearVelocity, dtype=np.float32)
    speed = float(np.linalg.norm(car_vel))

    track_xy = np.array([t[2:4] for t in track], dtype=np.float32)
    n = int(len(track_xy))

    # 1) Closest waypoint (robust local search)
    closest = _closest_idx(track_xy, car_pos, prev_idx)

    # 2) Lookahead selection by arc distance proxy (Euclidean steps)
    L = BASE_LOOKAHEAD + LOOKAHEAD_SPEED * min(speed, 40.0)
    target = closest
    for i in range(1, 120):
        j = _wrap_idx(closest + i, n)
        if float(np.linalg.norm(track_xy[j] - car_pos)) >= L:
            target = j
            break

    # 3) Heading and CTE computed at LOCAL closest (more reliable for offroad + control)
    cte, path_yaw = _cross_track_error(track_xy, closest, car_pos)
    heading_err = _angle_wrap(path_yaw - car_angle)

    # 4) Stanley steering
    stanley_term = math.atan2(STANLEY_K * cte, max(4.0, speed))
    raw_steer = HEADING_GAIN * heading_err + stanley_term
    raw_steer = float(np.clip(raw_steer, -1.0, 1.0))

    # 5) Rate-limit steering (real smoothing)
    steer = float(np.clip(raw_steer, prev_steer - MAX_STEER_DELTA, prev_steer + MAX_STEER_DELTA))

    # 6) Curvature estimate (ahead)
    a = track_xy[_wrap_idx(closest + 8, n)]
    b = track_xy[_wrap_idx(closest + 20, n)]
    c = track_xy[_wrap_idx(closest + 35, n)]
    ab = b - a
    bc = c - b
    ang = abs(_angle_wrap(
        math.atan2(float(bc[1]), float(bc[0])) - math.atan2(float(ab[1]), float(ab[0]))
    ))

    v_target = V_MAX - CURVATURE_SLOWDOWN * ang
    v_target = float(np.clip(v_target, V_MIN, V_MAX))

    # Gentle ramp-up early in episode
    if step_count < 80:
        ramp = step_count / 80.0
        v_target = V_MIN + (v_target - V_MIN) * ramp

    # 7) Longitudinal control
    speed_err = v_target - speed
    gas = 0.0
    brake = 0.0

    if speed_err > 1.5:
        gas = GAS_MAX * min(1.0, speed_err / 10.0) * (1.0 - 0.45 * abs(steer))
    elif speed_err < -1.5:
        brake = BRAKE_MAX * min(1.0, abs(speed_err) / 12.0)

    # Additional braking on strong steering
    if abs(steer) > 0.65:
        brake = max(brake, 0.20)

    action = np.array([steer, gas, brake], dtype=np.float32)

    # 8) Offroad detection (geometry-based)
    tw = _track_half_width(u)          # half road width
    off_road = abs(cte) > (tw * 1.10)  # slightly lenient threshold

    return action, closest, steer, off_road


def worker_func(worker_id: int):
    seed = int(time.time()) + worker_id * 1000
    rng = np.random.RandomState(seed)

    try:
        env = gym.make("CarRacing-v3", render_mode=None)
    except Exception:
        env = gym.make("CarRacing-v2", render_mode=None)

    states, actions = [], []
    success_count = 0
    fail_count = 0

    for ep in range(EPISODES_PER_WORKER):
        obs, _ = env.reset(seed=int(rng.randint(0, 1_000_000)))
        prev_idx = None
        prev_steer = 0.0

        ep_states, ep_actions = [], []
        off_road_streak = 0
        step_count = 0

        for t in range(MAX_STEPS):
            s = process_frame(obs)

            action, prev_idx, prev_steer, off_road = expert_controller(env, prev_idx, prev_steer, step_count)
            step_count += 1

            # Grace window (ignore offroad early while car stabilizes)
            if t < OFFROAD_GRACE_STEPS:
                off_road = False

            if off_road:
                off_road_streak += 1
            else:
                off_road_streak = max(0, off_road_streak - 1)

            if off_road_streak >= OFFROAD_STREAK_KILL:
                # Kill + discard this episode segment
                ep_states, ep_actions = [], []
                fail_count += 1
                break

            obs, _, term, trunc, _ = env.step(action)

            # Skip the very first frames (camera + spawn jitter)
            if t >= 10:
                ep_states.append(s)
                ep_actions.append(action)

            if term or trunc:
                break

        if len(ep_states) >= MIN_FRAMES_TO_KEEP:
            states.extend(ep_states)
            actions.extend(ep_actions)
            success_count += 1

        if (ep + 1) % 10 == 0:
            print(
                f"Worker {worker_id}: Ep {ep+1}/{EPISODES_PER_WORKER} | "
                f"Success: {success_count} | Fail (OffRoad): {fail_count} | "
                f"Frames: {len(states)}"
            )

    env.close()

    if len(states) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"race_{worker_id}.npz")
        np.savez(fname, states=np.array(states, dtype=np.uint8), actions=np.array(actions, dtype=np.float32))
        print(f"Worker {worker_id}: DONE. Saved {len(states)} CLEAN frames -> {fname}")
    else:
        print(f"Worker {worker_id}: FAILED. No valid data collected.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    with mp.Pool(NUM_WORKERS) as p:
        p.map(worker_func, range(NUM_WORKERS))
