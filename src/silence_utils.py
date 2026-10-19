import json
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
_missing_silence_logs = set()
_silence_file_index = {}

SILENCE_FEATURE_COMPONENTS = {
    "flag": ["flag"],
    "dur": ["dur"],
    "flag_dur": ["flag", "dur"],
    "dist": ["dist"],
    "flag_dur_dist": ["flag", "dur", "dist"],
}


def resolve_silence_components(feature_mode: str) -> List[str]:
    if feature_mode not in SILENCE_FEATURE_COMPONENTS:
        raise ValueError(f"unsupported silence_feature '{feature_mode}'")
    return SILENCE_FEATURE_COMPONENTS[feature_mode]


def silence_dim_for_mode(feature_mode: str) -> int:
    return len(resolve_silence_components(feature_mode))


def _candidate_silence_filenames(sub_id: str, lecture: Optional[str]) -> Iterable[str]:
    if sub_id:
        yield f"{sub_id}.silence.json"
        if not sub_id.endswith("_300k"):
            yield f"{sub_id}_300k.silence.json"
        else:
            yield f"{sub_id.rsplit('_300k', 1)[0]}.silence.json"
    if lecture:
        yield f"{lecture}.silence.json"
        if not lecture.endswith("_300k"):
            yield f"{lecture}_300k.silence.json"


def _get_silence_file_index(silence_root: str):
    cached = _silence_file_index.get(silence_root)
    if cached is not None:
        return cached

    index = {}
    if os.path.isdir(silence_root):
        for dirpath, _, filenames in os.walk(silence_root):
            for filename in filenames:
                if not filename.endswith(".silence.json"):
                    continue
                index.setdefault(filename, os.path.join(dirpath, filename))

    _silence_file_index[silence_root] = index
    return index


def load_silence_intervals(silence_root: str, sub_id: str, lecture: Optional[str] = None) -> List[Tuple[float, float]]:
    if not silence_root:
        return []
    for candidate in _candidate_silence_filenames(sub_id, lecture):
        file_path = os.path.join(silence_root, candidate)
        if not os.path.exists(file_path):
            file_path = _get_silence_file_index(silence_root).get(candidate, "")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    silences = json.load(f)
                except json.JSONDecodeError as exc:
                    logger.warning("failed to parse silence file %s: %s", file_path, exc)
                    return []
            intervals = []
            for item in silences:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
                intervals.append((start, end))
            intervals.sort(key=lambda span: span[0])
            return intervals
    key = (silence_root, sub_id)
    if key not in _missing_silence_logs:
        logger.warning("silence file not found for %s under %s", sub_id, silence_root)
        _missing_silence_logs.add(key)
    return []


SILENCE_FEATURE_DEF_VERSION = "fdef2"


def compute_silence_features(
    segment_times: Sequence[Tuple[float, float]],
    silence_intervals: Sequence[Tuple[float, float]],
    components: Sequence[str],
    near_sec: float,
    window_expand: float,
    max_dist: float,
    min_duration: float = 0.0,
) -> np.ndarray:
    """Paper-faithful silence features (definition version: fdef2).

    For clip i with midpoint t_mid, the local window is [t_mid - delta, t_mid + delta]
    where delta = near_sec (+ optional window_expand widening, default 0).
    Only silence intervals of length >= tau_min (= min_duration) are considered.

      f_flag : 1 iff some qualifying interval overlaps the window
      f_dur  : total overlap duration of all qualifying intervals with the window
      f_dist : min_{tau in E} |t_mid - tau| over start/end timestamps of qualifying
               intervals, clipped at d_max and divided by d_max (1.0 if none)
    """
    feature_dim = len(components)
    if feature_dim == 0 or not segment_times:
        return np.zeros((0, 0), dtype=np.float32)

    silence_intervals = list(silence_intervals)
    result = np.zeros((len(segment_times), feature_dim), dtype=np.float32)
    max_dist = max(max_dist, 1e-3)
    half_window = max(float(near_sec) + float(window_expand), 0.0)
    eps = 1e-9

    for idx, (seg_start, seg_end) in enumerate(segment_times):
        seg_mid = (float(seg_start) + float(seg_end)) * 0.5
        window_start = seg_mid - half_window
        window_end = seg_mid + half_window

        flag = 0
        dur = 0.0
        nearest = None
        for s_start, s_end in silence_intervals:
            s_start = float(s_start)
            s_end = float(s_end)
            if (s_end - s_start) < float(min_duration) - eps:
                continue
            overlap = min(window_end, s_end) - max(window_start, s_start)
            if overlap > eps:
                flag = 1
                dur += overlap
            d = min(abs(seg_mid - s_start), abs(seg_mid - s_end))
            nearest = d if nearest is None else min(nearest, d)

        norm_dist = 1.0 if nearest is None else min(nearest, max_dist) / max_dist

        values = []
        for comp in components:
            if comp == "flag":
                values.append(float(flag))
            elif comp == "dur":
                values.append(float(dur))
            elif comp == "dist":
                values.append(float(norm_dist))
            else:
                raise ValueError("unsupported silence component {}".format(comp))
        result[idx] = values
    return result


def get_silence_cache_path(cache_root: str, split: str, example_id: str) -> Optional[str]:
    if not cache_root:
        return None
    safe_id = example_id.replace("@@", "__")
    cache_dir = os.path.join(cache_root, split)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe_id}.silence.npy")


def load_cached_silence_features(cache_path: Optional[str], expected_len: int) -> Optional[np.ndarray]:
    if not cache_path:
        return None
    candidate_paths = [cache_path]
    if cache_path.endswith(".silence.npy"):
        candidate_paths.append(cache_path[: -len(".silence.npy")] + ".pause.npy")
    elif cache_path.endswith(".pause.npy"):
        candidate_paths.append(cache_path[: -len(".pause.npy")] + ".silence.npy")

    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        try:
            arr = np.load(candidate_path)
        except Exception:
            continue
        if arr.shape[0] != expected_len:
            continue
        return arr
    return None


def save_silence_features(cache_path: Optional[str], array: np.ndarray) -> None:
    if cache_path is None:
        return
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(cache_path, array.astype(np.float32))


PAUSE_FEATURE_COMPONENTS = SILENCE_FEATURE_COMPONENTS


def resolve_pause_components(feature_mode: str) -> List[str]:
    return resolve_silence_components(feature_mode)


def pause_dim_for_mode(feature_mode: str) -> int:
    return silence_dim_for_mode(feature_mode)


def load_pause_intervals(pause_root: str, sub_id: str, lecture: Optional[str] = None) -> List[Tuple[float, float]]:
    return load_silence_intervals(pause_root, sub_id, lecture)


def compute_pause_features(
    segment_times: Sequence[Tuple[float, float]],
    pause_intervals: Sequence[Tuple[float, float]],
    components: Sequence[str],
    near_sec: float,
    window_expand: float,
    max_dist: float,
) -> np.ndarray:
    return compute_silence_features(
        segment_times,
        pause_intervals,
        components,
        near_sec,
        window_expand,
        max_dist,
    )


def get_pause_cache_path(cache_root: str, split: str, example_id: str) -> Optional[str]:
    return get_silence_cache_path(cache_root, split, example_id)


def load_cached_pause_features(cache_path: Optional[str], expected_len: int) -> Optional[np.ndarray]:
    return load_cached_silence_features(cache_path, expected_len)


def save_pause_features(cache_path: Optional[str], array: np.ndarray) -> None:
    save_silence_features(cache_path, array)


# --- revision controls: EXP-C (shuffled silence) and EXP-H (modulation signal logging) ---
_SILENCE_SIGNAL_LOG = []


def shuffle_silence_batch(silence_features, seed):
    """Circularly shift each example's silence sequence by a per-example offset.

    Keeps every marginal statistic of the silence signal intact (same values, same counts)
    while destroying its alignment with the transcript, which is exactly the EXP-C control.
    """
    if seed is None or seed <= 0 or silence_features is None:
        return silence_features
    import torch as _t
    out = silence_features.clone()
    n = silence_features.shape[1]
    if n <= 1:
        return out
    g = _t.Generator(device="cpu").manual_seed(int(seed))
    for b in range(silence_features.shape[0]):
        shift = int(_t.randint(1, n, (1,), generator=g).item())
        out[b] = _t.roll(silence_features[b], shifts=shift, dims=0)
    return out


def record_silence_signal(tag, signal):
    """Append per-token modulation statistics for EXP-H."""
    if signal is None:
        return
    with_no_grad = signal.detach().float()
    _SILENCE_SIGNAL_LOG.append({
        "tag": tag,
        "mean": float(with_no_grad.mean()),
        "std": float(with_no_grad.std()) if with_no_grad.numel() > 1 else 0.0,
        "min": float(with_no_grad.min()),
        "max": float(with_no_grad.max()),
        "per_token": with_no_grad.mean(dim=-1).cpu().numpy().tolist(),
    })


def dump_silence_signal(path):
    import json as _j, os as _o
    if not path:
        return
    _o.makedirs(_o.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        _j.dump(_SILENCE_SIGNAL_LOG, f)


def clear_silence_signal():
    _SILENCE_SIGNAL_LOG.clear()
