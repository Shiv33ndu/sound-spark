import librosa
import numpy as np
from typing import Dict, Any

def _to_scalar(x):
    """Convert numpy arrays / numpy scalars / iterables to Python native floats/ints when possible."""
    if x is None:
        return None
    # if numpy array or list/tuple -> try to pick a representative scalar
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return None
            # prefer single-element value if present, else mean
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
            return float(arr.mean())
        except Exception:
            try:
                return float(x[0])
            except Exception:
                return None
    if isinstance(x, np.generic):
        return x.item()
    try:
        return float(x)
    except Exception:
        return x

def compute_basic_descriptors(path: str, sr: int = 22050) -> Dict[str, Any]:
    """
    Compute lightweight descriptors for an audio file and return JSON-safe python types.
    Tempo is guaranteed to be either a float or None.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = float(len(y) / sr)

    # tempo (may be scalar or array-like)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"[feature extractor] : The tempo = {tempo}")
        changed_tempo = _to_scalar(tempo)
        print(f"[feature extractor] : Changed The tempo = {changed_tempo}")
        # ensure tempo is float (or None)
        if tempo is not None:
            try:
                tempo = float(tempo)
            except Exception:
                pass
    except Exception:
        tempo = None

    # spectral features
    spec_cent = _to_scalar(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bw = _to_scalar(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = _to_scalar(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = _to_scalar(np.mean(librosa.feature.rms(y=y)))

    return {
        "duration": duration,
        "tempo": changed_tempo,
        "spectral_centroid": spec_cent,
        "spectral_bandwidth": spec_bw,
        "zero_crossing_rate": zcr,
        "rms": rms
    }