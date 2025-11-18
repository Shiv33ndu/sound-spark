# src/tools/synthesis_demo.py
import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter, fftconvolve
from typing import Dict, Any, Optional

# from src.tools.code_exec_tool import interpret_instructions

def load_mono(path: str, sr: int = 22050):
    y, sr2 = librosa.load(path, sr=sr, mono=True)
    return y, sr

def _sine_wave(freq_hz: float, duration_s: float, sr: int = 22050, amp: float = 0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return amp * np.sin(2 * np.pi * freq_hz * t)

def _lowpass(signal, cutoff, sr, order=4):
    nyquist = 0.5 * sr
    norm_cutoff = max(1e-6, min(cutoff / nyquist, 0.999))
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def _highpass(signal, cutoff, sr, order=4):
    nyquist = 0.5 * sr
    norm_cutoff = max(1e-6, min(cutoff / nyquist, 0.999))
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return lfilter(b, a, signal)

def _soft_distort(signal, drive=1.0):
    # simple tanh distortion
    return np.tanh(signal * drive)

def _add_delay(signal, sr, delay_ms=60, feedback=0.2):
    delay_s = delay_ms / 1000.0
    delay_samples = int(sr * delay_s)
    out = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        out[i] += feedback * out[i - delay_samples]
    # normalize
    m = np.max(np.abs(out)) + 1e-9
    if m > 1.0:
        out = out / m * 0.95
    return out

def _add_noise(signal, noise_amp=0.02):
    noise = np.random.randn(len(signal)) * noise_amp
    return signal + noise

def apply_patch(
    input_audio_path: str,
    out_path: str,
    instructions: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    sr: int = 22050,
    mix_ratio: float = 0.75
) -> Dict[str, Any]:
    """
    High-level tool: loads input audio, interprets instructions or params,
    applies synthesis and effects and writes out_path.
    Returns metadata with applied params and path.

    arg:
        input_audio_path: original audio file path uploaded by the user 
        out_path: path of a new synthesized file being written to
        instructions: LLM given JSON based instructions
        params: LLM suggested tweaks
        sr: sample rate of the audio file
        mix_ration: wet and dry ratio of the fx into original file

    return:
        JSON String that has output path to synthesized file, and applied params on it 
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # load
    y, sr = load_mono(input_audio_path, sr=sr)
    duration = len(y) / sr

    # If params are explicitly provided, trust them. Otherwise parse instructions.
    # if params is None:
    #     params = interpret_instructions(instructions or "", y, sr)

    # Start with original (or silence) and build layers
    base = y.copy()
    out = base.copy()

    # SUB-SINE layer
    if params.get("sub_sine", {}).get("enabled", False):
        f = params["sub_sine"].get("freq_hz", params["sub_sine"].get("ratio_freq_hz", 55.0))
        amp = params["sub_sine"].get("amp", 0.5)
        sub = _sine_wave(f, duration, sr=sr, amp=amp)
        # optional lowpass on sub
        if params["sub_sine"].get("lowpass_cutoff"):
            sub = _lowpass(sub, params["sub_sine"]["lowpass_cutoff"], sr)
        out = out * mix_ratio + sub * (1.0 - mix_ratio)

    # Noise
    if params.get("noise", {}).get("enabled", False):
        out = _add_noise(out, params["noise"].get("amp", 0.01))

    # Distortion
    if params.get("distortion", {}).get("enabled", False):
        drive = params["distortion"].get("drive", 1.0)
        out = _soft_distort(out, drive=drive)

    # Lowpass/Highpass global
    if params.get("global_lowpass"):
        out = _lowpass(out, params["global_lowpass"], sr)
    if params.get("global_highpass"):
        out = _highpass(out, params["global_highpass"], sr)

    # Delay
    if params.get("delay", {}).get("enabled", False):
        out = _add_delay(out, sr, delay_ms=params["delay"].get("ms", 60), feedback=params["delay"].get("feedback", 0.15))

    # Normalize and clip-safe
    maxv = np.max(np.abs(out)) + 1e-9
    if maxv > 1.0:
        out = out / maxv * 0.95

    sf.write(out_path, out.astype(np.float32), sr)

    return {"ok": True, "path": out_path, "params": params}
