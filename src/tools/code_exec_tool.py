# src/tools/code_exec_tool.py
import re
from typing import Dict, Any
from src.tools.synthesis_demo import apply_patch


ALLOWED_FUNCTIONS = {
    "apply_patch": apply_patch
}

def simple_freq_extract(text: str):
    # find frequencies like 120Hz, 120 Hz, or numbers followed by Hz
    m = re.search(r"(\d{2,4})\s*hz", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # find ms for delay
    return None

def interpret_instructions(instructions: str, sr: int = 22050):
    """
    Very small rule-based parser: returns params dict.
    LLM should ideally output structured params; this parser helps when LLM gives plain text.

    args:
        instructions: str, expected JSON string but if in case its plain text
        sr: sample rate
  
    """
    t = instructions.lower()
    params = {}

    # Sub-sine: look for "sub", "one octave below", "octave"
    if "sub" in t or "one octave" in t or "octave below" in t:
        # estimate pitch from audio if needed: for simplicity default 55Hz
        base_freq = 55.0
        if "hz" in t:
            f = simple_freq_extract(t)
            if f:
                base_freq = f
        params["sub_sine"] = {"enabled": True, "freq_hz": base_freq / 1.0, "amp": 0.45}
        # optional lowpass
        if "lowpass" in t:
            f_lp = simple_freq_extract(t) or 120.0
            params["sub_sine"]["lowpass_cutoff"] = f_lp

    # Distortion
    if "distort" in t or "distortion" in t or "drive" in t:
        drive = 1.0
        dm = re.search(r"drive\s*(?:=|:)?\s*(\d(?:\.\d)?)", t)
        if dm:
            try:
                drive = float(dm.group(1))
            except:
                pass
        params["distortion"] = {"enabled": True, "drive": drive}

    # Noise
    if "noise" in t:
        na = re.search(r"noise\s*(?:amp)?\s*(?:=|:)?\s*(0?\.\d+|\d)", t)
        amp = 0.01
        if na:
            try:
                amp = float(na.group(1))
            except:
                pass
        params["noise"] = {"enabled": True, "amp": amp}

    # Delay
    if "delay" in t or "echo" in t:
        ms = 60
        mm = re.search(r"(\d{1,3})\s*ms", t)
        if mm:
            ms = int(mm.group(1))
        fb = 0.15
        fbm = re.search(r"feedback\s*(?:=|:)?\s*(0?\.\d+|\d)", t)
        if fbm:
            try:
                fb = float(fbm.group(1))
            except:
                pass
        params["delay"] = {"enabled": True, "ms": ms, "feedback": fb}

    # Global filters
    if "lowpass" in t and "sub" not in t:
        f_lp = simple_freq_extract(t) or 8000
        params["global_lowpass"] = f_lp
    if "highpass" in t:
        f_hp = simple_freq_extract(t) or 20
        params["global_highpass"] = f_hp

    # Default fallback: enable sub-sine if nothing recognized
    if not params:
        params = {"sub_sine": {"enabled": True, "freq_hz": 55.0, "amp": 0.4}}

    return params

def execute_tool(function: str, args: Dict[str, Any], file_path: str, out_path: str) -> Dict[str, Any]:
    """
    Validate and execute a limited set of functions. Returns JSON-serializable result.

    args:
        function: takes the enum function name 
        args: LLM JSON string's keys 
    return:
        JSON String a python DICT 
    """
    if function not in ALLOWED_FUNCTIONS:
        return {"ok": False, "error": f"Function {function} not allowed."}
    # required args
    if "input_audio_path" not in args or "out_path" not in args:
        return {"ok": False, "error": "Missing required args: input_audio_path and out_path"}

    instructions = args.get("instructions", "")
    params = args.get("params")  # structured override
    sr = int(args.get("sr", 22050))
    mix_ratio = float(args.get("mix_ratio", 0.75))

    # if instructions present and no params provided, parse them:
    if params is None:
        params = interpret_instructions(instructions or "", None, sr)

    # finally call the function
    try:
        res = ALLOWED_FUNCTIONS[function](
            input_audio_path=file_path,
            out_path=out_path,
            instructions=instructions,
            params=params,
            sr=sr,
            mix_ratio=mix_ratio
        )
        return {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}
