import json, os

def classify_descriptors(descriptors: dict, llm):
    """
    Heuristic classifier for MVP:
      - uses spectral_centroid, zero_crossing_rate, rms to infer texture/tags
      - returns JSON-serializable dict; optionally refines via LLM if LLM_API_KEY present
    """
    sc = descriptors.get("spectral_centroid") or 0.0
    zcr = descriptors.get("zero_crossing_rate") or 0.0
    rms = descriptors.get("rms") or 0.0

    tags = []
    confidence = 0.6

    # Texture heuristics
    if sc is None:
        texture = "unknown"
    elif sc > 3000:
        texture = "bright"
        tags.append("bright")
        confidence = 0.75
    elif sc > 1500:
        texture = "mid-heavy"
        tags.append("thick")
        confidence = 0.7
    else:
        texture = "warm"
        tags.append("warm")
        confidence = 0.7

    # Additional tags
    try:
        if zcr > 0.06:
            tags.append("percussive")
            confidence += 0.05
    except Exception:
        pass
    try:
        if rms > 0.08:
            tags.append("loud")
            confidence += 0.05
    except Exception:
        pass

    confidence = min(0.95, round(confidence, 2))

    # Optional LLM refinement
    if os.getenv("LLM_API_KEY"):
        try:
            prompt = f"SYSTEM: classify audio descriptors to tags/texture JSON only. USER: {json.dumps(descriptors)}"
            resp = llm.call(prompt)
            parsed = json.loads(resp.get("text", "{}"))
            if isinstance(parsed, dict) and parsed.get("style_tags"):
                # trust LLM response if valid
                return {
                    "style_tags": parsed.get("style_tags", []),
                    "genre_suggestions": parsed.get("genre_suggestions", []),
                    "texture": parsed.get("texture", texture),
                    "confidence": parsed.get("confidence", confidence)
                }
        except Exception:
            pass

    return {
        "style_tags": tags,
        "genre_suggestions": [],
        "texture": texture,
        "confidence": confidence
    }
