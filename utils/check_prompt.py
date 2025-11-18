def has_audio_path(prompt: str) -> bool:
    """
    This utility checks whether the given prompt has audio path or not

    args: 
        prompt : string = user message
    
    return:
        boolean True or false
    """
    return any(prompt.lower().endswith(ext) for ext in [".wav", ".mp3", ".flac", ".ogg", ".aiff"]) \
        or "/audio/" in prompt.lower() or "\\audio\\" in prompt.lower()
