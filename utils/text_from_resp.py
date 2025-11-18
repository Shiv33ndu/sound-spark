import json
from typing import Any, List, Optional, Union

def _safe_get(o: Any, attr: str, default=None):
    """Get attribute or dict key safely."""
    if o is None:
        return default
    # dict-like
    if isinstance(o, dict):
        return o.get(attr, default)
    # object-like
    return getattr(o, attr, default)

def extract_human_text(
    events: List[Any],
    all: bool = False,
    parse_json: bool = False
) -> Optional[Union[str, List[str], dict]]:
    """
    Extract human-readable text parts from ADK events.
    
    Args:
        events: list of Event objects or dicts returned by ADK.
        all: if True, return a list of all text parts found (may be empty).
             if False (default), return the first text part found or None.
        parse_json: if True, attempt to json.loads the returned text (first or each).
    
    Returns:
        - If all=False and parse_json=False: first text string or None.
        - If all=False and parse_json=True: parsed JSON or raw string fallback or None.
        - If all=True and parse_json=False: list of text strings (possibly empty).
        - If all=True and parse_json=True: list with parsed JSON when possible or strings on fallback.
    """
    if not events:
        return [] if all else None

    found_texts: List[str] = []

    for event in events:
        content = _safe_get(event, "content")
        if content is None:
            continue

        parts = _safe_get(content, "parts")
        # If parts is None or not iterable, skip
        if not parts:
            continue

        # parts might be iterable-like; ensure we iterate safely
        try:
            iterator = iter(parts)
        except TypeError:
            continue

        for part in iterator:
            # part.text (attribute) or part.get('text') (dict)
            text = _safe_get(part, "text")
            if text is None:
                # sometimes 'text' may be inside a dict under 'content' or similar - try dict keys
                if isinstance(part, dict):
                    # try a few alternative keys if present
                    for alt_key in ("text", "body", "message"):
                        if alt_key in part and part[alt_key]:
                            text = part[alt_key]
                            break
            if not text:
                continue

            # normalize to str and strip
            try:
                text_str = str(text).strip()
            except Exception:
                text_str = None

            if not text_str:
                continue

            if parse_json:
                try:
                    parsed = json.loads(text_str)
                    found_texts.append(parsed)
                except Exception:
                    # fallback to raw text when JSON parsing fails
                    found_texts.append(text_str)
            else:
                found_texts.append(text_str)

            if not all:
                # return first found (possibly parsed)
                return found_texts[0]

    # end for
    if all:
        return found_texts
    return None
