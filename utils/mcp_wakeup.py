# mcp_wakeup.py
import asyncio
import time
import logging
from typing import Optional, Callable

import httpx

logger = logging.getLogger("mcp_wakeup")
logger.addHandler(logging.NullHandler())

# ---- Defaults you can override when calling functions ----
DEFAULT_PROBE_TIMEOUT = 5.0
DEFAULT_MAX_WAKE_SECONDS = 120
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 10.0

# ---- Core utilities ----

async def probe_server(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT, headers: dict | None = None) -> Optional[int]:
    """
    Probe a given URL once. Returns HTTP status code if reachable, or None on network error.
    - Accepts headers for authenticated probes (e.g., Authorization).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            return resp.status_code
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError) as exc:
        logger.debug("probe_server exception: %s", exc)
        return None

async def wait_for_wakeup(
    probe_url: str,
    headers: dict | None = None,
    max_wait: int = DEFAULT_MAX_WAKE_SECONDS,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    accept_status_predicate=None,
    on_wakeup_message: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Polls probe_url until it responds with a status considered 'up' (default: status < 500),
    or until max_wait seconds elapse.

    Returns True if the server became reachable (status accepted), False on timeout.

    Parameters:
    - probe_url: URL to probe (e.g., root or /health).
    - headers: optional HTTP headers for probe.
    - max_wait: overall maximum wait time in seconds.
    - initial_backoff / max_backoff: backoff parameters (seconds).
    - probe_timeout: timeout for each probe attempt.
    - accept_status_predicate: optional function(int)->bool to decide which HTTP codes mean "up".
        Default: lambda status: status is not None and status < 500
    - on_wakeup_message: optional callback(msg) called once when waking begins (for UI).
    """
    accept_status_predicate = accept_status_predicate or (lambda status: status is not None and status < 500)

    start = time.time()
    backoff = initial_backoff
    first_notify = True

    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            logger.warning("wait_for_wakeup: timed out after %.1f seconds", elapsed)
            return False

        status = await probe_server(probe_url, timeout=probe_timeout, headers=headers)

        if accept_status_predicate(status):
            # If we printed a wakeup message previously, optionally indicate success.
            if not first_notify and on_wakeup_message:
                try:
                    on_wakeup_message("Server is up (status {}).".format(status))
                except Exception:
                    pass
            return True

        # Not up yet -> notify first time and continue polling
        if first_notify:
            msg = "Server appears to be asleep / unreachable. Attempting to wake (polling {})...".format(probe_url)
            if on_wakeup_message:
                try:
                    on_wakeup_message(msg)
                except Exception:
                    pass
            else:
                logger.info(msg)
            first_notify = False
        else:
            logger.debug("Still waiting for server: status=%s, elapsed=%.1f", status, elapsed)

        await asyncio.sleep(min(backoff, max_backoff))
        backoff *= 1.8
