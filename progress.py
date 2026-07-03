"""Progress reporting for long pipeline runs.

Everything goes to stderr, flushed, so the CLI's stdout (the rendered draft)
stays clean and the run is observable live via `tail -f` when redirected.

Listeners (the Gradio UI) can subscribe with add_listener(fn); each gets
(monotonic_time, msg) per log call. Listener errors never break the run.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

_listeners: list[Callable[[float, str], None]] = []


def add_listener(fn: Callable[[float, str], None]) -> None:
    _listeners.append(fn)


def remove_listener(fn: Callable[[float, str], None]) -> None:
    try:
        _listeners.remove(fn)
    except ValueError:
        pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
    now = time.monotonic()
    for fn in list(_listeners):
        try:
            fn(now, msg)
        except Exception:
            pass
