#!/usr/bin/env python3
"""
A clear and simple example of using pytracecall with the RichPyTraceHandler.
"""

import logging
import time

from calltracer import DFMT, CallTracer, RichPyTraceHandler

# --- 1. Logger and Handler Setup ---
# We set up the logger once. We will swap out the handler to demo both modes.
log = logging.getLogger("rex_clean_demo")
log.setLevel(logging.DEBUG)
log.propagate = False  # Prevent duplicate output

# Create handlers for both display modes
append_handler = RichPyTraceHandler(overwrite=False)
overwrite_handler = RichPyTraceHandler(overwrite=True)


# --- 2. Tracer Configuration ---
# We only need one tracer instance. The display mode is controlled by the handler.
trace = CallTracer(logger=log, output="json", timing="Mh", timing_fmt=DFMT.SINGLE)


# --- 3. Function Definition ---
# The function is decorated in the standard way with the @ syntax.
@trace
def shurumburum(n: int, m: int, d: float = 0.0) -> int:
    """
    A recursive function to demonstrate tracing.
    """
    # Base case added to prevent infinite recursion from the original example
    if n <= 0 or m <= 0:
        return n

    time.sleep(d)

    match n % 2:
        case 0:
            return shurumburum(n / 2, m - 1, d)
        case 1:
            return shurumburum(m - (m % 2), n * 2, d) + shurumburum(n + 1, m // 2, d)


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("--- DEMO 1: Append-Only Mode (overwrite=False, default) ---")
    # Use the handler for the append-only mode
    log.handlers = [append_handler]
    shurumburum(13, 5)

    print("\n\n--- DEMO 2: Overwrite (Live) Mode (overwrite=True) ---")
    # Swap the logger's handler to demonstrate the other mode
    log.handlers = [overwrite_handler]
    shurumburum(13, 5, 0.5)
