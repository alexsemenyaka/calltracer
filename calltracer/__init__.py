"""
pytracecall: A debugging module with sync and async decorators (CallTracer and aCallTracer)
for tracing function calls, and a function (stack) for logging the current call stack
"""

import importlib.metadata

from .calltracer import CallTracer, aCallTracer, stack, tracer_indent

_metadata = importlib.metadata.metadata("pytracecall")
__version__ = _metadata["Version"]
__author__ = _metadata["Author-email"]
__license__ = _metadata["License"]

__all__ = ["CallTracer", "aCallTracer", "stack", "tracer_indent"]
