"""
calltracer: A debugging module with a decorator (CallTracer) for
tracing function calls and a function (stack) for logging the current call stack.
"""

import contextvars
import functools
import inspect
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Optional

# Define a logger for the entire module.
tracer_logger = logging.getLogger(__name__)

# A single context variable will hold the list of calls (the chain).
# It works safely for both sync and async code.
tracer_chain = contextvars.ContextVar("tracer_chain", default=[])

# A context var to control whether tracing is enabled for the current call stack.
tracing_enabled_context = contextvars.ContextVar("tracing_enabled", default=True)

# Accum for the daughter functions exec time
sub_duration_aggregator = contextvars.ContextVar(
    "sub_duration_aggregator", default=defaultdict(int)
)


### USEFUL CONSTANTS

# Constants for time conversion for better readability
_NS_IN_US = 1_000.0
_NS_IN_MS = 1_000_000.0
_NS_IN_SEC = 1_000_000_000.0
_NS_IN_MIN = 60 * _NS_IN_SEC
_NS_IN_HR = 60 * _NS_IN_MIN


class DFMT(Enum):
    """Defines the formatting style for time duration."""

    NANO = auto()
    MICRO = auto()
    SEC = auto()
    SINGLE = auto()
    HUMAN = auto()


_TIMERS = {
    "m": time.monotonic_ns,
    "h": time.perf_counter_ns,
    "c": time.process_time_ns,
    "t": time.thread_time_ns,
}

_TIMING_BLOCK_WIDTH = 50


### AUXILIARY FUNCTIONS


def _readable_duration(duration: int, fmt: DFMT) -> str:
    """
    Formats a duration given in nanoseconds into a human-readable string.

    Args:
        duration (int): The time duration in nanoseconds.
        fmt (DFMT): The desired output format.

    Returns:
        str: The formatted duration string.
    """
    if fmt == DFMT.NANO:
        return f"{duration} ns"

    if fmt == DFMT.MICRO:
        return f"{duration / _NS_IN_US} µs"

    if fmt == DFMT.SEC:
        return f"{duration / _NS_IN_SEC} s"

    if fmt == DFMT.SINGLE:
        if duration < _NS_IN_MS:
            return f"{duration} ns"
        if duration < 100 * _NS_IN_MS:
            return f"{duration / _NS_IN_US} µs"
        if duration < _NS_IN_MIN:
            return f"{duration / _NS_IN_SEC} s"
        if duration < _NS_IN_HR:
            return f"{duration / _NS_IN_MIN} min"
        return f"{duration / _NS_IN_HR} hr"

    if fmt == DFMT.HUMAN:
        if duration < 100 * _NS_IN_MS:
            # For durations less than 100ms, behavior is identical to SINGLE
            return _readable_duration(duration, DFMT.SINGLE)

        # Decompose duration into hours, minutes, and seconds
        hours, remainder_ns = divmod(duration, _NS_IN_HR)
        minutes, remainder_ns = divmod(remainder_ns, _NS_IN_MIN)
        seconds = remainder_ns / _NS_IN_SEC

        hours = int(hours)
        minutes = int(minutes)

        # Build the string based on non-zero values
        if hours > 0:
            return f"{hours} hr, {minutes} min, {seconds} s"
        if minutes > 0:
            return f"{minutes} min, {seconds} s"
        return f"{seconds} s"

    # Fallback for any unknown format
    raise ValueError(f"Unknown duration format: {fmt}")


def _get_timing_block(
    inclusive_durs: dict, exclusive_durs: dict, timing_str: str, fmt: DFMT
) -> str:
    """
    Generates a formatted block of execution times, choosing between
    inclusive and exclusive times based on the case of the timing character.
    """
    if not timing_str:
        return ""

    parts = []
    for char in timing_str:
        lower_char = char.lower()
        if lower_char in inclusive_durs:
            if char.isupper():
                duration = exclusive_durs[lower_char]
            else:
                duration = inclusive_durs[lower_char]

            readable_dur = _readable_duration(duration, fmt)
            # В лейбле используем оригинальный символ (T или t)
            parts.append(f"{char}: {readable_dur}")

    if not parts:
        return ""

    unpadded_block = f"[{' | '.join(parts)}]"
    return f"{unpadded_block:<{_TIMING_BLOCK_WIDTH}}"


def _get_arg_str(func, args, kwargs, tracer_instance):
    """
    Helper function to generate a string representation of function arguments.
    It applies transformations and length limits based on the tracer's config.
    """
    func_name = func.__qualname__
    try:
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        processed_args = []
        for name, value in bound_args.arguments.items():
            transform_key = (func_name, name)
            universal_transform_key = ("*", name)
            # Use tracer_instance to access the configuration
            if transform_key in tracer_instance.transform:
                display_value = tracer_instance.transform[transform_key](value)
            elif universal_transform_key in tracer_instance.transform:
                display_value = tracer_instance.transform[universal_transform_key](
                    value
                )
            else:
                display_value = value

            if display_value is not None:
                if tracer_instance.max_argval_len == 0:
                    val_str = "..."
                else:
                    val_str = repr(display_value)
                    if (
                        tracer_instance.max_argval_len and
                        len(val_str) > tracer_instance.max_argval_len
                    ):
                        val_str = val_str[: tracer_instance.max_argval_len] + "..."
                processed_args.append(f"{name}={val_str}")
            else:
                processed_args.append(f"{name}")

        arg_str = ", ".join(processed_args)

    except (ValueError, TypeError):  # pragma: no cover
        arg_str = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )

    return arg_str


def _log_trace_event(tracer, event_type: str, data: dict):
    """Builds a data record and logs it as either text or JSON."""

    base_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }

    if tracer.output == "json":
        log_string = json.dumps(base_record)
    else:  # 'text' format
        timing_block = data.get("timing_block", "")
        indent = data.get("indent", "")
        current_call_sig = data.get("current_call_sig", "")

        if data.get("ide") or data.get("osc8"):
            filename = data.get("filename")
            lineno = data.get("lineno")
            if data.get("ide"):
                output = f'File "{filename}", line {data.get("lineno")}, in {current_call_sig}'
            else:
                output = f"\033]8;;file://{filename}#{lineno}\033\\{current_call_sig}\033]8;;\033\\"
        else:
            output = (
                f"Calling {current_call_sig}"
                if event_type == "enter"
                else f"Exiting {current_call_sig}"
            )

        if event_type == "enter":
            log_string = f"{indent}--> {output}"
            if tracer.trace_chain and data.get("chain"):
                chain_str = " <== ".join(reversed(data["chain"]))
                log_string += f"  <== {chain_str}"

        elif event_type == "exit":
            display_result = repr(
                tracer.return_transform(data["result"])
                if tracer.return_transform
                else data["result"]
            )

            if (
                tracer.max_return_len is not None
                and len(display_result) > tracer.max_return_len
            ):
                display_result = display_result[: tracer.max_return_len] + "..."

            log_string = (
                f"{timing_block}{indent}<-- {output}, returned: {display_result}"
            )

        elif event_type == "exception":
            log_string = f"{timing_block}{indent}<!> {output} with exception: {repr(data['exception'])}"

    level = logging.WARNING if event_type == "exception" else tracer.level
    tracer.logger.log(level, log_string)


### OUR CLASSES


class _BaseTracer:
    """A common base class to hold shared initialization logic."""

    def __init__(
        self,
        level=logging.DEBUG,
        trace_chain=False,
        logger=None,
        transform=None,
        max_argval_len=None,
        return_transform: Optional[Callable] = None,
        max_return_len=None,
        condition: Optional[Callable] = None,
        timing: str = None,
        timing_fmt: DFMT = DFMT.SINGLE,
        output: str = "text",
        ide_support: bool = False,
        term_support: bool = False,
        rel_path: bool = True,
    ):
        """
        Initializes the factory.

        Args:
            level (int): The logging level for trace messages.
            trace_chain (bool): If True, accumulates and logs the call chain.
            logger (logging.Logger): The logger instance to use.
            transform (dict, optional): A dictionary of callbacks to transform
                argument values before logging. The key is a tuple of
                (func_qualname, arg_name), and the value is a callable that
                receives the argument's value and returns a new value for display.
                If returns None, only the argument name will be printed
                Example: {('MyClass.login', 'password'): lambda p: '***',
                          ('MyClass.method', 'self'): lambda s: None}
            max_argval_len (int, optional): Maximum length for the string
                representation of argument values in logs.
                - If None (default), no truncation is performed.
                - If 0, argument values are hidden (displayed as '...').
                - If > 0, the string representation is truncated to this length.
            return_transform (Callable, optional): A function to transform
                the return value before logging.
            max_return_len (int, optional): Maximum length for the string
                representation of the return value. Works like max_argval_len.
                Defaults to None (no limit).
            condition (Callable, optional): A function that determines if tracing
                should be active for a call. It receives the function's
                qualified name and its arguments (`(func_name, *args, **kwargs)`)
                and must return True or False. If it returns False, this
                call and all nested decorated calls will not be traced.
                Defaults to None (always trace).
            timing (str, optional): Enables "poor man's profiling". A string of
                characters specifying which clocks to use. The order of
                characters in the string is preserved in the output.

                Clock selection characters:
                'm': time.monotonic_ns
                'h': time.perf_counter_ns
                'c': time.process_time_ns
                't': time.thread_time_ns

                The case of each character determines the type of measurement:
                - **Lowercase** (m, h, c, t): Measures **inclusive** time
                  (the total time from the start to the end of the
                  function call).
                - **Uppercase** (M, H, C, T): Measures **exclusive** time
                  (the inclusive time minus the time spent in any directly
                  called child functions that are also decorated).
                Example: `timing="Mh"` will show exclusive monotonic time
                and inclusive perf_counter time.
                Defaults to None (disabled).
            timing_fmt (DFMT, optional): The format for displaying timing values.
                Defaults to DFMT.SINGLE. Possible values from the DFMT enum:
                - NANO: Plain nanoseconds (e.g., "123 ns").
                - MICRO: Plain microseconds (e.g., "0.123 µs").
                - SEC: Plain seconds (e.g., "0.000000123 s").
                - SINGLE: A "smart" format that uses the single most
                  appropriate unit (ns, µs, s, min, or hr) depending
                  on the magnitude.
                - HUMAN: A compound, human-readable format for larger
                  durations (e.g., "5 min, 23.4 s"). For small values
                  (<100ms), it behaves like SINGLE.
            output (str, optional): The output format for trace records.
                Can be 'text' (default) or 'json'.
            ide_support (bool, optional): If True, includes the file path
                and line number of the function definition in text logs
                using a format like 'File "path.py", line N, in func_sig'.
                This makes log entries clickable in IDEs like PyCharm and VSCode,
                linking directly to the source code.
                Only applies when output='text'.
                Defaults to False.
            term_support (bool, optional): If True, the file and a line number
                are printer as a hyperlink in OSC 8 format, supported by
                modern terminals (check the terminal docs to find out how to
                use it, like Cmd-Click on iTerm2).
                Only applies when output='text'.
                No effect if ide_support==True.
                Defaults to False.
            rel_path (bool, optional): If True and ide_support or term_support
                are True, uses relative path for filename, othervise absolute.
                Defaults to True
        """
        self.level = level
        self.trace_chain = trace_chain
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform or {}
        self.max_argval_len = max_argval_len
        self.return_transform = return_transform
        self.max_return_len = max_return_len
        self.condition = condition
        self.output = output
        self.enabled = True
        self.ide_support = ide_support
        self.term_support = term_support
        self.rel_path = rel_path

        self.timing_funcs = []
        self.timing = timing
        self.timing_fmt = timing_fmt
        if self.timing:
            for char in self.timing.lower():
                if char in _TIMERS:
                    self.timing_funcs.append(_TIMERS[char])

    def enable(self):
        """Enables tracing for this decorator instance."""
        self.enabled = True

    def disable(self):
        """Disables tracing for this decorator instance."""
        self.enabled = False


class CallTracer(_BaseTracer):  # pylint: disable=too-few-public-methods
    """A factory for creating decorators that trace SYNCHRONOUS function/method calls.

    This class, when instantiated, creates a decorator that can be applied to any
    function or method to log its entry, exit, arguments, and return value.

    Example:
        trace = CallTracer(level=logging.INFO)

        @trace
        def my_function(x, y):
            return x + y
    """

    def __call__(self, func):
        """Makes the instance callable and returns the actual decorator wrapper.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function that includes tracing logic.
        """
        if inspect.iscoroutinefunction(func):
            raise TypeError("Use aCallTracer for async functions")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not tracing_enabled_context.get() or not self.enabled:
                return func(*args, **kwargs)

            should_run_tracer = (
                self.condition(func.__qualname__, *args, **kwargs)
                if self.condition
                else True
            )
            token_enabled = tracing_enabled_context.set(should_run_tracer)

            if not should_run_tracer:
                try:
                    return func(*args, **kwargs)
                finally:
                    tracing_enabled_context.reset(token_enabled)

            chain = tracer_chain.get()
            indent = "    " * len(chain)
            arg_str = _get_arg_str(func, args, kwargs, self)
            current_call_sig = f"{func.__qualname__}({arg_str})"
            if (self.ide_support or self.term_support) and self.output == "text":
                filename = inspect.getfile(func)
                filename = (
                    os.path.relpath(filename)
                    if self.rel_path
                    else os.path.abspath(filename)
                )
                try:
                    lineno = (
                        inspect.getsourcelines(func)[1] + 1
                    )  # lineno is the decorator's line :)
                except OSError:
                    lineno = "undef"
            else:
                filename = None
                lineno = None

            _log_trace_event(
                self,
                "enter",
                {
                    "indent": indent,
                    "current_call_sig": current_call_sig,
                    "chain": chain,
                    "filename": filename,
                    "lineno": lineno,
                    "ide": self.ide_support,
                    "osc8": self.term_support,
                },
            )

            exc = None
            token_chain = tracer_chain.set(chain + [current_call_sig])
            parent_aggregator = sub_duration_aggregator.get()
            token_agg = sub_duration_aggregator.set(defaultdict(int))
            start_times = (
                {
                    char: timer()
                    for char, timer in zip(self.timing.lower(), self.timing_funcs)
                }
                if self.timing
                else {}
            )

            try:
                result = func(*args, **kwargs)
                # everyting is in finally!
                return result
            except Exception as e:
                exc = e
                # everyting is in finally!
                raise
            finally:
                end_times = (
                    {
                        char: timer()
                        for char, timer in zip(self.timing.lower(), self.timing_funcs)
                    }
                    if self.timing
                    else {}
                )
                children_aggregator = sub_duration_aggregator.get()
                inclusive_durs, exclusive_durs = {}, {}
                if self.timing:
                    for char in self.timing.lower():
                        total_duration = end_times[char] - start_times[char]
                        inclusive_durs[char] = total_duration
                        exclusive_durs[char] = (
                            total_duration - children_aggregator[char]
                        )
                        parent_aggregator[char] += total_duration

                sub_duration_aggregator.reset(token_agg)
                timing_block = _get_timing_block(
                    inclusive_durs, exclusive_durs, self.timing, self.timing_fmt
                )

                log_data = {
                    "indent": indent,
                    "current_call_sig": current_call_sig,
                    "timing_block": timing_block,
                    "timings_ns": {
                        "inclusive": inclusive_durs,
                        "exclusive": exclusive_durs,
                    },
                    "filename": filename,
                    "lineno": lineno,
                    "ide": self.ide_support,
                    "osc8": self.term_support,
                }

                if exc is None:
                    log_data["result"] = result
                    _log_trace_event(self, "exit", log_data)
                else:
                    log_data["exception"] = exc
                    _log_trace_event(self, "exception", log_data)

                tracer_chain.reset(token_chain)
                tracing_enabled_context.reset(token_enabled)

        return sync_wrapper


class aCallTracer(_BaseTracer):  # pylint: disable=too-few-public-methods
    """A factory for creating decorators that trace ASYNCHRONOUS function calls."""

    def __call__(self, func):
        """Makes the instance callable and returns the actual decorator wrapper.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function that includes tracing logic.
        """
        if not inspect.iscoroutinefunction(func):
            raise TypeError("Use CallTracer for sync functions.")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not tracing_enabled_context.get() or not self.enabled:
                return await func(*args, **kwargs)

            should_run_tracer = (
                self.condition(func.__qualname__, *args, **kwargs)
                if self.condition
                else True
            )
            token_enabled = tracing_enabled_context.set(should_run_tracer)

            if not should_run_tracer:
                try:
                    return await func(*args, **kwargs)
                finally:
                    tracing_enabled_context.reset(token_enabled)

            chain = tracer_chain.get()
            indent = "    " * len(chain)
            arg_str = _get_arg_str(func, args, kwargs, self)
            current_call_sig = f"{func.__qualname__}({arg_str})"

            if (self.ide_support or self.term_support) and self.output == "text":
                filename = inspect.getfile(func)
                filename = (
                    os.path.relpath(filename)
                    if self.rel_path
                    else os.path.abspath(filename)
                )
                lineno = inspect.getsourcelines(func)[1]
            else:
                filename = None
                lineno = None

            _log_trace_event(
                self,
                "enter",
                {
                    "indent": indent,
                    "current_call_sig": current_call_sig,
                    "chain": chain,
                    "filename": filename,
                    "lineno": lineno,
                    "ide": self.ide_support,
                    "osc8": self.term_support,
                },
            )

            exc = None
            token_chain = tracer_chain.set(chain + [current_call_sig])
            parent_aggregator = sub_duration_aggregator.get()
            token_agg = sub_duration_aggregator.set(defaultdict(int))
            start_times = (
                {
                    char: timer()
                    for char, timer in zip(self.timing.lower(), self.timing_funcs)
                }
                if self.timing
                else {}
            )

            try:
                result = await func(*args, **kwargs)
                # everyting is in finally!
                return result
            except Exception as e:
                exc = e
                # everyting is in finally!
                raise
            finally:
                end_times = (
                    {
                        char: timer()
                        for char, timer in zip(self.timing.lower(), self.timing_funcs)
                    }
                    if self.timing
                    else {}
                )
                children_aggregator = sub_duration_aggregator.get()
                inclusive_durs, exclusive_durs = {}, {}
                if self.timing:
                    for char in self.timing.lower():
                        total_duration = end_times[char] - start_times[char]
                        inclusive_durs[char] = total_duration
                        exclusive_durs[char] = (
                            total_duration - children_aggregator[char]
                        )
                        parent_aggregator[char] += total_duration

                sub_duration_aggregator.reset(token_agg)
                timing_block = _get_timing_block(
                    inclusive_durs, exclusive_durs, self.timing, self.timing_fmt
                )

                log_data = {
                    "indent": indent,
                    "current_call_sig": current_call_sig,
                    "timing_block": timing_block,
                    "timings_ns": {
                        "inclusive": inclusive_durs,
                        "exclusive": exclusive_durs,
                    },
                    "filename": filename,
                    "lineno": lineno,
                    "ide": self.ide_support,
                    "osc8": self.term_support,
                }

                if exc is None:
                    log_data["result"] = result
                    _log_trace_event(self, "exit", log_data)
                else:
                    log_data["exception"] = exc
                    _log_trace_event(self, "exception", log_data)

                tracer_chain.reset(token_chain)
                tracing_enabled_context.reset(token_enabled)

        return async_wrapper


no_self = {("*", "self"): (lambda _: None)}


### OUR FUNCTION


def stack(level=logging.DEBUG, logger=tracer_logger, limit=None, start=0):
    """Logs the current call stack to the specified logger.

    This function creates a "snapshot" of how the code reached this point,
    which is useful for point-in-time debugging.

    Args:
        level (int): The logging level for the message. Defaults to logging.DEBUG.
        logger (logging.Logger): The logger instance to use. Defaults to the module logger.
        limit (int, optional): The maximum number of frames to display. Defaults to None (all).
        start (int, optional): The offset of the first frame to display. Defaults to 0.
    """
    frames = inspect.stack()

    caller_frame = frames[1]
    caller_file, caller_line, caller_func = (
        caller_frame.filename,
        caller_frame.lineno,
        caller_frame.function,
    )

    logger.log(
        level, "Stack trace at %s:%d in %s():", caller_file, caller_line, caller_func
    )

    begin = start + 2
    end = min(begin + limit, len(frames)) if limit else len(frames)

    # This loop is corrected to access frame attributes by name
    for frame_info in frames[begin:end]:
        logger.log(
            level,
            "  ↳ Called from: %s, line %d, in %s",
            frame_info.filename,
            frame_info.lineno,
            frame_info.function,
        )
