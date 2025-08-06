"""
calltracer: A debugging module with a decorator (CallTracer) for
tracing function calls and a function (stack) for logging the current call stack.
"""
import time
import asyncio
import functools
import inspect
import logging
import contextvars
from enum import Enum, auto
import math

# Define a logger for the entire module.
tracer_logger = logging.getLogger(__name__)

# A single context variable will hold the list of calls (the chain).
# It works safely for both sync and async code.
tracer_chain = contextvars.ContextVar('tracer_chain', default=[])


### USEFUL CONSTANTS

# Constants for time conversion for better readability
_NS_IN_US = 1_000.0  # наносекунд в микросекунде
_NS_IN_MS = 1_000_000.0  # наносекунд в миллисекунде
_NS_IN_SEC = 1_000_000_000.0  # наносекунд в секунде
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
    'm': time.monotonic_ns,
    'h': time.perf_counter_ns,
    'c': time.process_time_ns,
    't': time.thread_time_ns,
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

def _get_timing_block(start_times: list, end_times: list, timing_str: str, fmt: DFMT) -> str:
    """
    Generates a formatted block of execution times based on start/end timestamps.
    Preserves the order from the original timing_str.
    """
    if not start_times:
        return ""
    
    labels = timing_str
    
    parts = [
        f"{label}: {_readable_duration(end - start, fmt)}"
        for label, start, end in zip(labels, start_times, end_times)
    ]
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
            universal_transform_key = ('*', name)
            # Use tracer_instance to access the configuration
            if transform_key in tracer_instance.transform:
                display_value = tracer_instance.transform[transform_key](value)
            elif universal_transform_key in tracer_instance.transform:
                display_value = tracer_instance.transform[universal_transform_key](value)
            else:
                display_value = value

            if display_value is not None:
                if tracer_instance.max_argval_len == 0:
                    val_str = "..."
                else:
                    val_str = repr(display_value)
                    if tracer_instance.max_argval_len and len(val_str) > tracer_instance.max_argval_len:
                        val_str = val_str[:tracer_instance.max_argval_len] + "..."
                processed_args.append(f"{name}={val_str}")
            else:
                processed_args.append(f"{name}")
        
        arg_str = ", ".join(processed_args)

    except (ValueError, TypeError):
        arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])

    return arg_str


### OUR CLASSES

class _BaseTracer:
    """A common base class to hold shared initialization logic."""
    def __init__(self, level=logging.DEBUG, trace_chain=False, logger=None,
                 transform=None, max_argval_len=None, duration_fmt: DFMT = None,
                 timing: str = None, timing_fmt: DFMT = DFMT.SINGLE):
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
            duration_fmt (DFMT, optional): Enables and sets the format for
                logging the total execution time. Defaults to None (disabled).
            timing (str, optional): Enables "poor man's profiling". A case-insensitive
                string of characters specifying which clocks to use.
                'm': monotonic, 'h': perf_counter, 'c': process_time, 't': thread_time.
                Defaults to None (disabled).
            timing_fmt (DFMT, optional): The format for displaying timing values.
                Defaults to DFMT.SINGLE.
        """
        self.level = level
        self.trace_chain = trace_chain
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform or {}
        self.max_argval_len = max_argval_len

        self.timing = timing
        self.timing_fmt = timing_fmt
        self.timing_funcs = []
        if self.timing:
            for char in self.timing.lower():
                if char in _TIMERS:
                    self.timing_funcs.append(_TIMERS[char])

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
            chain = tracer_chain.get()
            indent = '    ' * len(chain)

            arg_str = _get_arg_str(func, args, kwargs, self)
            current_call_sig = f"{func.__qualname__}({arg_str})"
            
            # Build the entry log message
            log_entry = f"{indent}--> Calling {current_call_sig}"
            if self.trace_chain and chain:
                chain_str = " <== ".join(reversed(chain))
                log_entry += f"  <== {chain_str}"
            self.logger.log(self.level, log_entry)

            # Update the context for the next level down
            token = tracer_chain.set(chain + [current_call_sig])

            # Start time measurements
            start_times = [f() for f in self.timing_funcs]

            try:
                result = func(*args, **kwargs)

                # Stop time measurements
                end_times = [f() for f in self.timing_funcs]
                timing_block = _get_timing_block(start_times, end_times, self.timing, self.timing_fmt)
 
                # Build the exit log message
                log_exit = f"{timing_block}{indent}<-- Exiting {current_call_sig}, returned: {repr(result)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exit += f"  (pending: {pending_str})"
                self.logger.log(self.level, log_exit)
                
                return result
            except Exception as e:
                # Stop time measurements
                end_times = [f() for f in self.timing_funcs]
                timing_block = _get_timing_block(start_times, end_times, self.timing, self.timing_fmt)

                # Build the exception log message
                log_exc = f"{timing_block}{indent}<!> Exiting {current_call_sig} with exception: {repr(e)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exc += f"  (pending: {pending_str})"
                self.logger.warning(log_exc)
                raise
            finally:
                # Restore the context for the upper level
                tracer_chain.reset(token)
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
            chain = tracer_chain.get()
            indent = '    ' * len(chain)

            arg_str = _get_arg_str(func, args, kwargs, self)
            current_call_sig = f"{func.__qualname__}({arg_str})"

            # Build the entry log message
            log_entry = f"{indent}--> Calling {current_call_sig}"
            if self.trace_chain and chain:
                chain_str = " <== ".join(reversed(chain))
                log_entry += f"  <== {chain_str}"
            self.logger.log(self.level, log_entry)

            token = tracer_chain.set(chain + [current_call_sig])

            # Start time measurements
            start_times = [f() for f in self.timing_funcs]

            try:
                result = await func(*args, **kwargs)

                # Stop time measurements
                end_times = [f() for f in self.timing_funcs]
                timing_block = _get_timing_block(start_times, end_times, self.timing, self.timing_fmt)
                
                # Build the exit log message
                log_exit = f"{timing_block}{indent}<-- Exiting {current_call_sig}, returned: {repr(result)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exit += f"  (pending: {pending_str})"
                self.logger.log(self.level, log_exit)
                
                return result
            except Exception as e:
                # Stop time measurements
                end_times = [f() for f in self.timing_funcs]
                timing_block = _get_timing_block(start_times, end_times, self.timing, self.timing_fmt)

                # Build the exception log message
                log_exc = f"{timing_block}{indent}<!> Exiting {current_call_sig} with exception: {repr(e)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exc += f"  (pending: {pending_str})"
                self.logger.warning(log_exc)
                raise
            finally:
                tracer_chain.reset(token)
        return async_wrapper


no_self = {('*', 'self'): (lambda _:None)}


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
