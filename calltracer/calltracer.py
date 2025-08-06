"""
calltracer: A debugging module with a decorator (CallTracer) for
tracing function calls and a function (stack) for logging the current call stack.
"""
import asyncio
import functools
import inspect
import logging
import contextvars


# Define a logger for the entire module.
tracer_logger = logging.getLogger(__name__)

# A single context variable will hold the list of calls (the chain).
# It works safely for both sync and async code.
tracer_chain = contextvars.ContextVar('tracer_chain', default=[])


class CallTracer:  # pylint: disable=too-few-public-methods
    """A factory for creating decorators that trace SYNCHRONOUS function/method calls.

    This class, when instantiated, creates a decorator that can be applied to any
    function or method to log its entry, exit, arguments, and return value.

    Example:
        trace = CallTracer(level=logging.INFO)

        @trace
        def my_function(x, y):
            return x + y
    """

    def __init__(self, level=logging.DEBUG, trace_chain=False, logger=None, transform=None, max_argval_len=None):
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
                If returns None, only the argargument name will be printed
                Example: {('MyClass.login', 'password'): lambda p: '***',
                          ('MyClass.method', 'self'): lambda s: None}
            max_argval_len (int, optional): Maximum length for the string
                representation of argument values in logs.
                - If None (default), no truncation is performed.
                - If 0, argument values are hidden (displayed as '...').
                - If > 0, the string representation is truncated to this length.
        """
        self.level = level
        self.trace_chain = trace_chain
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform or {}
        self.max_argval_len = max_argval_len

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
            
            func_name = func.__qualname__

            try:
                # Bind the passed args/kwargs to the function signature to get the names of all arguments
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
    
                processed_args = []
                for name, value in bound_args.arguments.items():
                    # if there is a callback for transforming this argument, apply it
                    transform_key = (func_name, name)
                    if transform_key in self.transform:
                        display_value = self.transform[transform_key](value)
                    else:
                        display_value = value

                    if display_value is not None:
                        # format the value into a string taking into account the max_argval_len parameter
                        if self.max_argval_len == 0:
                            val_str = "..."
                        else:
                            val_str = repr(display_value)
                            if self.max_argval_len and len(val_str) > self.max_argval_len:
                                val_str = val_str[:self.max_argval_len] + "..."
        
                        processed_args.append(f"{name}={val_str}")
                    else:
                        processed_args.append(f"{name}")

        
                arg_str = ", ".join(processed_args)

            except (ValueError, TypeError):
                # Emergency option for objects whose signature cannot be analyzed (e.g., some built-in functions).
                # In this case, transformation and truncation are not applied.
                arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])

            current_call_sig = f"{func_name}({arg_str})"

            # Build the entry log message
            log_entry = f"{indent}--> Calling {current_call_sig}"
            if self.trace_chain and chain:
                chain_str = " <== ".join(reversed(chain))
                log_entry += f"  <== {chain_str}"
            self.logger.log(self.level, log_entry)

            # Update the context for the next level down
            token = tracer_chain.set(chain + [current_call_sig])

            try:
                result = func(*args, **kwargs)
                
                # Build the exit log message
                log_exit = f"{indent}<-- Exiting {current_call_sig}, returned: {repr(result)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exit += f"  (pending: {pending_str})"
                self.logger.log(self.level, log_exit)
                
                return result
            except Exception as e:
                # Build the exception log message
                log_exc = f"{indent}<!> Exiting {current_call_sig} with exception: {repr(e)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exc += f"  (pending: {pending_str})"
                self.logger.warning(log_exc)
                raise
            finally:
                # Restore the context for the upper level
                tracer_chain.reset(token)
        return sync_wrapper

class aCallTracer:  # pylint: disable=too-few-public-methods
    """A factory for creating decorators that trace ASYNCHRONOUS function calls."""
    
    def __init__(self, level=logging.DEBUG, trace_chain=False, logger=None, transform=None, max_argval_len=None):
        """
        Initializes the factory.
        
        Args:
            level (int): The logging level for trace messages.
            trace_chain (bool): If True, accumulates and logs the call chain.
            logger (logging.Logger): The logger instance to use.
            transform (dict, optional): A dictionary of callbacks to transform
                argument values before logging. The key is a tuple of
                (func_qualname, arg_name), and the value is a callable that
                If returns None, only the argargument name will be printed
                receives the argument's value and returns a new value for display.
                Example: {('MyClass.login', 'password'): lambda p: '***',
                          ('MyClass.method', 'self'): lambda s: None}
            max_argval_len (int, optional): Maximum length for the string
                representation of argument values in logs.
                - If None (default), no truncation is performed.
                - If 0, argument values are hidden (displayed as '...').
                - If > 0, the string representation is truncated to this length.
        """
        self.level = level
        self.trace_chain = trace_chain
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform or {}
        self.max_argval_len = max_argval_len

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

            func_name = func.__qualname__
            try:
                # Bind the passed args/kwargs to the function signature to get the names of all arguments
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()

                processed_args = []
                for name, value in bound_args.arguments.items():
                    # if there is a callback for transforming this argument, apply it
                    transform_key = (func_name, name)
                    if transform_key in self.transform:
                        display_value = self.transform[transform_key](value)
                    else:
                        display_value = value

                    if display_value is not None:
                            # format the value into a string taking into account the max_argval_len parameter
                        if self.max_argval_len == 0:
                            val_str = "..."
                        else:
                            val_str = repr(display_value)
                            if self.max_argval_len and len(val_str) > self.max_argval_len:
                                val_str = val_str[:self.max_argval_len] + "..."

                        processed_args.append(f"{name}={val_str}")
                    else:
                        processed_args.append(f"{name}")
        
                arg_str = ", ".join(processed_args)

            except (ValueError, TypeError):
                # Emergency option for objects whose signature cannot be analyzed (e.g., some built-in functions).
                # In this case, transformation and truncation are not applied.
                arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])

            current_call_sig = f"{func_name}({arg_str})"

            # Build the entry log message
            log_entry = f"{indent}--> Calling {current_call_sig}"
            if self.trace_chain and chain:
                chain_str = " <== ".join(reversed(chain))
                log_entry += f"  <== {chain_str}"
            self.logger.log(self.level, log_entry)

            token = tracer_chain.set(chain + [current_call_sig])

            try:
                result = await func(*args, **kwargs)
                
                # Build the exit log message
                log_exit = f"{indent}<-- Exiting {current_call_sig}, returned: {repr(result)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exit += f"  (pending: {pending_str})"
                self.logger.log(self.level, log_exit)
                
                return result
            except Exception as e:
                # Build the exception log message
                log_exc = f"{indent}<!> Exiting {current_call_sig} with exception: {repr(e)}"
                if self.trace_chain and chain:
                    pending_str = " <== ".join(reversed(chain))
                    log_exc += f"  (pending: {pending_str})"
                self.logger.warning(log_exc)
                raise
            finally:
                tracer_chain.reset(token)
        return async_wrapper


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
