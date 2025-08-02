import logging
import functools
import inspect

# Define a logger for the entire module.
tracer_logger = logging.getLogger(__name__)


class CallTracer:
    """A factory for creating decorators that trace function/method calls.

    This class, when instantiated, creates a decorator that can be applied to any
    function or method to log its entry, exit, arguments, and return value.

    Example:
        trace = CallTracer(level=logging.INFO)

        @trace
        def my_function(x, y):
            return x + y
    """
    _indent_level = 0

    def __init__(self, level=logging.DEBUG, logger=tracer_logger):
        """Initializes the tracer factory.

        Args:
            level (int): The logging level to use for trace messages (e.g., logging.DEBUG).
            logger (logging.Logger): The logger instance to use.
        """
        self.level = level
        self.logger = logger

    def __call__(self, func):
        """Makes the instance callable and returns the actual decorator wrapper.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function that includes tracing logic.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            indent = '    ' * CallTracer._indent_level
            func_name = func.__qualname__
            arg_str = ", ".join(
                [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            )

            self.logger.log(self.level, "%s--> Calling %s(%s)", indent, func_name, arg_str)

            CallTracer._indent_level += 1
            try:
                result = func(*args, **kwargs)
                self.logger.log(self.level, "%s<-- Exiting %s, returned: %s", indent, func_name, repr(result))
                return result
            except Exception as e:
                self.logger.warning("%s<!> Exiting %s with exception: %s", indent, func_name, repr(e))
                raise
            finally:
                CallTracer._indent_level -= 1
        return wrapper


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

    # The frame that called stack()
    caller_frame = frames[1]
    caller_file, caller_line, caller_func = caller_frame[1:4]

    logger.log(level, "Stack trace at %s:%d in %s():", caller_file, caller_line, caller_func)

    # Determine the slice range to display
    begin = start + 2
    end = min(begin + limit, len(frames)) if limit else len(frames)

    # Log the stack frames
    for frame_info in frames[begin:end]:
        file, line, func, _, _, _ = frame_info
        logger.log(level, "  ↳ Called from: %s, line %d, in %s", file, line, func)
