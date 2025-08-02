# calltracer
[![PyPI version](https://img.shields.io/pypi/v/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - License](https://img.shields.io/pypi/l/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![Coverage Status](https://coveralls.io/repos/github/alexsemenyaka/calltracer/badge.svg?branch=main)](https://coveralls.io/github/alexsemenyaka/calltracer?branch=main)
[![CI/CD Status](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml)

# Python Call Tracer Module

A debugging module with a decorator (`CallTracer`) for tracing function calls and a function (`stack`) for logging the call stack.

This module provides simple yet powerful tools to help you understand your code's execution flow without the need for a full step-by-step debugger. It is designed to integrate seamlessly with Python's standard `logging` module.

***

## Features

-   **Function Call Tracing**: Use the `@trace` decorator to automatically log when a function is entered and exited.
-   **Data Flow Visibility**: Logs function arguments and return values to see how data flows through your application.
-   **Recursion Visualization**: Automatically indents log messages to clearly show recursion depth.
-   **Stack Inspection**: Use the `stack()` function to log the current call stack at any point in your code.
-   **Logging Integration**: Works with the standard `logging` module, allowing for flexible configuration of output and levels.

***

## Installation

You can install the package from the Python Package Index (PyPI) using **`pip`**.

```bash
pip install calltracer
```

To ensure you have the latest version, you can use the `--upgrade` flag:

```bash
pip install --upgrade calltracer
```

***

## Usage

First, ensure you configure Python's `logging` module to see the output.

### Basic Configuration

```python
import logging

# Configure logging to display DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
```

### Basic Tracing

Use the `CallTracer` instance as a decorator to trace a function.

```python
from calltracer import CallTracer

trace = CallTracer()

@trace
def add(x, y):
    return x + y

add(10, 5)
```

**Output:**

```
21:15:10 - DEBUG - --> Calling add(10, 5)
21:15:10 - DEBUG - <-- Exiting add, returned: 15
```

***

## API Reference

### `CallTracer` Class

A factory for creating decorators that trace function/method calls. This class, when instantiated, creates a decorator that can be applied to any function or method to log its entry, exit, arguments, and return value.

#### Initialization

```python
from calltracer import CallTracer
import logging

# Create a tracer that logs at the INFO level
trace_info = CallTracer(level=logging.INFO)
```

-   **`level`** (`int`, optional): The logging level to use for trace messages. Defaults to `logging.DEBUG`.
-   **`logger`** (`logging.Logger`, optional): The logger instance to use. Defaults to the internal module logger.

### `stack()` Function

Logs the current call stack to the specified logger. This function creates a "snapshot" of how the code reached a certain point, which is useful for point-in-time debugging.

#### Signature

```python
stack(level=logging.DEBUG, logger=tracer_logger, limit=None, start=0)
```

-   **`level`** (`int`, optional): The logging level for the message. Defaults to `logging.DEBUG`.
-   **`logger`** (`logging.Logger`, optional): The logger instance to use. Defaults to the internal module logger.
-   **`limit`** (`int`, optional): The maximum number of frames to display. Defaults to `None` (all frames).
-   **`start`** (`int`, optional): The offset of the first frame to display. Defaults to `0`.

***

## Advanced Example

This example demonstrates using both the `@trace` decorator and the `stack()` function to debug a recursive function.

```python
import logging
from calltracer import CallTracer, stack

# Basic logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Create a tracer instance
trace = CallTracer()

class AdvancedCalculator:
    @trace
    def factorial(self, n):
        """Calculates factorial and demonstrates stack tracing."""
        # Use stack() for point-in-time analysis when n hits 2
        if n == 2:
            logging.info("--- Dumping stack, because n == 2 ---")
            # Call stack() with INFO level to make it stand out
            stack(level=logging.INFO)

        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        else:
            return n * self.factorial(n - 1)

# Run the code
calc = AdvancedCalculator()
logging.info("--- Starting recursive call with stack dump ---")
calc.factorial(4)
```

### Example Output

The output clearly shows the trace of `factorial` calls, interrupted by the stack dump when `n` is `2`.

```
INFO - --- Starting recursive call with stack dump ---
DEBUG - --> Calling AdvancedCalculator.factorial(<__main__.AdvancedCalculator object at ...>, 4)
DEBUG -     --> Calling AdvancedCalculator.factorial(<__main__.AdvancedCalculator object at ...>, 3)
DEBUG -         --> Calling AdvancedCalculator.factorial(<__main__.AdvancedCalculator object at ...>, 2)
INFO - --- Dumping stack, because n == 2 ---
INFO - Stack trace at /path/to/main.py:18 in factorial():
INFO -   ↳ Called from: /path/to/main.py:25, in factorial
INFO -   ↳ Called from: /path/to/main.py:25, in factorial
INFO -   ↳ Called from: /path/to/main.py:31, in <module>
DEBUG -             --> Calling AdvancedCalculator.factorial(<__main__.AdvancedCalculator object at ...>, 1)
DEBUG -                 --> Calling AdvancedCalculator.factorial(<__main__.AdvancedCalculator object at ...>, 0)
DEBUG -                 <-- Exiting AdvancedCalculator.factorial, returned: 1
DEBUG -             <-- Exiting AdvancedCalculator.factorial, returned: 1
DEBUG -         <-- Exiting AdvancedCalculator.factorial, returned: 2
DEBUG -     <-- Exiting AdvancedCalculator.factorial, returned: 6
DEBUG - <-- Exiting AdvancedCalculator.factorial, returned: 24
```
