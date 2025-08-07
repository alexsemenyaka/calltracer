[![PyPI version](https://img.shields.io/pypi/v/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - License](https://img.shields.io/pypi/l/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![Coverage Status](https://coveralls.io/repos/github/alexsemenyaka/calltracer/badge.svg?branch=main)](https://coveralls.io/github/alexsemenyaka/calltracer?branch=main)
[![CI/CD Status](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml)

A powerful, flexible, and user-friendly debugging module for tracing function calls in Python.

`pytracecall` provides simple yet powerful tools to help you understand your code's execution flow without a full step-by-step debugger. It is designed to integrate seamlessly with Python's standard `logging` module and can produce output for human analysis, IDEs, and automated systems.

---

## Why PyTraceCall?

* **Unmatched Insight, Zero Intrusion**: Get deep insights into your code's execution flow, arguments, return values, and performance without modifying your core logic. The decorator pattern keeps your code clean and readable.
* **Debug Concurrency with Confidence**: Built from the ground up with `contextvars`, `pytracecall` provides clear, isolated traces for complex `asyncio` applications, eliminating the guesswork of concurrent execution flows.
* **From Quick Glance to Deep Analysis**: Whether you need a quick print-style debug, a detailed performance profile with exclusive timings, or structured JSON for automated analysis, the flexible API scales to your needs.
* **Highly Configurable & User-Friendly**: Fine-tune everything from output colors and argument visibility to conditional tracing triggers. The power is in your hands.
* **A Joy to Use**: With features like clickable IDE/terminal integration and beautiful `rich` tree views, debugging stops being a chore and becomes an insightful, and even enjoyable, experience.

---

## Features

-   **Synchronous and Asynchronous Tracing**: Decorators for both standard (`def`) and asynchronous (`async def`) functions.
-   **Rich Interactive Output**: Optional integration with the `rich` library to render call stacks as beautiful, dynamic trees.
-   **IDE & Terminal Integration**: Generates log entries that are clickable in modern IDEs (VSCode, PyCharm) and terminals (iTerm2 with OSC 8 support), taking you directly to the source code line.
-   **Advanced Performance Profiling**: Measure execution time with multiple system clocks. Differentiate between **inclusive** time (total) and **exclusive** time (function's own work, excluding children).
-   **Conditional Tracing**: Define custom rules to activate tracing only for specific calls, preventing log spam and focusing on what matters.
-   **Argument & Return Value Control**: Mask sensitive data (like passwords), truncate long values, and even hide arguments (like `self`) from the output.
-   **Structured JSON Output**: Log trace events as JSON objects for easy parsing, filtering, and analysis by automated systems.
-   **Runtime Control**: Programmatically enable or disable any tracer instance on the fly.
-   **Concurrency Safe**: Uses `contextvars` to safely trace concurrent tasks without mixing up call chains.

---

## Installation

You can install the package from PyPI using **`pip`**.

```bash
pip install pytracecall
```

To enable the optional `rich` integration for beautiful tree-like logging, install the `rich` extra:

```bash
pip install "pytracecall[rich]"
```

---

## Usage Examples

### Basic Synchronous Tracing

First, ensure you configure Python's `logging` module to see the output.

```python
import logging
from calltracer import CallTracer

# Configure logging to display DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

trace = CallTracer()

@trace
def add(x, y):
    return x + y

add(10, 5)
```

**Output:**
```
21:15:10 - --> Calling add(x=10, y=5)
21:15:10 - <-- Exiting add(x=10, y=5), returned: 15
```

### Advanced Features Showcase

The true power of `pytracecall` lies in its rich configuration.

#### Rich Interactive Trees

For the most intuitive visualization, use the `RichPyTraceHandler`.

**Code (`rex.py`):**
```python
import logging
from calltracer import CallTracer, DFMT, RichPyTraceHandler

# 1. Configure a logger to use the Rich handler exclusively
log = logging.getLogger("rich_demo")
log.setLevel(logging.DEBUG)
log.handlers = [RichPyTraceHandler(overwrite=False)] # `overwrite=False` for append-only tree
log.propagate = False

# 2. Configure the tracer to output JSON for the handler to consume
trace = CallTracer(logger=log, output="json", timing="Mh")

@trace
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib(5)
```

**Output:**

![Rich Tree Output](http://googleusercontent.com/file_content/52)

#### Call Chain Tracing (`trace_chain`)

Set `trace_chain=True` to see the full context for every call.

```python
trace_with_chain = CallTracer(trace_chain=True)
```

**Output:**

![Trace Chain Output](http://googleusercontent.com/file_content/53)

#### Performance Profiling (`timing`)

Measure performance using different clocks. Use **lowercase** for inclusive time and **uppercase** for exclusive time.

```python
# M: Exclusive monotonic time, h: Inclusive perf_counter time
profile_trace = CallTracer(timing="Mh", timing_fmt=DFMT.SINGLE)
```

#### IDE & Terminal Integration

Make your logs clickable! `ide_support` creates links for IDEs, while `term_support` uses OSC 8 for modern terminals.

```python
# For clickable links in PyCharm/VSCode
ide_trace = CallTracer(ide_support=True)

# For Ctrl/Cmd-Click in iTerm2 and other modern terminals
term_trace = CallTracer(term_support=True)
```

**Output in iTerm2 (with `term_support=True`):** The function signature becomes a clickable link that opens the file at the correct line in your editor.

#### Asynchronous Tracing

`aCallTracer` handles `async` functions and concurrency flawlessly, keeping call chains isolated.

**Output:**

![Async Tracing Output](http://googleusercontent.com/file_content/54)

---

## Full API Reference

The `CallTracer` and `aCallTracer` classes share the same rich set of initialization parameters, inherited from a base class.

### `CallTracer` / `aCallTracer` Parameters

```python
__init__(self,
         level=logging.DEBUG,
         trace_chain=False,
         logger=None,
         transform=None,
         max_argval_len=None,
         return_transform: Optional[Callable] = None,
         max_return_len: Optional[int] = None,
         condition: Optional[Callable] = None,
         timing: str = None,
         timing_fmt: DFMT = DFMT.SINGLE,
         output: str = 'text',
         ide_support: bool = False,
         term_support: bool = False,
         rel_path: bool = True)
```

-   **`level`** (`int`): The logging level for trace messages.
-   **`trace_chain`** (`bool`): If `True`, logs the full call chain for each event.
-   **`logger`** (`logging.Logger`): A custom logger instance to use.
-   **`transform`** (`dict`): A dictionary of callbacks to transform/hide argument values. Keys are `(func_qualname, arg_name)` tuples. A wildcard `('*', arg_name)` can be used. If a callback returns `None`, only the argument name is printed.
-   **`max_argval_len`** (`int`): Maximum length for the string representation of argument values.
-   **`return_transform`** (`Callable`): A function to transform the return value before logging.
-   **`max_return_len`** (`int`): Maximum length for the string representation of the return value.
-   **`condition`** (`Callable`): A function `(func_name, *args, **kwargs) -> bool` that determines if tracing should be active for a call. If it returns `False`, this call and all nested calls are skipped.
-   **`timing`** (`str`): Enables [poor mens'] profiling. A string of characters specifying clocks to use (`m`onotonic, `h`igh-perf, `c`pu, `t`hread). **Lowercase** measures inclusive (total) time. **Uppercase** measures exclusive time (total time minus decorated child calls).
-   **`timing_fmt`** (`DFMT`): The display format for timing values (`DFMT.NANO`, `DFMT.MICRO`, `DFMT.SEC`, `DFMT.SINGLE`, `DFMT.HUMAN`). See docstrings for details.
-   **`output`** (`str`): The output format. `'text'` (default) for human-readable logs or `'json'` for structured logging.
-   **`ide_support`** (`bool`): If `True`, formats text logs to be clickable in IDEs (e.g., PyCharm, VSCode).
-   **`term_support`** (`bool`): If `True`, formats text logs with OSC 8 hyperlinks for modern terminals.
-   **`rel_path`** (`bool`): If `True`, uses relative paths for `ide_support` and `term_support`.

Methods:
-   **`enable()` / `disable()`**: Each tracer instance has these methods to control tracing at runtime.

### `RichPyTraceHandler` Parameters

The handler for beautiful `rich` tree output.

```python
__init__(self,
         overwrite: bool = False,
         color_enter: str = "green",
         color_exit: str = "bold blue",
         color_exception: str = "bold red",
         color_timing: str = "yellow")
```
-   **`overwrite`** (`bool`): If `False` (default), creates an append-only tree showing both enter and exit events. If `True`, uses a `Live` animated display to overwrite enter nodes with exit information.
-   **`color_*`** (`str`): Rich markup strings to customize the output colors.

### `stack()` Function

```python
stack(level=logging.DEBUG, logger=None, limit=None, start=0)
```
Logs the current call stack.

-   **`level`** (`int`): The logging level for the stack trace message.
-   **`logger`** (`logging.Logger`): The logger to use.
-   **`limit`** (`int`): Maximum number of frames to show.
-   **`start`** (`int`): Frame offset to start from.
