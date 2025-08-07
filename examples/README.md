# pytracecall: A Zero-Dependency Python Tracer
[![PyPI version](https://img.shields.io/pypi/v/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - License](https://img.shields.io/pypi/l/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![Coverage Status](https://coveralls.io/repos/github/alexsemenyaka/calltracer/badge.svg?branch=main)](https://coveralls.io/github/alexsemenyaka/calltracer?branch=main)
[![CI/CD Status](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml)

# Examples for `pytracecall`

This directory contains practical examples demonstrating the various features of the `pytracecall` library. Each file is designed to be run independently and showcases a specific set of capabilities.

To run the examples that use the `rich` library, you must first install the optional dependency:
```bash
pip install "pytracecall[rich]"
```

---

## `ex.py` - Core Features & Text Output

This file is a showcase of all the core features of the synchronous `CallTracer` using the standard text-based logger. It's a great starting point to understand the raw data and formatting options provided by the module.

### Key Concepts Shown:

* **Basic Tracing**: Simple decoration of a recursive function.
* **Call Chain (`trace_chain=True`)**: See the full call stack in every log message for deep context.
* **Performance Profiling (`timing`)**: Demonstrates logging of inclusive (`'chm'`) and exclusive (`'CHM'`) execution times using different clocks.
* **IDE Integration (`ide_support=True`)**: Shows how log output can be formatted to create clickable links in IDEs like VSCode and PyCharm.
* **Terminal Hyperlinks (`term_support=True`)**: Demonstrates OSC 8 hyperlinks for modern terminals like iTerm2, allowing you to `Cmd-Click` a function call to jump to its source code.
* **Stack Inspection**: Shows the use of the `stack()` function to dump the call stack at a specific point.

### How to Run:

```bash
python3 ex.py
```

---

## `aex.py` - Asynchronous & Concurrent Tracing

This file focuses on `aCallTracer` and its critical ability to safely trace concurrent asynchronous code. It is the best example to understand the power of using `contextvars` for isolating execution contexts.

### Key Concepts Shown:

* **Tracing `async def` Functions**: The basic usage of `aCallTracer`.
* **Concurrency Safety**: This is the main highlight. The example uses `asyncio.gather` to run two processing tasks concurrently. The log output clearly shows that even though the tasks are interleaved, their call chains (`trace_chain`) and indentation are kept perfectly separate.

### How to Run:

```bash
python3 aex.py
```

### When is this useful?

This demonstrates the most important feature for modern Python applications. If you are debugging a web server, a data pipeline, or any `asyncio`-based program with multiple simultaneous operations, this proves that `pytracecall` can trace them reliably without mixing up their logs.

---

## `rex.py` - Rich Interactive Trees

This file showcases the advanced visual debugging capabilities provided by the `RichPyTraceHandler`, which is available as an optional feature.

### Key Concepts Shown:

* **JSON Output (`output='json'`)**: This example demonstrates the recommended pattern for rich logging. The tracer is configured to produce structured JSON data, which the `RichPyTraceHandler` then consumes to build the visual tree. This decouples data generation from presentation.
* **Append-Only Mode (`overwrite=False`)**: The default `rich` mode. It creates a beautiful, downward-growing tree where both the "enter" (➡️) and "exit" (⬅️) for each call are preserved. This is excellent for post-mortem analysis of a complete function run.
* **Live Overwrite Mode (`overwrite=True`)**: This mode uses `rich.live.Live` to create a dynamic display where the "enter" node for a function is overwritten by its corresponding "exit" node. This provides a more compact, real-time view that is ideal for monitoring longer-running processes.

### How to Run:

It is recommended to run this example as a module using the `-m` flag from the project's root directory. This ensures Python's import system can correctly locate the `calltracer` package.

```bash
# From the project's root directory
python3 -m examples.rex
```
