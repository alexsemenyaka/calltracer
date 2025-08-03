# pytracecall
[![PyPI version](https://img.shields.io/pypi/v/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - License](https://img.shields.io/pypi/l/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![Coverage Status](https://coveralls.io/repos/github/alexsemenyaka/calltracer/badge.svg?branch=main)](https://coveralls.io/github/alexsemenyaka/calltracer?branch=main)
[![CI/CD Status](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml)

# Python Call Tracer Module

A debugging module with decorators (`CallTracer`, `aCallTracer`) and a function (`stack`) for tracing function calls and logging the call stack.

This module provides simple yet powerful tools to help you understand your code's execution flow without the need for a full step-by-step debugger. It is designed to integrate seamlessly with Python's standard `logging` module.

***

## Features

-   **Synchronous and Asynchronous Tracing**: Decorators for both standard (`def`) and asynchronous (`async def`) functions.
-   **Concurrency Safe**: The async tracer (`aCallTracer`) uses `contextvars` to safely trace concurrent tasks without mixing up logs.
-   **Data Flow Visibility**: Logs function arguments and return values.
-   **Recursion Visualization**: Automatically indents log messages to clearly show recursion depth, even in concurrent contexts.
-   **Stack Inspection**: Use the `stack()` function to log the current call stack at any point in your code.
-   **Logging Integration**: Works with the standard `logging` module.

***

## Installation

You can install the package from the Python Package Index (PyPI) using **`pip`**.

```bash
pip install pytracecall
```

To ensure you have the latest version, you can use the `--upgrade` flag:

```bash
pip install --upgrade pytracecall
```

***

## Synchronous Usage

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

Use the `CallTracer` instance as a decorator to trace a synchronous function.

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

## Asynchronous Usage

The `aCallTracer` decorator is designed specifically for `async def` functions and is safe for concurrent execution.

### Async Example

```python
import asyncio
import logging
from calltracer import aCallTracer, stack

# Basic logging configuration (can be done once)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Create an instance of the ASYNCHRONOUS tracer
async_trace = aCallTracer(level=logging.DEBUG)

class AsyncDataFetcher:
    @async_trace
    async def process_item(self, item_id: str, delay: float):
        await asyncio.sleep(delay)
        return f"Processed {item_id}"

async def main():
    fetcher = AsyncDataFetcher()
    print("\n--- Running two tasks concurrently to show task-safety ---")
    results = await asyncio.gather(
        fetcher.process_item(item_id="A", delay=0.2),
        fetcher.process_item(item_id="B", delay=0.1),
    )
    logging.info("Concurrent results: %s", results)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Output (Async)

Note how the logs from Task A and B are interleaved, but the indentation remains correct for each independent task.

```
--- Running two tasks concurrently to show task-safety ---
22:01:05 - calltracer.async_tracer - DEBUG - --> Calling AsyncDataFetcher.process_item(<...>, item_id='A', delay=0.2)
22:01:05 - calltracer.async_tracer - DEBUG - --> Calling AsyncDataFetcher.process_item(<...>, item_id='B', delay=0.1)
22:01:05 - calltracer.async_tracer - DEBUG - <-- Exiting AsyncDataFetcher.process_item, returned: 'Processed B'
22:01:05 - calltracer.async_tracer - DEBUG - <-- Exiting AsyncDataFetcher.process_item, returned: 'Processed A'
22:01:05 - root - INFO - Concurrent results: ['Processed A', 'Processed B']
```

***

## API Reference

### `CallTracer` Class (Synchronous)

A factory for creating decorators that trace synchronous (`def`) function/method calls.

#### Initialization

```python
from calltracer import CallTracer
trace = CallTracer(level=logging.INFO)
```

### `aCallTracer` Class (Asynchronous)

A factory for creating decorators that trace asynchronous (`async def`) function/method calls. This tracer is task-safe for concurrent code using `contextvars`.

#### Initialization

```python
from calltracer import aCallTracer
async_trace = aCallTracer(level=logging.INFO)
```

**Parameters for both classes:**

-   **`level`** (`int`, optional): The logging level for trace messages. Defaults to `logging.DEBUG`.
-   **`trace_chain`** (`bool`, optional): If True, accumulates and logs the call chain. Default to False.
-   **`logger`** (`logging.Logger`, optional): The logger instance to use. Defaults to the internal module logger.

### `stack()` Function

Logs the current call stack to the specified logger. This function works in both synchronous and asynchronous code.

#### Signature

```python
stack(level=logging.DEBUG, logger=None, limit=None, start=0)
```

-   **`level`** (`int`, optional): The logging level for the message. Defaults to `logging.DEBUG`.
-   **`logger`** (`logging.Logger`, optional): The logger instance to use. Defaults to the internal module logger.
-   **`limit`** (`int`, optional): The maximum number of frames to display. Defaults to `None` (all frames).
-   **`start`** (`int`, optional): The offset of the first frame to display. Defaults to `0`.
