# pytracecall: A Zero-Dependency Python Tracer
[![PyPI version](https://img.shields.io/pypi/v/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![PyPI - License](https://img.shields.io/pypi/l/pytracecall.svg)](https://pypi.org/project/pytracecall/)
[![Coverage Status](https://coveralls.io/repos/github/alexsemenyaka/calltracer/badge.svg?branch=main)](https://coveralls.io/github/alexsemenyaka/calltracer?branch=main)
[![CI/CD Status](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsemenyaka/calltracer/actions/workflows/ci.yml)

`pytracecall` is a lightweight, zero-dependency Python debugging toolkit designed to provide clear insight into your code's execution flow. Whether you're untangling complex recursion, visualizing concurrent `asyncio` tasks, or simply want to see the journey of your data, `pytracecall` offers a simple and powerful solution without the overhead of a full debugger.

It integrates seamlessly with Python's built-in `logging` module, making it a natural fit for any project.

***

## Key Features

-    **Sync & Async Support**: Dedicated, purpose-built decorators for both standard (`def`) and asynchronous (`async def`) functions.
-    **Optional Call Chain Tracing**: Go beyond simple indentation and see the complete call stack for every traced function call.
-    **Concurrency-Safe**: The async tracer is built with `contextvars` to safely trace hundreds of concurrent tasks without mixing up their logs.
-    **Point-in-Time Stack Inspection**: A simple `stack()` function to dump the current call stack at any point in your code.
-    **Lightweight & Zero-Dependency**: Pure Python with no external libraries required, ensuring easy integration.

***

## Installation

You can install the package from the Python Package Index (PyPI) using **`pip`**.

```bash
pip install pytracecall
```

***

## Quick Start

Get a feel for `pytracecall` in 30 seconds.

```python
import logging
from calltracer import CallTracer

# Basic logging config
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create a tracer instance and decorate your function
trace = CallTracer()

@trace
def greet(name):
    print(f"Hello, {name}!")

greet("World")
```

**Output:**

```
--> Calling greet('World')
Hello, World!
<-- Exiting greet('World'), returned: None
```

***

## Core Concepts & Examples

`pytracecall` provides two main decorator classes: `CallTracer` for synchronous code and `aCallTracer` for asynchronous code.

### Synchronous Tracing with `CallTracer`

Use `CallTracer` to debug standard Python functions, especially useful for understanding recursion.

```python
import logging
from calltracer import CallTracer, chtrace

# A tracer with the optional call chain feature enabled
chtrace = CallTracer(level=logging.DEBUG, trace_chain=True)

class AdvancedCalculator:
    @chtrace
    def factorial(self, n):
        if n == 0:
            return 1
        return n * self.factorial(n - 1)

calc = AdvancedCalculator("MyCalc")
calc.factorial(4)
```

The `trace_chain=True` parameter provides incredibly detailed logs, showing not just the depth but the entire history of traced calls for each step.

**Output with Chained Tracing:**

```
--> Calling SecondAdvancedCalculator.factorial(..., 4)
    --> Calling SecondAdvancedCalculator.factorial(..., 3)  <== SecondAdvancedCalculator.factorial(..., 4)
        --> Calling SecondAdvancedCalculator.factorial(..., 2)  <== SecondAdvancedCalculator.factorial(..., 3) <== SecondAdvancedCalculator.factorial(..., 4)
            --> Calling SecondAdvancedCalculator.factorial(..., 1)  <== SecondAdvancedCalculator.factorial(..., 2) <== ...
                --> Calling SecondAdvancedCalculator.factorial(..., 0)  <== SecondAdvancedCalculator.factorial(..., 1) <== ...
                <-- Exiting SecondAdvancedCalculator.factorial(..., 0), returned: 1  (pending: SecondAdvancedCalculator.factorial(..., 1) <== ...)
            <-- Exiting SecondAdvancedCalculator.factorial(..., 1), returned: 1  (pending: SecondAdvancedCalculator.factorial(..., 2) <== ...)
        <-- Exiting SecondAdvancedCalculator.factorial(..., 2), returned: 2  (pending: SecondAdvancedCalculator.factorial(..., 3) <== ...)
    <-- Exiting SecondAdvancedCalculator.factorial(..., 3), returned: 6  (pending: SecondAdvancedCalculator.factorial(..., 4))
<-- Exiting SecondAdvancedCalculator.factorial(..., 4), returned: 24
```

### Asynchronous Tracing with `aCallTracer`

Use `aCallTracer` to trace `async def` functions. It uses `contextvars` to ensure that even thousands of interleaved concurrent tasks are traced independently and correctly.

```python
import asyncio
import logging
from calltracer import aCallTracer

async_chtrace = aCallTracer(level=logging.DEBUG, trace_chain=True)

class AsyncDataFetcher:
    @async_chtrace
    async def process_item_upper(self, item_id: str, delay: float):
        return await self.process_item_medium(item_id, delay)

    @async_chtrace
    async def process_item_medium(self, item_id: str, delay: float):
        return await self.process_item(item_id, delay)

    @async_chtrace
    async def process_item(self, item_id: str, delay: float):
        await asyncio.sleep(delay)
        return f"Processed {item_id}"

async def main():
    fetcher = AsyncDataFetcher()
    await asyncio.gather(
        fetcher.process_item_upper(item_id="A", delay=0.2),
        fetcher.process_item_upper(item_id="B", delay=0.1),
    )

asyncio.run(main())
```

**Output with Concurrent Chained Tracing:**

The output clearly shows two tasks (for item 'A' and 'B') starting concurrently. Each task maintains its own separate call chain, demonstrating the power of `contextvars`.

```
--> Calling AsyncSecondDataFetcher.process_item_upper(..., item_id='A', delay=0.2)
    --> Calling AsyncSecondDataFetcher.process_item_medium(..., 'A', 0.2)  <== AsyncSecondDataFetcher.process_item_upper(...)
        --> Calling AsyncSecondDataFetcher.process_item(..., 'A', 0.2)  <== AsyncSecondDataFetcher.process_item_medium(...) <== ...
--> Calling AsyncSecondDataFetcher.process_item_upper(..., item_id='B', delay=0.1)
    --> Calling AsyncSecondDataFetcher.process_item_medium(..., 'B', 0.1)  <== AsyncSecondDataFetcher.process_item_upper(...)
        --> Calling AsyncSecondDataFetcher.process_item(..., 'B', 0.1)  <== AsyncSecondDataFetcher.process_item_medium(...) <== ...
<-- Exiting AsyncSecondDataFetcher.process_item(..., 'B', 0.1), returned: 'Processed B'  (pending: AsyncSecondDataFetcher.process_item_medium(...) <== ...)
<-- Exiting AsyncSecondDataFetcher.process_item_medium(..., 'B', 0.1), returned: 'Processed B'  (pending: AsyncSecondDataFetcher.process_item_upper(...))
<-- Exiting AsyncSecondDataFetcher.process_item_upper(..., item_id='B', delay=0.1), returned: 'Processed B'
<-- Exiting AsyncSecondDataFetcher.process_item(..., 'A', 0.2), returned: 'Processed A'  (pending: AsyncSecondDataFetcher.process_item_medium(...) <== ...)
<-- Exiting AsyncSecondDataFetcher.process_item_medium(..., 'A', 0.2), returned: 'Processed A'  (pending: AsyncSecondDataFetcher.process_item_upper(...))
<-- Exiting AsyncSecondDataFetcher.process_item_upper(..., item_id='A', delay=0.2), returned: 'Processed A'
```

***

## API Reference

### `CallTracer(level=DEBUG, trace_chain=False, logger=None)`

Creates a decorator for tracing **synchronous** (`def`) functions.

### `aCallTracer(level=DEBUG, trace_chain=False, logger=None)`

Creates a decorator for tracing **asynchronous** (`async def`) functions.

### `stack(level=DEBUG, logger=None, limit=None, start=0)`

A function that logs the current Python call stack at the point it is called. Works in both sync and async code.
