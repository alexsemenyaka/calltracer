#!/usr/bin/env python3
"""
Examples for the calltracer module
"""

import logging

from calltracer import CallTracer, no_self, stack

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

trace = CallTracer(level=logging.DEBUG)

chtrace = CallTracer(level=logging.DEBUG, trace_chain=True, transform=no_self)

tchtrace = CallTracer(
    level=logging.DEBUG, trace_chain=True, transform=no_self, timing="chm"
)

techtrace = CallTracer(
    level=logging.DEBUG, trace_chain=True, transform=no_self, timing="CHM"
)

idetrace = CallTracer(level=logging.DEBUG, trace_chain=True, ide_support=True)

termtrace = CallTracer(
    level=logging.DEBUG, trace_chain=True, term_support=True, rel_path=False
)


class AdvancedCalculator:  # pylint: disable=too-few-public-methods
    """A calculator to demonstrate tracing."""

    def __init__(self, name):
        self.name = name

    @trace
    def factorial(self, n):
        """Calculates factorial and demonstrates stack tracing"""
        if n == 2:
            logging.info("--- Dumping stack, because n == 2 ---")
            # Call stack() with INFO level to make it stand out in the log
            stack(level=logging.INFO)

        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)


class SecondAdvancedCalculator:  # pylint: disable=too-few-public-methods
    """A copy of the calculator to demonstrate tracing with chaining"""

    def __init__(self, name):
        self.name = name

    @chtrace
    def factorial(self, n):
        """Calculates factorial"""

        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)


class ThirdAdvancedCalculator:  # pylint: disable=too-few-public-methods
    """Another copy of the calculator to demonstrate tracing with chaining and profiling times"""

    def __init__(self, name):
        self.name = name

    @tchtrace
    def factorial(self, n):
        """Calculates factorial"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)


class FourthAdvancedCalculator:  # pylint: disable=too-few-public-methods
    """And another copy of the calculator to demonstrate tracing with chaining and profiling times"""

    def __init__(self, name):
        self.name = name

    @techtrace
    def factorial(self, n):
        """Calculates factorial"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)


@techtrace
def factorial_function(i: int) -> int:
    match i:
        case 0:
            return 1
        case 1:
            return 1
        case 2:
            return 2
        case _:
            return i * factorial_function(i - 1)


@idetrace
def divide_function(i: int) -> int:
    match i:
        case 0:
            return 1
        case 1:
            return 1
        case 2:
            return 2
        case _:
            return divide_function(i // 2 + 1)


@termtrace
def fib_function(i: int) -> int:
    match i:
        case 0:
            return 1
        case 1:
            return 1
        case 2:
            return 2
        case 3:
            return 5
        case _:
            return fib_function(i - 1) + fib_function(i - 2)


calc = AdvancedCalculator("MyCalc")
logging.info("--- Starting recursive call with stack dump ---")
calc.factorial(4)

calc = SecondAdvancedCalculator("MyCalc2")
logging.info("--- Starting recursive call with stack dump and chained tracing---")
calc.factorial(4)

calc = ThirdAdvancedCalculator("MyCalc3")
logging.info(
    "--- Starting recursive call with stack dump, chained tracing and profiling times---"
)
calc.factorial(4)

calc = FourthAdvancedCalculator("MyCalc4")
logging.info(
    "--- Starting recursive call with stack dump, chained tracing and exclusive profiling times---"
)
calc.factorial(4)

logging.info(
    "--- Starting recursive simple function call with chained tracing and exclusive profiling times---"
)
factorial_function(6)

logging.info("--- Starting recursive simple function call with IDE support---")
divide_function(100)

logging.info(
    "--- Starting recursive simple function call with OSC8 support by terminal---"
)
fib_function(7)
