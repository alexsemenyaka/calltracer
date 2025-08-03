#!/usr/bin/env python3
"""
Examples for the calltracer module
"""

import logging

from calltracer import CallTracer, stack

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


trace = CallTracer(level=logging.DEBUG)
chtrace = CallTracer(level=logging.DEBUG, trace_chain=True)

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
        """Calculates factorial and demonstrates stack tracing."""
        if n == 2:
            logging.info("--- Dumping stack, because n == 2 ---")
            # Call stack() with INFO level to make it stand out in the log
            stack(level=logging.INFO)

        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)



calc = AdvancedCalculator("MyCalc")
logging.info("--- Starting recursive call with stack dump ---")
calc.factorial(4)

calc = SecondAdvancedCalculator("MyCalc2")
logging.info("--- Starting recursive call with stack dump and chained tracing---")
calc.factorial(4)
