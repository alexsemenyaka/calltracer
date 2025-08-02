#!/usr/bin/env python3
"""
Examples for the calltracer module
"""

import logging

from calltracer import CallTracer, stack

# 1. Configure the logging system
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# 2. Create an instance of the tracer
trace = CallTracer(level=logging.DEBUG)


class AdvancedCalculator:  # pylint: disable=too-few-public-methods
    """A calculator to demonstrate tracing."""

    def __init__(self, name):
        self.name = name

    @trace
    def factorial(self, n):
        """Calculates factorial and demonstrates stack tracing."""
        # 3. Use stack() for point-in-time analysis
        if n == 2:
            logging.info("--- Dumping stack, because n == 2 ---")
            # Call stack() with INFO level to make it stand out in the log
            stack(level=logging.INFO)

        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n - 1)


# 4. Run the code
calc = AdvancedCalculator("MyCalc")

logging.info("--- Starting recursive call with stack dump ---")
calc.factorial(4)
