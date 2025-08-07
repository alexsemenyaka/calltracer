# tests/test_ct.py

"""calltracer synchronous tests"""

import logging

import pytest

from calltracer import DFMT, CallTracer, no_self, stack
from calltracer.calltracer import _readable_duration


class TestCallTracer:
    """Tests for the CallTracer decorator factory."""

    def test_init_defaults(self):
        """Verify that default arguments are set correctly."""
        tracer = CallTracer()
        assert tracer.level == logging.DEBUG
        assert tracer.trace_chain is False
        assert tracer.logger.name == "calltracer.calltracer"

    def test_init_custom_logger_and_level(self):
        """Verify that custom arguments are handled."""
        custom_logger = logging.getLogger("custom_test_logger")
        tracer = CallTracer(
            level=logging.CRITICAL, trace_chain=True, logger=custom_logger
        )
        assert tracer.level == logging.CRITICAL
        assert tracer.trace_chain is True
        assert tracer.logger == custom_logger

    def test_simple_call_with_args_and_kwargs(self, caplog):
        """Test a simple function call, checking logs and return value."""
        caplog.set_level(logging.INFO, logger="calltracer.calltracer")
        trace = CallTracer(level=logging.INFO)

        @trace
        def add(a, b):
            return a + b

        result = add(5, b=10)
        assert result == 15
        assert len(caplog.records) == 2
        assert "-->" in caplog.records[0].message
        assert "<--" in caplog.records[1].message

    def test_function_raising_exception(self, caplog):
        """Test that exceptions are logged correctly and re-raised."""
        caplog.set_level(logging.DEBUG, logger="calltracer.calltracer")
        trace = CallTracer()

        @trace
        def raise_error():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            raise_error()

        assert len(caplog.records) == 2
        exception_record = caplog.records[1]
        assert exception_record.levelno == logging.WARNING
        assert "<!>" in exception_record.message
        assert "raise_error" in exception_record.message

    def test_recursion_and_indentation(self, caplog):
        """Test a recursive function to verify indentation logic."""
        caplog.set_level(logging.DEBUG, logger="calltracer.calltracer")
        trace = CallTracer()

        @trace
        def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n - 1)

        factorial(2)
        assert len(caplog.records) == 6
        assert caplog.records[1].message.lstrip().startswith("--> Calling")
        assert caplog.records[2].message.lstrip().startswith("--> Calling")
        # Check that indentation is present and increases
        assert len(caplog.records[2].message) > len(caplog.records[1].message)

    def test_trace_chain_enabled(self, caplog):
        """Verify that the trace_chain=True feature works correctly."""
        caplog.set_level(logging.INFO, logger="calltracer.calltracer")
        trace = CallTracer(level=logging.INFO, trace_chain=True)

        @trace
        def mid_level(y):
            low_level(y + 1)

        @trace
        def low_level(z):
            return z

        @trace
        def high_level(x):
            mid_level(x * 2)

        high_level(10)

        assert len(caplog.records) == 6

        mid_entry = caplog.records[1].message
        assert "mid_level" in mid_entry
        assert "<==" in mid_entry
        assert "high_level" in mid_entry

        low_entry = caplog.records[2].message
        assert "low_level" in low_entry
        assert "mid_level" in low_entry
        assert "high_level" in low_entry

        low_exit = caplog.records[3].message
        assert "low_level" in low_exit
        assert "returned: 21" in low_exit

    def test_all_features_sync(self, caplog):
        """A comprehensive test for multiple features interacting."""
        test_logger = logging.getLogger("test_all_features_sync")

        def even_only(func_name, n):
            return n % 2 == 0

        trace = CallTracer(
            logger=test_logger,
            condition=even_only,
            timing="Mh",
            ide_support=True,
            return_transform=lambda r: f"Result was {r}",
            max_return_len=15,
        )

        @trace
        def recursive_func(n):
            if n <= 0:
                return 0
            return n + recursive_func(n - 1)

        with caplog.at_level(logging.DEBUG, logger="test_all_features_sync"):
            recursive_func(4)

        assert len(caplog.records) == 2
        assert "--> Calling recursive_func(n=4)" not in caplog.text
        assert "--> File" in caplog.records[0].message
        assert "[M:" in caplog.records[1].message
        assert "Result was 10" in caplog.records[1].message

    def test_json_output_sync(self, caplog):
        """Test that output='json' produces valid JSON."""
        import json

        test_logger = logging.getLogger("test_json_output_sync")
        trace = CallTracer(logger=test_logger, output="json")

        @trace
        def simple_func():
            return "OK"

        with caplog.at_level(logging.DEBUG, logger="test_json_output_sync"):
            simple_func()

        enter_record = json.loads(caplog.records[0].message)
        exit_record = json.loads(caplog.records[1].message)

        assert enter_record["event"] == "enter"
        assert exit_record["event"] == "exit"
        assert exit_record["result"] == "OK"

    def test_wildcard_transform(self, caplog):
        """Test transform with a wildcard for the function name."""
        test_logger = logging.getLogger("test_wildcard_transform")
        trace = CallTracer(
            logger=test_logger, transform={("*", "password"): lambda p: "***"}
        )

        @trace
        def login(username, password):
            pass

        with caplog.at_level(logging.DEBUG, logger="test_wildcard_transform"):
            login("admin", "secret123")

        assert "password='***'" in caplog.text
        assert "secret123" not in caplog.text


class TestReadableDuration:
    """Tests the _readable_duration helper function."""

    def test_formats(self):
        ns = 1234567
        assert _readable_duration(ns, DFMT.NANO) == "1234567 ns"
        assert _readable_duration(ns, DFMT.MICRO) == "1234.567 µs"
        assert _readable_duration(ns, DFMT.SEC) == "0.001234567 s"

    def test_single_format(self):
        assert "ns" in _readable_duration(100, DFMT.SINGLE)
        assert "µs" in _readable_duration(2_000_000, DFMT.SINGLE)
        assert "s" in _readable_duration(200_000_000, DFMT.SINGLE)
        assert "min" in _readable_duration(70_000_000_000, DFMT.SINGLE)
        assert "hr" in _readable_duration(4_000_000_000_000, DFMT.SINGLE)

    def test_human_format(self):
        duration = 3723_400_000_000  # 1hr, 2min, 3.4s
        assert _readable_duration(duration, DFMT.HUMAN) == "1 hr, 2 min, 3.4 s"
        assert _readable_duration(123_400_000_000, DFMT.HUMAN) == "2 min, 3.4 s"
        assert _readable_duration(3_400_000_000, DFMT.HUMAN) == "3.4 s"
        assert "ns" in _readable_duration(100, DFMT.HUMAN)  # Falls back to SINGLE
        with pytest.raises(ValueError):
            _readable_duration(1, "unknown_fmt")


# --- Helper functions for testing stack() ---
def outer_func_for_stack_test(level=logging.DEBUG, limit=None, start=0):
    """outer_func_for_stack_test"""
    middle_func_for_stack_test(level, limit, start)


def middle_func_for_stack_test(level, limit, start):
    """middle_func_for_stack_test"""
    inner_func_for_stack_test(level, limit, start)


def inner_func_for_stack_test(level, limit, start):
    """testing stack() inside the inner function"""
    stack(level=level, limit=limit, start=start)


class TestStack:
    """Tests for the stack() inspection function."""

    def test_stack_defaults(self, caplog):
        """Test stack() with default arguments from a nested call."""
        # The stack function uses the module-level logger by default
        with caplog.at_level(logging.DEBUG, logger="calltracer.calltracer"):
            outer_func_for_stack_test()

        assert len(caplog.records) > 3
        header = caplog.records[0]
        assert "Stack trace at" in header.message

        log_text = caplog.text
        assert "in middle_func_for_stack_test" in log_text
        assert "in outer_func_for_stack_test" in log_text

    def test_stack_with_limit(self, caplog):
        """Test the 'limit' parameter to restrict stack frame output."""
        with caplog.at_level(logging.DEBUG, logger="calltracer.calltracer"):
            outer_func_for_stack_test(limit=1)

        log_text = caplog.text
        assert "in middle_func_for_stack_test" in log_text
        assert "in outer_func_for_stack_test" not in log_text

    def test_stack_with_start(self, caplog):
        """Test the 'start' parameter to skip stack frames."""
        with caplog.at_level(logging.DEBUG, logger="calltracer.calltracer"):
            outer_func_for_stack_test(start=1)

        log_text = caplog.text
        assert "in middle_func_for_stack_test" not in log_text
        assert "in outer_func_for_stack_test" in log_text
