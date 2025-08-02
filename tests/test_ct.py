"""calltracer tests"""
import logging

import pytest

from calltracer.calltracer import CallTracer, stack

# In tests/test_ct.py, replace the whole TestCallTracer class


class TestCallTracer:
    """Tests for the CallTracer decorator factory."""

    def test_init_defaults(self):
        """Verify that default arguments are set correctly."""
        tracer = CallTracer()
        assert tracer.level == logging.DEBUG
        assert tracer.logger.name == "calltracer.calltracer"

    def test_init_custom_logger_and_level(self):
        """Verify that custom arguments are handled."""
        custom_logger = logging.getLogger("custom_test_logger")
        tracer = CallTracer(level=logging.CRITICAL, logger=custom_logger)
        assert tracer.level == logging.CRITICAL
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

        # FIX: Check for key parts instead of an exact match
        enter_log = caplog.records[0].message
        assert "--> Calling" in enter_log
        assert "add(5, b=10)" in enter_log

        exit_log = caplog.records[1].message
        assert "<-- Exiting" in exit_log
        assert "add" in exit_log
        assert "returned: 15" in exit_log

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

        # Now, assert against the record's attributes
        assert exception_record.levelno == logging.WARNING

        # And assert against the message content from the record
        assert "<!> Exiting" in exception_record.message
        assert "raise_error" in exception_record.message
        assert "ValueError('Something went wrong')" in exception_record.message

    def test_recursion_and_indentation(self, caplog):
        """Test a recursive function to verify indentation logic."""
        caplog.set_level(logging.DEBUG, logger="calltracer.calltracer")
        trace = CallTracer()

        @trace
        def factorial(n):
            if n == 0:
                return 1
            return n * factorial(n - 1)

        result = factorial(2)
        assert result == 2
        assert len(caplog.records) == 6

        # FIX: Check for startswith for indentation, and 'in' for content
        # factorial(2)
        assert caplog.records[0].message.startswith("--> Calling")
        assert "factorial(2)" in caplog.records[0].message
        # factorial(1)
        assert caplog.records[1].message.startswith("    --> Calling")
        assert "factorial(1)" in caplog.records[1].message
        # factorial(0)
        assert caplog.records[2].message.startswith("        --> Calling")
        assert "factorial(0)" in caplog.records[2].message


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
