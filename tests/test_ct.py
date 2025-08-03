# tests/test_ct.py

"""calltracer synchronous tests"""
import logging
import pytest
from calltracer.calltracer import CallTracer, stack

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
        tracer = CallTracer(level=logging.CRITICAL, trace_chain=True, logger=custom_logger)
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
        assert "--> Calling" in caplog.records[0].message
        assert "<-- Exiting" in caplog.records[1].message

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
        assert "<!> Exiting" in exception_record.message
        assert "raise_error" in exception_record.message

    def test_function_raising_exception_with_chaining(self, caplog):
        """Test that exceptions are logged correctly and re-raised."""
        caplog.set_level(logging.DEBUG, logger="calltracer.calltracer")
        trace = CallTracer(trace_chain=True)

        @trace
        def raise_error():
            raise ValueError("Something went wrong")

        @trace
        def error_deeper():
            raise_error()

        with pytest.raises(ValueError, match="Something went wrong"):
            error_deeper()

        assert len(caplog.records) == 4
        exception_record1 = caplog.records[1]
        exception_record3 = caplog.records[3]
        assert exception_record1.levelno == logging.DEBUG
        assert exception_record3.levelno == logging.WARNING
        assert "<!> Exiting" in exception_record3.message
        assert "raise_error" in exception_record1.message
        assert "<==" in exception_record1.message

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
        assert caplog.records[1].message.startswith("    --> Calling")
        assert caplog.records[2].message.startswith("        --> Calling")

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
 
        # --- Corrected Assertions ---
        mid_entry = caplog.records[1].message
        assert "mid_level(20)" in mid_entry
        assert "<==" in mid_entry
        assert "high_level(10)" in mid_entry
 
        low_entry = caplog.records[2].message
        assert "low_level(21)" in low_entry
        assert "mid_level(20)" in low_entry
        assert "high_level(10)" in low_entry
 
        low_exit = caplog.records[3].message
        assert "low_level(21), returned: 21" in low_exit
        assert "(pending:" in low_exit
        assert "mid_level(20)" in low_exit

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
