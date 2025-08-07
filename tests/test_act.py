# tests/test_act.py
"""Tests for the asynchronous aCallTracer decorator factory"""

import asyncio
import logging

import pytest

from calltracer import CallTracer, aCallTracer


class TestACallTracer:
    """Tests for the asynchronous aCallTracer decorator factory"""

    def test_init_defaults(self):
        """Verify that default arguments are set correctly"""
        tracer = aCallTracer()
        assert tracer.level == logging.DEBUG
        assert tracer.logger.name == "calltracer.calltracer"

    def test_raises_type_error_on_sync_function(self):
        """Verify that decorating a synchronous function raises a TypeError"""
        trace = aCallTracer()

        def sync_function():
            pass  # pragma: no cover

        with pytest.raises(TypeError, match="Use CallTracer for sync functions."):
            trace(sync_function)

    @pytest.mark.asyncio
    async def test_simple_async_call(self, caplog):
        """Test a simple async function call"""
        caplog.set_level(logging.INFO, logger="calltracer.calltracer")
        trace = aCallTracer(level=logging.INFO)

        @trace
        async def async_add(a, b):
            return a + b

        result = await async_add(10, 5)
        assert result == 15
        assert len(caplog.records) == 2

    @pytest.mark.asyncio
    async def test_trace_chain_enabled_async(self, caplog):
        """Verify trace_chain=True works correctly for async functions"""
        caplog.set_level(logging.INFO, logger="calltracer.calltracer")
        trace = aCallTracer(level=logging.INFO, trace_chain=True)

        @trace
        async def async_mid(y):
            await async_low(y + 1)

        @trace
        async def async_low(z):
            return z

        @trace
        async def async_high(x):
            await async_mid(x * 2)

        await async_high(5)

        assert len(caplog.records) == 6

        mid_entry = caplog.records[1].message
        assert (
            "async_mid" in mid_entry
            and "<==" in mid_entry
            and "async_high" in mid_entry
        )

        low_entry = caplog.records[2].message
        assert (
            "async_low" in low_entry
            and "async_mid" in low_entry
            and "async_high" in low_entry
        )

        low_exit = caplog.records[3].message
        assert "async_low" in low_exit and "returned: 11" in low_exit

    @pytest.mark.asyncio
    async def test_async_exception(self, caplog):
        """Test that exceptions in async functions are logged and re-raised"""
        test_logger = logging.getLogger("test_async_exception")
        caplog.set_level(logging.DEBUG, logger="test_async_exception")
        trace = aCallTracer(logger=test_logger)

        @trace
        async def raise_async_error():
            await asyncio.sleep(0.001)
            raise RuntimeError("Async error occurred")

        with pytest.raises(RuntimeError, match="Async error occurred"):
            await raise_async_error()

        assert len(caplog.records) == 2
        exception_record = caplog.records[1]
        assert exception_record.levelno == logging.WARNING
        assert "<!>" in exception_record.message
        assert "raise_async_error" in exception_record.message
        assert "RuntimeError" in exception_record.message

    @pytest.mark.asyncio
    async def test_async_recursion_with_contextvars(self, caplog):
        """Test a recursive async function to verify contextvars-based indentation"""
        test_logger = logging.getLogger("test_async_recursion")
        caplog.set_level(logging.DEBUG)
        trace = aCallTracer(logger=test_logger)

        @trace
        async def async_factorial(n):
            if n == 0:
                return 1
            await asyncio.sleep(0.001)
            return n * await async_factorial(n - 1)

        await async_factorial(2)
        assert len(caplog.records) == 6

        # Check indentation, which is managed by contextvars
        assert caplog.records[0].message.startswith("--> Calling")
        assert caplog.records[1].message.startswith("    --> Calling")
        assert caplog.records[2].message.startswith("        --> Calling")
        assert caplog.records[3].message.startswith("        <-- Exiting")

    @pytest.mark.asyncio
    async def test_concurrency_safety(self, caplog):
        """
        Verify that contextvars keep indentation levels separate for
        concurrently running tasks. This is the most important async test
        """
        test_logger = logging.getLogger("test_concurrency_safety")
        caplog.set_level(logging.DEBUG)
        trace = aCallTracer(logger=test_logger)

        @trace
        async def concurrent_task(name, delay):
            await asyncio.sleep(delay)
            return f"Task {name} finished"

        await asyncio.gather(concurrent_task("A", 0.02), concurrent_task("B", 0.01))

        assert len(caplog.records) == 4

        task_a_entry = next(
            r for r in caplog.records if "Calling" in r.message and "'A'" in r.message
        )
        task_b_entry = next(
            r for r in caplog.records if "Calling" in r.message and "'B'" in r.message
        )

        assert task_a_entry.message.startswith("-->")
        assert task_b_entry.message.startswith("-->")

    @pytest.mark.asyncio
    async def test_sync_with_coro_raises_error(self):
        """Test that CallTracer raises a TypeError for coroutine functions."""
        trace = CallTracer()

        async def coro():
            pass  # pragma: no cover

        with pytest.raises(TypeError, match="Use aCallTracer for async functions"):
            trace(coro)

    @pytest.mark.asyncio
    async def test_all_features_async(self, caplog):
        """A comprehensive test for multiple features interacting in async."""
        test_logger = logging.getLogger("test_all_features_async")

        def even_only(func_name, n):
            return n % 2 == 0

        trace = aCallTracer(
            logger=test_logger,
            condition=even_only,
            timing="Mh",
            ide_support=True,
            return_transform=lambda r: f"Async result {r}",
            max_return_len=20,
        )

        @trace
        async def recursive_func(n):
            if n <= 0:
                return 0
            await asyncio.sleep(0.001)
            return n + await recursive_func(n - 1)

        # Use the context manager to ensure capture
        with caplog.at_level(logging.DEBUG, logger="test_all_features_async"):
            await recursive_func(
                2
            )  # Changed to even top-level; n=1 odd skipped, propagates to n=0

        # n=1 odd skipped by condition (propagates to sub n=0),
        # so only 2 records (enter/exit for n=2)
        assert len(caplog.records) == 2

        # Verify enter log: ide_support format, no timing
        enter_msg = caplog.records[0].message
        assert "--> " in enter_msg
        assert ", line " in enter_msg
        assert (
            ", in TestACallTracer.test_all_features_async.<locals>.recursive_func(n=2)"
            in enter_msg
        )

        # Verify exit log: timing block, transformed/truncated return
        exit_msg = caplog.records[1].message
        assert "[M: " in exit_msg  # Exclusive monotonic
        assert " | h: " in exit_msg  # Inclusive perf_counter
        assert "<-- " in exit_msg
        assert ", returned: 'Async result 3'" in exit_msg  # Transformed, len ok

    @pytest.mark.asyncio
    async def test_json_output_async(self, caplog):
        """Test that output='json' produces valid JSON for async functions."""
        import json

        test_logger = logging.getLogger("test_json_output_async")
        caplog.set_level(logging.DEBUG)
        trace = aCallTracer(logger=test_logger, output="json")

        @trace
        async def simple_async_func():
            return "OK_ASYNC"

        await simple_async_func()

        enter_record = json.loads(caplog.records[0].message)
        exit_record = json.loads(caplog.records[1].message)

        assert enter_record["event"] == "enter"
        assert exit_record["event"] == "exit"
        assert exit_record["result"] == "OK_ASYNC"

    @pytest.mark.asyncio
    async def test_enable_disable_async(self, caplog):
        """Tests that the async tracer can be enabled and disabled."""
        test_logger = logging.getLogger("test_enable_disable_async")
        caplog.set_level(logging.DEBUG)
        trace = aCallTracer(logger=test_logger)

        @trace
        async def simple_async_func():
            pass

        await simple_async_func()
        assert len(caplog.records) == 2
        caplog.clear()

        trace.disable()
        await simple_async_func()
        assert len(caplog.records) == 0

        trace.enable()
        await simple_async_func()
        assert len(caplog.records) == 2
