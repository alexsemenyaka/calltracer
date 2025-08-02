import asyncio
import logging
import pytest
import contextvars

from calltracer import aCallTracer, tracer_indent

class TestACallTracer:
    """Tests for the asynchronous aCallTracer decorator factory."""

    def test_init_defaults(self):
        """Verify that default arguments are set correctly."""
        tracer = aCallTracer()
        assert tracer.level == logging.DEBUG
        assert tracer.logger.name == 'calltracer.calltracer'

    def test_raises_type_error_on_sync_function(self):
        """Verify that decorating a synchronous function raises a TypeError."""
        trace = aCallTracer()

        def sync_function():
            pass  # pragma: no cover

        with pytest.raises(TypeError, match="aCallTracer can only be used to decorate async functions."):
            trace(sync_function)

    @pytest.mark.asyncio
    async def test_simple_async_call(self, caplog):
        """Test a simple async function call, checking logs and return value."""
        caplog.set_level(logging.INFO, logger='calltracer.async_tracer')
        trace = aCallTracer(level=logging.INFO, logger=logging.getLogger('calltracer.async_tracer'))

        @trace
        async def async_add(a, b):
            await asyncio.sleep(0.01)
            return a + b

        result = await async_add(10, 5)
        assert result == 15
        assert len(caplog.records) == 2

        enter_log = caplog.records[0].message
        assert "--> Calling" in enter_log
        assert "async_add(10, 5)" in enter_log

        exit_log = caplog.records[1].message
        assert "<-- Exiting" in exit_log
        assert "async_add" in exit_log
        assert "returned: 15" in exit_log

    @pytest.mark.asyncio
    async def test_async_exception(self, caplog):
        """Test that exceptions in async functions are logged and re-raised."""
        caplog.set_level(logging.DEBUG, logger='calltracer.async_tracer')
        trace = aCallTracer(logger=logging.getLogger('calltracer.async_tracer'))

        @trace
        async def raise_async_error():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error occurred")

        with pytest.raises(RuntimeError, match="Async error occurred"):
            await raise_async_error()

        assert len(caplog.records) == 2
        exception_record = caplog.records[1]
        assert exception_record.levelno == logging.WARNING
        assert "<!> Exiting" in exception_record.message
        assert "raise_async_error" in exception_record.message
        assert "RuntimeError('Async error occurred')" in exception_record.message

    @pytest.mark.asyncio
    async def test_async_recursion_with_contextvars(self, caplog):
        """Test a recursive async function to verify contextvars-based indentation."""
        caplog.set_level(logging.DEBUG, logger='calltracer.async_tracer')
        trace = aCallTracer(logger=logging.getLogger('calltracer.async_tracer'))

        @trace
        async def async_factorial(n):
            if n == 0:
                return 1
            await asyncio.sleep(0.01)
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
        concurrently running tasks. This is the most important async test.
        """
        caplog.set_level(logging.DEBUG, logger='calltracer.async_tracer')
        trace = aCallTracer(logger=logging.getLogger('calltracer.async_tracer'))

        @trace
        async def concurrent_task(name, delay):
            await asyncio.sleep(delay)
            return f"Task {name} finished"

        # Run two tasks concurrently
        await asyncio.gather(
            concurrent_task("A", 0.02),
            concurrent_task("B", 0.01)
        )

        assert len(caplog.records) == 4
        
        # Filter entry and exit logs for each task
        task_a_entry = next(r for r in caplog.records if "Calling" in r.message and "'A'" in r.message)
        task_b_entry = next(r for r in caplog.records if "Calling" in r.message and "'B'" in r.message)

        # CRITICAL CHECK: Both tasks should have an indentation level of 0.
        # This proves that contextvars are isolating their states.
        assert not task_a_entry.message.startswith("    ")
        assert not task_b_entry.message.startswith("    ")
