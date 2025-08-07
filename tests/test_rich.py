# tests/test_rich.py
"""
Tests for the RichPyTraceHandler, including capturing `rich` output.
"""

import builtins
import importlib
import logging
from io import StringIO

import pytest
from pyte import Screen, Stream

# This import is conditional because we test the case where rich is not installed
try:
    from rich.console import Console

    from calltracer import CallTracer, RichPyTraceHandler

    RICH_INSTALLED = True
except ImportError:
    RICH_INSTALLED = False

# Skip all tests in this file if rich is not installed
pytestmark = pytest.mark.skipif(not RICH_INSTALLED, reason="rich is not installed")


def get_final_text(output: str, width=80, height=24) -> str:
    screen = Screen(width, height)  # Размер произвольный, но достаточный для дерева
    stream = Stream(screen)
    stream.feed(output)
    return "\n".join(line.rstrip() for line in screen.display if line.strip())


@pytest.fixture
def rich_tester(request):
    """A pytest fixture to set up a logger with a rich handler writing to a buffer."""
    handler_kwargs = getattr(request, "param", {})

    string_io = StringIO()
    # force_terminal=True is essential to get ANSI codes for color tests
    test_console = Console(
        file=string_io, force_terminal=True, color_system="truecolor"
    )

    handler = RichPyTraceHandler(**handler_kwargs)
    handler.console = test_console

    # Use a unique logger name for each test run to avoid state leakage
    log = logging.getLogger(f"rich_test_logger_{id(handler_kwargs)}")
    log.setLevel(logging.DEBUG)
    log.handlers = [handler]
    log.propagate = False

    yield log, string_io

    log.handlers = []


@pytest.mark.parametrize("rich_tester", [{"overwrite": False}], indirect=True)
def test_rich_handler_append_mode(rich_tester):
    """Tests the default append-only tree mode."""
    log, buffer = rich_tester
    trace = CallTracer(logger=log, output="json")

    @trace
    def recursive_func(n):
        if n > 0:
            recursive_func(n - 1)

    recursive_func(1)

    output = buffer.getvalue()
    # In append mode, both enter (➡️) and exit (⬅️) symbols should be present
    assert "➡️" in output
    assert "⬅️" in output


@pytest.mark.parametrize("rich_tester", [{"overwrite": True}], indirect=True)
def test_rich_handler_overwrite_mode(rich_tester):
    """Tests the live overwrite mode."""
    log, buffer = rich_tester
    trace = CallTracer(logger=log, output="json")

    @trace
    def recursive_func(n):
        if n > 0:
            recursive_func(n - 1)

    recursive_func(1)

    output = buffer.getvalue()
    # In overwrite mode, the final output has been overwritten, so we don't see the enter symbol
    output = buffer.getvalue()
    final_text = get_final_text(output)
    assert "➡️" not in final_text
    assert "⬅" in final_text


@pytest.mark.parametrize(
    "rich_tester", [{"color_enter": "magenta", "color_exit": "cyan"}], indirect=True
)
def test_rich_handler_custom_colors(rich_tester):
    """Tests custom color configuration."""
    log, buffer = rich_tester
    trace = CallTracer(logger=log, output="json")

    @trace
    def func_a():
        pass

    func_a()

    output = buffer.getvalue()
    # Check for ANSI escape codes for the specified colors, not the names
    assert "\x1b[35m" in output  # Magenta
    assert "\x1b[36m" in output  # Cyan


def test_rich_handler_raises_error_if_rich_not_installed(monkeypatch):
    """
    Verify an ImportError is raised when rich is not installed by
    simulating its absence.
    """
    from calltracer import richlogging

    original = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rich" or name.startswith("rich."):
            raise ImportError(f"No module named '{name}'")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Reload our handler module. Now, its 'import rich' will fail.
    importlib.reload(richlogging)

    # Reload our handler module. Now, its 'import rich' will fail.
    importlib.reload(richlogging)

    assert not richlogging.RICH_INSTALLED

    # Assert that instantiating the handler now raises our specific error
    with pytest.raises(ImportError, match="pip install pytracecall\\[rich\\]"):
        richlogging.RichPyTraceHandler()

    # It's good practice to restore the original state after the test
    monkeypatch.undo()
    importlib.reload(richlogging)
