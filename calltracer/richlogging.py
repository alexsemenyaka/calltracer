# pytracecall/richlogging.py

import json
import logging
import threading
from typing import Dict, Tuple

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree

    RICH_INSTALLED = True
except ImportError:
    RICH_INSTALLED = False
    Console, Live, Text, Tree = object, object, object, object

# State storage for both modes
# For overwrite mode: {thread_id: (Live, Tree, node_stack)}
# For append mode:    {thread_id: (Tree, node_stack)}
_thread_contexts: Dict[int, Tuple] = {}


class RichPyTraceHandler(logging.Handler):
    """
    A logging handler that renders pytracecall events as a rich Tree.
    Requires the 'rich' extra to be installed (`pip install pytracecall[rich]`).
    """

    def __init__(
        self,
        overwrite: bool = False,
        color_enter: str = "green",
        color_exit: str = "bold blue",
        color_exception: str = "bold red",
        color_timing: str = "yellow",
        *args,
        **kwargs,
    ):
        """
        Initializes the handler.

        Args:
            overwrite (bool): If True, uses a Live display to overwrite enter
                nodes with exit information. If False (default), creates an
                append-only tree showing both enter and exit events.
            color_enter (str): Rich markup for the enter event message.
            color_exit (str): Rich markup for the exit event message.
            color_exception (str): Rich markup for the exception event message.
            color_timing (str): Rich markup for the timing information.
        """
        super().__init__(*args, **kwargs)
        if not RICH_INSTALLED:
            raise ImportError(
                "The 'rich' library is required for RichPyTraceHandler. "
                "Please install it with: pip install pytracecall[rich]"
            )
        self.console = Console()
        self.overwrite = overwrite
        # Store colors
        self.color_enter = color_enter
        self.color_exit = color_exit
        self.color_exception = color_exception
        self.color_timing = color_timing

    def emit(self, record: logging.LogRecord) -> None:
        """Processes a log record and updates/prints the call tree."""
        try:
            data = json.loads(record.msg)
            if "event" not in data or "current_call_sig" not in data:
                self.console.print(record.getMessage())
                return
        except (json.JSONDecodeError, TypeError):  # pragma: no cover
            self.console.print(record.getMessage())
            return

        # Dispatch to the correct handler method based on the mode
        if self.overwrite:
            self._emit_overwrite(data)
        else:
            self._emit_append(data)

    def _emit_append(self, data: dict):
        """Handles the append-only tree mode (default)."""
        thread_id = threading.get_ident()
        event = data["event"]

        if thread_id not in _thread_contexts:
            tree = Tree("", hide_root=True)
            _thread_contexts[thread_id] = (tree, [])  # State: (tree_root, node_stack)

        tree_root, node_stack = _thread_contexts[thread_id]

        if event == "enter":
            parent_node = node_stack[-1] if node_stack else tree_root
            markup = (
                f"[{self.color_enter}]➡️ {data['current_call_sig']}[/{self.color_enter}]"
            )
            new_node = parent_node.add(Text.from_markup(markup))
            node_stack.append(new_node)

        elif event in ("exit", "exception"):
            if not node_stack:  # pragma: no cover
                return
            current_node = node_stack.pop()

            timing_block = data.get("timing_block", "").strip()
            if timing_block:
                timing_block = (
                    f" [{self.color_timing}]({timing_block})[/{self.color_timing}]"
                )

            if event == "exit":
                result_repr = repr(data["result"])
                markup = f"[{self.color_exit}]⬅️ returned: {result_repr}[/{self.color_exit}]{timing_block}"
            else:  # exception
                exc_repr = repr(data["exception"])
                markup = f"[{self.color_exception}]💥 raised: {exc_repr}[/{self.color_exception}]{timing_block}"

            current_node.add(Text.from_markup(markup))

            if not node_stack:
                self.console.print(tree_root)
                del _thread_contexts[thread_id]

    def _emit_overwrite(self, data: dict):
        """Handles the Live overwrite mode."""
        thread_id = threading.get_ident()
        event = data["event"]

        if thread_id not in _thread_contexts:
            tree = Tree("", hide_root=True)
            live = Live(tree, console=self.console, auto_refresh=False, transient=False)
            live.start(refresh=True)
            _thread_contexts[thread_id] = (live, tree, [])  # State: (live, tree, stack)

        live, tree_root, node_stack = _thread_contexts[thread_id]

        if event == "enter":
            parent_node = node_stack[-1] if node_stack else tree_root
            markup = (
                f"[{self.color_enter}]➡️ {data['current_call_sig']}[/{self.color_enter}]"
            )
            new_node = parent_node.add(Text.from_markup(markup))
            node_stack.append(new_node)
            live.refresh()

        elif event in ("exit", "exception"):
            if not node_stack:
                return
            current_node = node_stack.pop()

            timing_block = data.get("timing_block", "").strip()
            if timing_block:
                timing_block = (
                    f" [{self.color_timing}]({timing_block})[/{self.color_timing}]"
                )

            if event == "exit":
                result_repr = repr(data["result"])
                markup = f"[{self.color_exit}]⬅️ {data['current_call_sig']}[/{self.color_exit}], returned: {result_repr}{timing_block}"
            else:  # exception
                exc_repr = repr(data["exception"])
                markup = f"[{self.color_exception}]💥 {data['current_call_sig']}[/{self.color_exception}], raised: {exc_repr}{timing_block}"

            current_node.label = Text.from_markup(markup)
            live.refresh()

            if not node_stack:
                live.stop()
                del _thread_contexts[thread_id]
