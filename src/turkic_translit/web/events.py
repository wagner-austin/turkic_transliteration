"""Gradio event listeners, typed without the stubs Gradio writes at import.

Gradio attaches each component's event methods (``click``, ``change`` and
the rest) at runtime, and the first time it is imported it writes a
``.pyi`` beside every component module in its own install directory
describing them. Mypy therefore saw ``Button.click`` only in an
environment where something had already imported Gradio: a fresh install
failed ``make lint`` with ``attr-defined`` on every listener, and the same
tree passed once the tests had run. Measured 2026-11-04 on a fresh Gradio
6.24.0 install: no ``.pyi`` under ``gradio/components`` before
``import gradio``, 47 after it.

The listeners are reached here instead, by ``getattr`` bound directly to a
Protocol stating the part of the signature this project calls, so type
checking no longer depends on what ran before it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, ParamSpec, Protocol, TypeVar

from gradio.blocks import Block
from gradio.events import Dependency

P = ParamSpec("P")
R = TypeVar("R")


class Listener(Protocol):
    """One event of one component, bound to it.

    The signature is the leading part of Gradio's own listener signature
    (``fn``, ``inputs``, ``outputs``), which every event shares.
    """

    def __call__(
        self, fn: Callable[P, R], inputs: Sequence[Block], outputs: Sequence[Block]
    ) -> Dependency:
        """Register ``fn`` to run when the event fires.

        Args:
            fn: The function Gradio calls with the input components' values.
            inputs: Components whose values are passed to ``fn``.
            outputs: Components that receive ``fn``'s return values.

        Returns:
            The dependency Gradio recorded on the enclosing ``Blocks``.
        """
        ...


def listener(component: Block, event: Literal["click", "change"]) -> Listener:
    """Return one of ``component``'s event listeners.

    Args:
        component: A Gradio component created inside a ``Blocks`` context.
        event: The event to listen for.

    Returns:
        The component's bound listener for ``event``.

    Raises:
        AttributeError: When the component has no such event.
    """
    bound: Listener = getattr(component, event)
    return bound


__all__ = ["Listener", "listener"]
