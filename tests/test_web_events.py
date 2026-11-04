"""The typed listener registers on real Gradio components.

Each case builds a real ``Blocks``, registers a handler through
:func:`turkic_translit.web.events.listener`, and reads back the function
Gradio recorded, so the Protocol is checked against the listener Gradio
actually attaches rather than against its own description of it.
"""

from __future__ import annotations

import gradio as gr
import pytest

from turkic_translit.web.events import listener


def shout(text: str) -> str:
    """Upper-case ``text``, as a handler with a real signature.

    Args:
        text: The value of the input component.

    Returns:
        ``text`` in upper case.
    """
    return text.upper()


def test_a_click_listener_records_its_handler_and_components() -> None:
    """``listener(button, "click")`` registers the handler for that click."""
    with gr.Blocks() as demo:
        source = gr.Textbox()
        target = gr.Textbox()
        button = gr.Button("Go")
        listener(button, "click")(shout, [source], [target])

    recorded = [block_fn for block_fn in demo.fns.values() if block_fn.fn is shout]
    assert len(recorded) == 1
    assert recorded[0].targets == [(button._id, "click")]
    assert [block._id for block in recorded[0].inputs] == [source._id]
    assert [block._id for block in recorded[0].outputs] == [target._id]


def test_a_change_listener_is_bound_to_the_change_event() -> None:
    """``"change"`` reaches the component's change event, not its click."""
    with gr.Blocks() as demo:
        box = gr.Textbox()
        echo = gr.Textbox()
        listener(box, "change")(shout, inputs=[box], outputs=[echo])

    recorded = [block_fn for block_fn in demo.fns.values() if block_fn.fn is shout]
    assert [block_fn.targets for block_fn in recorded] == [[(box._id, "change")]]


def test_an_event_the_component_lacks_is_an_attribute_error() -> None:
    """A Markdown block has no click event, and asking for one raises."""
    with gr.Blocks(), pytest.raises(AttributeError, match="click"):
        listener(gr.Markdown(), "click")
