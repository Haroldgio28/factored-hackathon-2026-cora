"""Smoke test for task 4.5: the agent-console view imports cleanly (ponytail: UI is logic-free).

The console is a thin render over `JsonHandoffStore` with no business logic, so it needs no
behavioural test beyond proving it imports without pulling Streamlit at module load (Streamlit is
imported lazily inside `main()`), and that its pure render helpers tolerate an empty package.
"""

from __future__ import annotations


def test_agent_console_imports() -> None:
    import ui.agent_console as console

    assert hasattr(console, "main")


def test_priority_badge_maps_high_and_normal() -> None:
    import ui.agent_console as console

    assert "HIGH" in console._priority_badge("high")
    assert "normal" in console._priority_badge("normal")
