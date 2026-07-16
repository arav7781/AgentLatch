"""Pytest configuration and global hooks."""

from __future__ import annotations


def pytest_sessionstart(session):
    """Called before the test session starts.

    Prints the AgentLatch cosmic ASCII startup banner.
    """
    from agentlatch.banner import initialize_latch
    initialize_latch()
