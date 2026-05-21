"""Tests for fund checkpointer."""

from fund.checkpointer import thread_id


def test_thread_id_format():
    tid = thread_id("user1", "110011", "2026-05-22")
    assert isinstance(tid, str)
    assert len(tid) == 16


def test_thread_id_deterministic():
    t1 = thread_id("user1", "110011", "2026-05-22")
    t2 = thread_id("user1", "110011", "2026-05-22")
    assert t1 == t2


def test_thread_id_differs_for_different_inputs():
    t1 = thread_id("user1", "110011", "2026-05-22")
    t2 = thread_id("user2", "110011", "2026-05-22")
    assert t1 != t2
