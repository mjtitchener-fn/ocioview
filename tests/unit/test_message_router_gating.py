# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from queue import Empty

import pytest

from ocioview.message_router import (
    MessageRunner,
    MessageRouterGate,
    message_queue,
)


def _drain_queue():
    while True:
        try:
            message_queue.get_nowait()
        except Empty:
            break


def test_request_enables_and_release_disables(qapp):
    runner = MessageRunner()
    assert runner.updates_allowed("processor") is False

    runner.request_updates("processor")
    assert runner.updates_allowed("processor") is True

    runner.release_updates("processor")
    assert runner.updates_allowed("processor") is False


def test_reference_count_is_order_independent(qapp):
    runner = MessageRunner()
    runner.request_updates("processor")
    runner.request_updates("processor")  # second requester
    assert runner.updates_allowed("processor") is True

    runner.release_updates("processor")  # first releases
    assert runner.updates_allowed("processor") is True  # still on for 2nd

    runner.release_updates("processor")
    assert runner.updates_allowed("processor") is False


def test_release_floors_at_zero(qapp):
    runner = MessageRunner()
    runner.release_updates("config")  # never requested
    assert runner.updates_allowed("config") is False
    runner.request_updates("config")
    assert runner.updates_allowed("config") is True


def test_zero_to_one_rebroadcasts_cached_record(qapp):
    runner = MessageRunner()
    runner._prev_config = object()  # stand-in cached record
    _drain_queue()

    runner.request_updates("config")  # 0 -> 1 rebroadcasts
    assert message_queue.get_nowait() is runner._prev_config

    runner.request_updates("config")  # 1 -> 2 does NOT rebroadcast
    _drain_queue()  # (nothing expected; just ensure no crash)


def test_unknown_update_type_raises(qapp):
    runner = MessageRunner()
    with pytest.raises(ValueError):
        runner.request_updates("bogus")


def test_gate_requests_and_releases_on_diff(qapp):
    runner = MessageRunner()
    gate = MessageRouterGate(router=runner)

    gate.set_requested("processor")
    assert runner.updates_allowed("processor") is True

    gate.set_requested("processor", "config")  # add config
    assert runner.updates_allowed("config") is True
    assert runner.updates_allowed("processor") is True

    gate.set_requested("config")  # drop processor
    assert runner.updates_allowed("processor") is False
    assert runner.updates_allowed("config") is True

    gate.set_requested()  # release all
    assert runner.updates_allowed("config") is False


def test_gate_set_requested_is_idempotent(qapp):
    runner = MessageRunner()
    gate = MessageRouterGate(router=runner)

    gate.set_requested("processor")
    gate.set_requested("processor")  # same set -> no extra request
    assert runner._update_counts["processor"] == 1


def test_two_gates_sharing_a_type_are_order_independent(qapp):
    runner = MessageRunner()
    gate_a = MessageRouterGate(router=runner)
    gate_b = MessageRouterGate(router=runner)

    gate_a.set_requested("processor")
    gate_b.set_requested("processor")
    assert runner._update_counts["processor"] == 2

    gate_a.set_requested()  # a hides; b still visible
    assert runner.updates_allowed("processor") is True
