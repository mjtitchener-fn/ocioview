# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from queue import Empty

import pytest

from ocioview.message_router import (
    MessageRunner,
    MessageRouterGate,
    UpdateType,
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
    assert runner.updates_allowed(UpdateType.PROCESSOR) is False

    runner.request_updates(UpdateType.PROCESSOR)
    assert runner.updates_allowed(UpdateType.PROCESSOR) is True

    runner.release_updates(UpdateType.PROCESSOR)
    assert runner.updates_allowed(UpdateType.PROCESSOR) is False


def test_reference_count_is_order_independent(qapp):
    runner = MessageRunner()
    runner.request_updates(UpdateType.PROCESSOR)
    runner.request_updates(UpdateType.PROCESSOR)  # second requester
    assert runner.updates_allowed(UpdateType.PROCESSOR) is True

    runner.release_updates(UpdateType.PROCESSOR)  # first releases
    assert (
        runner.updates_allowed(UpdateType.PROCESSOR) is True
    )  # still on for 2nd

    runner.release_updates(UpdateType.PROCESSOR)
    assert runner.updates_allowed(UpdateType.PROCESSOR) is False


def test_release_floors_at_zero(qapp):
    runner = MessageRunner()
    runner.release_updates(UpdateType.CONFIG)  # never requested
    assert runner.updates_allowed(UpdateType.CONFIG) is False
    runner.request_updates(UpdateType.CONFIG)
    assert runner.updates_allowed(UpdateType.CONFIG) is True


def test_zero_to_one_rebroadcasts_cached_record(qapp):
    runner = MessageRunner()
    runner._prev_config = object()  # stand-in cached record
    _drain_queue()

    runner.request_updates(UpdateType.CONFIG)  # 0 -> 1 rebroadcasts
    assert message_queue.get_nowait() is runner._prev_config

    runner.request_updates(UpdateType.CONFIG)  # 1 -> 2 does NOT rebroadcast
    _drain_queue()  # (nothing expected; just ensure no crash)


def test_unknown_update_type_raises(qapp):
    runner = MessageRunner()
    with pytest.raises(ValueError):
        runner.request_updates("bogus")


def test_gate_requests_and_releases_on_diff(qapp):
    runner = MessageRunner()
    gate = MessageRouterGate(router=runner)

    gate.set_requested(UpdateType.PROCESSOR)
    assert runner.updates_allowed(UpdateType.PROCESSOR) is True

    gate.set_requested(UpdateType.PROCESSOR, UpdateType.CONFIG)  # add config
    assert runner.updates_allowed(UpdateType.CONFIG) is True
    assert runner.updates_allowed(UpdateType.PROCESSOR) is True

    gate.set_requested(UpdateType.CONFIG)  # drop processor
    assert runner.updates_allowed(UpdateType.PROCESSOR) is False
    assert runner.updates_allowed(UpdateType.CONFIG) is True

    gate.set_requested()  # release all
    assert runner.updates_allowed(UpdateType.CONFIG) is False


def test_gate_set_requested_is_idempotent(qapp):
    runner = MessageRunner()
    gate = MessageRouterGate(router=runner)

    gate.set_requested(UpdateType.PROCESSOR)
    gate.set_requested(UpdateType.PROCESSOR)  # same set -> no extra request
    assert runner._update_counts[UpdateType.PROCESSOR] == 1


def test_two_gates_sharing_a_type_are_order_independent(qapp):
    runner = MessageRunner()
    gate_a = MessageRouterGate(router=runner)
    gate_b = MessageRouterGate(router=runner)

    gate_a.set_requested(UpdateType.PROCESSOR)
    gate_b.set_requested(UpdateType.PROCESSOR)
    assert runner._update_counts[UpdateType.PROCESSOR] == 2

    gate_a.set_requested()  # a hides; b still visible
    assert runner.updates_allowed(UpdateType.PROCESSOR) is True
