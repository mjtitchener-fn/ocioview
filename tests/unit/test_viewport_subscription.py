# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from ocioview.signal_router import SignalRouter
from ocioview.transform_manager import TransformManager
from ocioview.viewer.viewport_subscription import ViewportSubscription


def test_signal_connection_is_torn_down():
    received = []
    sub = ViewportSubscription()
    router = SignalRouter.get_instance()

    sub.connect_signal(router.mode_changed, lambda: received.append(1))
    router.mode_changed.emit()
    assert received == [1]

    sub.teardown()
    router.mode_changed.emit()
    assert received == [1]  # no further delivery after teardown


def test_transform_menu_subscription_is_torn_down():
    calls = []

    def on_menu(items):
        calls.append(items)

    sub = ViewportSubscription()
    sub.subscribe_transform_menu(on_menu)
    # Immediate callback on subscribe (TransformManager behavior)
    assert len(calls) == 1
    assert on_menu in TransformManager._tf_menu_subscribers

    sub.teardown()
    assert on_menu not in TransformManager._tf_menu_subscribers


def test_transform_subscription_init_is_torn_down():
    def on_init(slot):
        pass

    sub = ViewportSubscription()
    sub.subscribe_transform_subscription_init(on_init)
    assert on_init in TransformManager._tf_subscribers[-1]

    sub.teardown()
    assert on_init not in TransformManager._tf_subscribers[-1]


def test_transforms_at_callback_is_torn_down():
    def on_tf(slot, fwd, inv):
        pass

    sub = ViewportSubscription()
    sub.set_transforms_callback(on_tf)
    sub.subscribe_transforms_at(3)
    assert on_tf in TransformManager._tf_subscribers[3]

    sub.teardown()
    # unsubscribe_from_all_transforms removes it from every slot list
    assert on_tf not in TransformManager._tf_subscribers[3]


def test_teardown_is_idempotent():
    received = []
    sub = ViewportSubscription()
    router = SignalRouter.get_instance()

    sub.connect_signal(router.mode_changed, lambda: received.append(1))
    sub.teardown()
    sub.teardown()  # second call must be a safe no-op

    router.mode_changed.emit()
    assert received == []
