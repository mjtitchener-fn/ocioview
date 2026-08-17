# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from __future__ import annotations

from typing import Callable

from PySide6 import QtCore

from ..transform_manager import TransformManager


class ViewportSubscription:
    """
    Owns a viewport's connections to the app-wide routers and tears them
    all down symmetrically. Prevents the global routers from retaining
    callbacks into a destroyed viewport (e.g. on tab close).
    """

    def __init__(self) -> None:
        self._teardowns: list[Callable[[], None]] = []
        self._transforms_callback: Callable | None = None

    def connect_signal(
        self, signal: QtCore.SignalInstance, slot: Callable
    ) -> None:
        """Connect a Qt signal and register its disconnect for teardown."""
        signal.connect(slot)
        self._teardowns.append(lambda: signal.disconnect(slot))

    def subscribe_transform_menu(self, callback: Callable) -> None:
        """Subscribe to transform-menu updates with matching teardown."""
        TransformManager.subscribe_to_transform_menu(callback)
        self._teardowns.append(
            lambda: TransformManager.unsubscribe_from_transform_menu(callback)
        )

    def subscribe_transform_subscription_init(
        self, callback: Callable
    ) -> None:
        """Subscribe to subscription-init updates with matching teardown."""
        TransformManager.subscribe_to_transform_subscription_init(callback)
        self._teardowns.append(
            lambda: TransformManager.unsubscribe_from_transform_subscription_init(
                callback
            )
        )

    def set_transforms_callback(self, callback: Callable) -> None:
        """
        Register the per-slot transform callback. It is unsubscribed from
        all slots on teardown, and used implicitly by
        ``subscribe_transforms_at`` / ``unsubscribe_transforms``.
        """
        self._transforms_callback = callback
        self._teardowns.append(
            lambda: TransformManager.unsubscribe_from_all_transforms(callback)
        )

    def subscribe_transforms_at(self, slot: int) -> None:
        """Subscribe the registered transform callback to a specific slot."""
        TransformManager.subscribe_to_transforms_at(
            slot, self._transforms_callback
        )

    def unsubscribe_transforms(self) -> None:
        """Unsubscribe the registered transform callback from all slots."""
        TransformManager.unsubscribe_from_all_transforms(
            self._transforms_callback
        )

    def teardown(self) -> None:
        """Undo every registered connection/subscription."""
        for teardown in reversed(self._teardowns):
            teardown()
        self._teardowns.clear()
        self._transforms_callback = None
