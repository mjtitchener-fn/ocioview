# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from __future__ import annotations

from typing import Type

import PyOpenColorIO as ocio

from ..config_cache import ConfigCache
from ..items.view_model import ViewModel
from ..mode import OCIOViewMode
from ..processor_context import ProcessorContext
from ..ref_space_manager import ReferenceSpaceManager


class ViewerTransformBuilder:
    """
    Pure construction of the OCIO processor context and viewer transform
    from explicit inputs. Holds no widget state so it can be unit-tested
    in isolation.
    """

    @staticmethod
    def make_processor_context(
        mode: OCIOViewMode,
        input_color_space: str | None,
        view: str | None,
        transform_item_type: Type | None,
        transform_item_name: str | None,
        transform_direction: ocio.TransformDirection,
    ) -> ProcessorContext:
        """Build a processor context for the given viewer state."""
        if mode == OCIOViewMode.Preview:
            return ProcessorContext(
                input_color_space,
                ViewModel.__item_type__,
                view,
                ocio.TRANSFORM_DIR_FORWARD,
            )
        else:  # Edit
            return ProcessorContext(
                input_color_space,
                transform_item_type,
                transform_item_name,
                transform_direction,
            )

    @staticmethod
    def make_transform(
        mode: OCIOViewMode,
        display: str,
        view: str,
        transform_fwd: ocio.Transform | None,
        transform_inv: ocio.Transform | None,
        transform_direction: ocio.TransformDirection,
    ) -> ocio.Transform | None:
        """
        Build the viewer transform, relative to the current config's
        scene reference space. Returns None in Preview when no display/
        view is selected; returns a no-op ExponentTransform in Edit when
        no subscription transform pair is available (matching the prior
        behavior that avoids stale processors on mode switches).
        """
        if mode == OCIOViewMode.Preview:
            if display and view:
                scene_reference = (
                    ReferenceSpaceManager.scene_reference_space().getName()
                )
                return ocio.DisplayViewTransform(
                    src=scene_reference,
                    display=display,
                    view=view,
                    direction=ocio.TRANSFORM_DIR_FORWARD,
                )
            return None

        # Edit
        if transform_fwd is not None and transform_inv is not None:
            if transform_direction == ocio.TRANSFORM_DIR_INVERSE:
                return transform_inv
            return transform_fwd
        return ocio.ExponentTransform()

    @staticmethod
    def default_color_space() -> str:
        """Get a reasonable default scene-referred color space name."""
        all_color_spaces = ConfigCache.get_color_space_names(
            ocio.SEARCH_REFERENCE_SPACE_SCENE
        )
        default_color_space = ConfigCache.get_default_color_space_name()
        if (
            default_color_space is not None
            and default_color_space in all_color_spaces
        ):
            return default_color_space
        elif all_color_spaces:
            return all_color_spaces[0]
        else:
            return ""

    @staticmethod
    def displays() -> list[str]:
        """Get all active OCIO displays."""
        config = ocio.GetCurrentConfig()
        return list(config.getDisplays())

    @staticmethod
    def default_display() -> str:
        """Get the default OCIO display."""
        config = ocio.GetCurrentConfig()
        return config.getDefaultDisplay()

    @staticmethod
    def views(display: str, input_color_space: str) -> list[str]:
        """Get active OCIO views for a display and input color space."""
        config = ocio.GetCurrentConfig()
        if input_color_space:
            return config.getViews(display, input_color_space)
        return config.getViews(display)

    @staticmethod
    def default_view(
        display: str, input_color_space: str | None = None
    ) -> str:
        """
        Get the default OCIO view for a display, honoring viewing rules for
        the given input color space when one is provided.
        """
        config = ocio.GetCurrentConfig()
        if input_color_space:
            return config.getDefaultView(display, input_color_space)
        return config.getDefaultView(display)
