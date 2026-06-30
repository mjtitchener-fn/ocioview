# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

import PyOpenColorIO as ocio

from ocioview.mode import OCIOViewMode
from ocioview.items.view_model import ViewModel
from ocioview.viewer.transform_builder import ViewerTransformBuilder


def test_processor_context_preview_uses_view_model_type():
    ctx = ViewerTransformBuilder.make_processor_context(
        mode=OCIOViewMode.Preview,
        input_color_space="acescg",
        view="sRGB",
        transform_item_type=None,
        transform_item_name=None,
        transform_direction=ocio.TRANSFORM_DIR_INVERSE,
    )
    assert ctx.input_color_space == "acescg"
    assert ctx.transform_item_type is ViewModel.__item_type__
    assert ctx.transform_item_name == "sRGB"
    assert ctx.transform_direction == ocio.TRANSFORM_DIR_FORWARD


def test_processor_context_edit_passes_item_fields_through():
    ctx = ViewerTransformBuilder.make_processor_context(
        mode=OCIOViewMode.Edit,
        input_color_space="acescg",
        view="ignored-in-edit",
        transform_item_type=int,
        transform_item_name="MyLook",
        transform_direction=ocio.TRANSFORM_DIR_INVERSE,
    )
    assert ctx.transform_item_type is int
    assert ctx.transform_item_name == "MyLook"
    assert ctx.transform_direction == ocio.TRANSFORM_DIR_INVERSE


def test_make_transform_preview_builds_display_view_transform():
    tf = ViewerTransformBuilder.make_transform(
        mode=OCIOViewMode.Preview,
        display="sRGB - Display",
        view="ACES 1.0 - SDR Video",
        transform_fwd=None,
        transform_inv=None,
        transform_direction=ocio.TRANSFORM_DIR_FORWARD,
    )
    assert isinstance(tf, ocio.DisplayViewTransform)
    assert tf.getDisplay() == "sRGB - Display"
    assert tf.getView() == "ACES 1.0 - SDR Video"


def test_make_transform_preview_without_display_view_is_none():
    tf = ViewerTransformBuilder.make_transform(
        mode=OCIOViewMode.Preview,
        display="",
        view="",
        transform_fwd=None,
        transform_inv=None,
        transform_direction=ocio.TRANSFORM_DIR_FORWARD,
    )
    assert tf is None


def test_make_transform_edit_returns_forward_or_inverse():
    fwd = ocio.ExponentTransform([2.2, 2.2, 2.2, 1.0])
    inv = ocio.ExponentTransform([1.0, 1.0, 1.0, 1.0])

    got_fwd = ViewerTransformBuilder.make_transform(
        mode=OCIOViewMode.Edit,
        display="",
        view="",
        transform_fwd=fwd,
        transform_inv=inv,
        transform_direction=ocio.TRANSFORM_DIR_FORWARD,
    )
    assert got_fwd is fwd

    got_inv = ViewerTransformBuilder.make_transform(
        mode=OCIOViewMode.Edit,
        display="",
        view="",
        transform_fwd=fwd,
        transform_inv=inv,
        transform_direction=ocio.TRANSFORM_DIR_INVERSE,
    )
    assert got_inv is inv


def test_make_transform_edit_without_pair_returns_noop_exponent():
    tf = ViewerTransformBuilder.make_transform(
        mode=OCIOViewMode.Edit,
        display="",
        view="",
        transform_fwd=None,
        transform_inv=None,
        transform_direction=ocio.TRANSFORM_DIR_FORWARD,
    )
    assert isinstance(tf, ocio.ExponentTransform)


def test_default_view_uses_input_color_space_when_given(raw_config):
    # With a basic config the default view for the display is "Raw"; passing an
    # input color space must use the two-arg getDefaultView path without error.
    display = ViewerTransformBuilder.default_display()
    assert ViewerTransformBuilder.default_view(display) == "Raw"
    # Non-empty input color space -> two-arg path (must still return a valid view).
    assert ViewerTransformBuilder.default_view(display, "raw") == "Raw"
    # Empty / None input color space -> one-arg path.
    assert ViewerTransformBuilder.default_view(display, "") == "Raw"
    assert ViewerTransformBuilder.default_view(display, None) == "Raw"


def test_default_color_space_prefers_config_default(raw_config):
    # The raw config's default color space is "raw"; it is scene-referred, so
    # default_color_space() should return it.
    assert ViewerTransformBuilder.default_color_space() == "raw"
