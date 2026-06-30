# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

import numpy as np

from ocioview.viewer.camera import PanZoomCamera2D


def test_initial_state_is_identity():
    cam = PanZoomCamera2D()
    np.testing.assert_allclose(cam.model_view_mat, np.eye(4))
    np.testing.assert_allclose(cam.proj_mat, np.eye(4))
    assert cam.image_scale == 1.0


def test_set_image_size_updates_model_view():
    cam = PanZoomCamera2D()
    cam.set_image_size(10, 20)
    np.testing.assert_allclose(cam.image_size, [10, 20])
    assert cam.model_view_mat[0, 0] == 10
    assert cam.model_view_mat[1, 1] == -20


def test_resize_builds_orthographic_projection():
    cam = PanZoomCamera2D()
    cam.resize(200, 100)
    assert cam.proj_mat[0, 0] == 0.01
    assert cam.proj_mat[1, 1] == 0.02


def test_relative_pan_translates_by_offset_over_scale():
    cam = PanZoomCamera2D()
    cam.set_image_size(10, 10)
    cam.image_scale = 2.0
    cam.pan(np.array([4.0, 2.0]))
    np.testing.assert_allclose(cam.image_pos, [2.0, 1.0])


def test_absolute_pan_sets_position():
    cam = PanZoomCamera2D()
    cam.set_image_size(10, 10)
    cam.image_scale = 2.0
    cam.pan(np.array([4.0, 2.0]), absolute=True)
    np.testing.assert_allclose(cam.image_pos, [2.0, 1.0])


def test_fit_sets_scale_and_centers():
    cam = PanZoomCamera2D()
    cam.set_image_size(100, 50)
    cam.resize(200, 100)
    cam.fit()
    assert cam.image_scale == 2.0
    np.testing.assert_allclose(cam.image_pos, [0.0, 0.0])


def test_zoom_about_point_keeps_that_point_fixed():
    cam = PanZoomCamera2D()
    cam.set_image_size(100, 100)
    cam.resize(200, 200)
    cam.image_scale = 1.0
    cam._update_model_view()
    offset = np.array([30.0, -20.0])
    before = cam.image_pos.copy()
    cam.zoom(offset, 2.0, absolute=True)
    expected = before - offset / 1.0 + offset / 2.0
    np.testing.assert_allclose(cam.image_pos, expected)
    assert cam.image_scale == 2.0


def test_screen_to_image_round_trips_center():
    cam = PanZoomCamera2D()
    cam.set_image_size(100, 100)
    cam.resize(200, 200)
    cam.image_scale = 1.0
    cam.image_pos = np.array([0.0, 0.0])
    cam._update_model_view()
    pixel = cam.screen_to_image(100.0, 100.0, 200, 200)
    np.testing.assert_allclose(pixel, [50.0, 50.0], atol=1.0)
