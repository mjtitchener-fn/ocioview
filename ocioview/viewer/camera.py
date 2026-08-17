# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from __future__ import annotations

import numpy as np

from .utils import model_view_matrix, orthographic_proj_matrix


class PanZoomCamera2D:
    """
    2D pan/zoom/fit camera for an image plane. Pure math (no GL or Qt): it
    owns the image position/size/scale and produces the model-view and
    projection matrices the renderer consumes. Viewport geometry and cursor
    offsets are passed in by the widget so this class stays testable in
    isolation.
    """

    def __init__(self) -> None:
        self.image_pos = np.array([0.0, 0.0])
        self.image_size = np.array([1.0, 1.0])
        self.image_scale = 1.0
        self.model_view_mat = np.eye(4)
        self.proj_mat = np.eye(4)

    def set_image_size(self, width: float, height: float) -> None:
        """
        Set the loaded image's pixel size and reset position. (The original
        widget set pos to [0, 1] here, but load_image always calls fit()
        right after, which resets to [0, 0]; reset to [0, 0] directly.)
        """
        self.image_pos = np.array([0.0, 0.0])
        self.image_size = np.array([width, height], dtype=np.float64)
        self._update_model_view()

    def resize(self, width: int, height: int) -> None:
        """Rebuild the orthographic projection for a viewport size."""
        # fmt: off
        self.proj_mat = orthographic_proj_matrix(
            -1.0,             # near
             1.0,             # far
            -width / 2.0,     # left
             width / 2.0,     # right
             height / 2.0,    # top
            -height / 2.0,    # bottom
        )
        # fmt: on
        self._update_model_view()

    def pan(self, offset: np.ndarray, absolute: bool = False) -> None:
        """
        Pan by a screen-space offset (pixels). Absolute treats the offset as
        a position to translate from the origin.
        """
        if self.image_scale > 0:
            if absolute:
                self.image_pos = offset / self.image_scale
            else:
                self.image_pos = self.image_pos + offset / self.image_scale
        self._update_model_view()

    def zoom(
        self, center_offset: np.ndarray, amount: float, absolute: bool = False
    ) -> None:
        """
        Zoom about a point given as a screen-space offset from the viewport
        center, keeping that point fixed. Absolute sets the scale directly.
        """
        self.pan(-center_offset)
        if absolute:
            self.image_scale = amount
        else:
            self.image_scale += amount
        self._update_model_view()
        self.pan(center_offset)

    def fit(self) -> None:
        """Scale so the image fits the current viewport, centered."""
        width, height = self.proj_viewport()
        if height > width:
            self.image_scale = width / self.image_size[0]
        else:
            self.image_scale = height / self.image_size[1]
        self.image_pos = np.array([0.0, 0.0])
        self._update_model_view()

    def proj_viewport(self) -> tuple[float, float]:
        """
        Recover (width, height) from the projection matrix (a=2/width,
        b=2/height), so fit() does not need the size passed again.
        """
        a = self.proj_mat[0, 0]
        b = self.proj_mat[1, 1]
        width = 2.0 / a if a else 0.0
        height = 2.0 / b if b else 0.0
        return width, height

    def mvp(self) -> np.ndarray:
        """Combined projection * model-view matrix."""
        return self.proj_mat @ self.model_view_mat

    def screen_to_image(
        self, x: float, y: float, widget_w: int, widget_h: int
    ) -> np.ndarray:
        """
        Map a widget-space cursor position to image pixel coordinates by
        tracing through the inverse MVP matrix.
        """
        screen_pos = np.array(
            [
                x / widget_w * 2.0 - 1.0,
                (widget_h - y - 1) / widget_h * 2.0 - 1.0,
                0.0,
                1.0,
            ]
        )
        model_pos = np.linalg.inv(self.mvp()) @ screen_pos
        return (
            np.array([model_pos[0] + 0.5, model_pos[1] + 0.5])
            * self.image_size
        )

    def _update_model_view(self) -> None:
        self.model_view_mat = model_view_matrix(
            self.image_scale, self.image_pos, self.image_size
        )
