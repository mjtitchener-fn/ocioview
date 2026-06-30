# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL import GL


class ImagePlaneGeometry:
    """
    Owns the image plane's GL geometry: the image texture (bound at GL
    texture unit 0) and the textured-quad VAO/VBOs. The owning widget makes
    its GL context current via the passed-in callable.
    """

    def __init__(self, make_current) -> None:
        self._make_current = make_current
        self.image_tex = None
        self._vao = None
        self._position_vbo = None
        self._tex_coord_vbo = None
        self._index_vbo = None

    def initialize(self, width: float, height: float) -> None:
        """
        Create the image texture (placeholder ``width`` x ``height``) and the
        textured-quad VAO/VBOs. The caller must have made the context current.
        """
        # Init image texture
        self.image_tex = GL.glGenTextures(1)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB32F,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            ctypes.c_void_p(0),
        )

        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )

        # Init image plane geometry
        # fmt: off
        plane_position_data = np.array(
            [
                -0.5,  0.5, 0.0,  # top-left
                 0.5,  0.5, 0.0,  # top-right
                 0.5, -0.5, 0.0,  # bottom-right
                -0.5, -0.5, 0.0,  # bottom-left
            ],
            dtype=np.float32,
        )
        plane_tex_coord_data = np.array(
            [
                0.0, 1.0,  # top-left
                1.0, 1.0,  # top-right
                1.0, 0.0,  # bottom-right
                0.0, 0.0,  # bottom-left
            ],
            dtype=np.float32,
        )
        plane_index_data = np.array(
            [0, 1, 2, 0, 2, 3],  # triangles: top-left, bottom-right
            dtype=np.uint32,
        )
        # fmt: on

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        (
            self._position_vbo,
            self._tex_coord_vbo,
            self._index_vbo,
        ) = GL.glGenBuffers(3)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._position_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            plane_position_data.nbytes,
            plane_position_data,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0)
        )
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._tex_coord_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            plane_tex_coord_data.nbytes,
            plane_tex_coord_data,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._index_vbo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            plane_index_data.nbytes,
            plane_index_data,
            GL.GL_STATIC_DRAW,
        )

    def upload_image(
        self, image_array: np.ndarray, width: int, height: int
    ) -> None:
        """Upload RGB float image data into the image texture."""
        self._make_current()

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB32F,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            image_array.ravel(),
        )

    def draw(self) -> None:
        """Bind the image texture (unit 0) and VAO and draw the quad."""
        GL.glActiveTexture(GL.GL_TEXTURE0 + 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)

        GL.glBindVertexArray(self._vao)

        GL.glDrawElements(
            GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        GL.glBindVertexArray(0)

    def delete_gl(self) -> None:
        """Delete the image texture and the plane VAO/VBOs."""
        if self.image_tex is not None:
            GL.glDeleteTextures([self.image_tex])
            self.image_tex = None
        if self._vao is not None:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        vbos = [
            vbo
            for vbo in (
                self._position_vbo,
                self._tex_coord_vbo,
                self._index_vbo,
            )
            if vbo is not None
        ]
        if vbos:
            GL.glDeleteBuffers(len(vbos), vbos)
            self._position_vbo = None
            self._tex_coord_vbo = None
            self._index_vbo = None
