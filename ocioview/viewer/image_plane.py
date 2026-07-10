# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

# TODO: Much of the OpenGL code in this module is adapted from the
#       oglapphelpers library bundled with OCIO. We should fully
#       reimplement that in Python for direct use in applications.

from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
from OpenGL import GL

import PyOpenColorIO as ocio
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

from ..log_handlers import message_queue
from ..processor_context import ProcessorContext
from ..ref_space_manager import ReferenceSpaceManager
from .utils import load_image
from .camera import PanZoomCamera2D
from .geometry import ImagePlaneGeometry
from .ocio_pipeline import (
    OCIOGpuPipeline,
    build_viewing_pipelines,
    next_channel_hot,
)

logger = logging.getLogger(__name__)


GLSL_VERT_SRC = """#version 400 core

uniform mat4 mvpMat;
in vec3 in_position;
in vec2 in_texCoord;

out vec2 vert_texCoord;

void main() {
    vert_texCoord = in_texCoord;
    gl_Position = mvpMat * vec4(in_position, 1.0);
}

"""
"""
Simple vertex shader which transforms all vertices with a 
model-view-projection matrix uniform.
"""

GLSL_FRAG_SRC = """#version 400 core

uniform sampler2D imageTex;
in vec2 vert_texCoord;

out vec4 frag_color;

void main() {{
    frag_color = texture(imageTex, vert_texCoord);
}}
"""
"""
Simple fragment shader which performs a 2D texture lookup to map an 
image texture onto UVs. This is used when OCIO is unavailable, like 
before its shader initialization.
"""

GLSL_FRAG_OCIO_SRC_FMT = """#version 400 core

uniform sampler2D imageTex;
in vec2 vert_texCoord;

out vec4 frag_color;

{ocio_src}

void main() {{
    vec4 inColor = texture(imageTex, vert_texCoord);
    vec4 outColor = OCIOMain(inColor);
    frag_color = outColor;
}}
"""
"""
Fragment shader which performs a 2D texture lookup to map an image 
texture onto UVs and processes fragments through an OCIO-provided 
shader program segment, which itself utilizes additional texture 
lookups, dynamic property uniforms, and various native GLSL op 
implementations. Note that this shader's cost will increase with 
additional LUTs in an OCIO processor, since each adds its own 
2D or 3D texture.
"""


class ImagePlane(QtOpenGLWidgets.QOpenGLWidget):
    """
    Qt-wrapped OpenGL window for drawing with PyOpenGL.
    """

    image_loaded = QtCore.Signal(Path, int, int)
    sample_changed = QtCore.Signal(
        int, int, float, float, float, float, float, float
    )
    scale_changed = QtCore.Signal(float)
    tf_subscription_requested = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(QtCore.QSize(10, 10))

        # Clicking on/tabbing to widget restores focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

        # Set to True after initializeGL is called. Don't allow grabbing
        # OpenGL context until that point.
        self._gl_ready = False

        # Color management
        self._ocio_tf = None
        self._ocio_exposure = 0.0
        self._ocio_gamma = 1.0
        self._ocio_channel_hot = [1, 1, 1, 1]
        self._ocio_proc_context = ProcessorContext()
        self._ocio_proc = None
        self._ocio_proc_cpu = None
        self._ocio_proc_cache_id = None
        self._ocio_shader_cache_id = None
        self.ocio_pipeline = OCIOGpuPipeline(self.makeCurrent)

        # Keyboard shortcuts
        self._shortcuts = []

        # Mouse info
        self._mouse_pressed = False
        self._mouse_last_pos = QtCore.QPointF()

        # Image texture
        self._image_array = None

        # 2D pan/zoom/fit camera (owns image pos/size/scale + matrices)
        self.camera = PanZoomCamera2D()

        # Image plane GL geometry (image texture + textured-quad VAO/VBOs)
        self.geometry = ImagePlaneGeometry(self.makeCurrent)

        # GLSL shader program
        self._vert_shader = None
        self._frag_shader = None
        self._shader_program = None

        # Setup keyboard shortcuts
        self._install_shortcuts()

    def initializeGL(self) -> None:
        """
        Set up OpenGL resources and state (called once).
        """
        self._gl_ready = True

        self.makeCurrent()

        # Drain any stale GL errors left by PySide6 context initialization
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass

        self.geometry.initialize(
            self.camera.image_size[0], self.camera.image_size[1]
        )

        self._build_program()

        # Free GL resources when the context is destroyed (e.g. tab close)
        self.context().aboutToBeDestroyed.connect(self.cleanupGL)

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the widget is resized.

        :param w: Window width (logical pixels)
        :param h: Window height (logical pixels)
        """
        self.makeCurrent()

        # resizeGL receives logical pixels but the framebuffer is device-sized;
        # work in device pixels so rendering fills it and stays crisp on HiDPI.
        dpr = self.devicePixelRatioF()
        device_w = round(w * dpr)
        device_h = round(h * dpr)

        GL.glViewport(0, 0, device_w, device_h)

        self.camera.resize(device_w, device_h)
        self._refresh_tex_interp()

    def paintGL(self) -> None:
        """
        Called whenever a repaint is needed. Calling ``update()`` will
        schedule a repaint.
        """
        self.makeCurrent()

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self._shader_program is not None:
            GL.glUseProgram(self._shader_program)

            self.ocio_pipeline.use_textures(self._shader_program)
            self.ocio_pipeline.use_uniforms(self._shader_program)

            # Set uniforms
            mvp_mat = self.camera.mvp()
            mvp_mat_loc = GL.glGetUniformLocation(
                self._shader_program, "mvpMat"
            )
            GL.glUniformMatrix4fv(mvp_mat_loc, 1, GL.GL_FALSE, mvp_mat.T)

            image_tex_loc = GL.glGetUniformLocation(
                self._shader_program, "imageTex"
            )
            GL.glUniform1i(image_tex_loc, 0)

            # Bind texture, VAO, and draw
            self.geometry.draw()

    def load_image(self, image_path: Path) -> None:
        """
        Load an image into the image plane texture.

        :param image_path: Image file path
        """
        config = ocio.GetCurrentConfig()

        # Get input color space (file rule)
        color_space_name, rule_idx = config.getColorSpaceFromFilepath(
            image_path.as_posix()
        )
        if not color_space_name:
            # Use previous or config default
            if self._ocio_proc_context:
                color_space_name = self._ocio_proc_context.input_color_space
            else:
                color_space_name = ocio.ROLE_DEFAULT

        if self._ocio_proc_context:
            proc_context = ProcessorContext(
                color_space_name,
                self._ocio_proc_context.transform_item_type,
                self._ocio_proc_context.transform_item_name,
                self._ocio_proc_context.transform_direction,
            )
        else:
            proc_context = ProcessorContext(color_space_name)

        # Load image data via an available image library
        self._image_array = load_image(image_path)

        width = self._image_array.shape[1]
        height = self._image_array.shape[0]

        # Stash image size for pan/zoom calculations
        self.camera.set_image_size(width, height)

        # Load image data into texture
        self.geometry.upload_image(self._image_array, width, height)

        self.image_loaded.emit(image_path, width, height)

        self.update_ocio_proc(proc_context=proc_context)
        self.fit()

        # Log image change after load and render
        self.broadcast_image()

    def broadcast_image(self) -> None:
        """
        Broadcast current image array, if one is loaded, through the
        message queue for other app components.
        """
        if self._image_array is not None:
            message_queue.put_nowait(self._image_array)

    def input_color_space(self) -> str | None:
        """
        :return: Current input OCIO color space name
        """
        if self._ocio_proc_context:
            return self._ocio_proc_context.input_color_space
        else:
            return None

    def transform(self) -> Optional[ocio.Transform]:
        """
        :return: Current OCIO transform
        """
        return self._ocio_tf

    def clear_transform(self) -> None:
        """
        Clear current OCIO transform, passing through the input image.
        """

        self._ocio_tf = None

        input_color_space = (
            self._ocio_proc_context.input_color_space
            if self._ocio_proc_context
            else None
        )
        self.update_ocio_proc(
            ProcessorContext(input_color_space),
            force_update=True,
        )

    def reset_ocio_proc(self, update: bool = False) -> None:
        """
        Reset the OCIO GPU renderer to a passthrough state.

        :param update: Whether to redraw viewport
        """
        self._ocio_proc_context = None
        self._ocio_tf = None
        self._ocio_exposure = 0.0
        self._ocio_gamma = 1.0
        self._ocio_channel_hot = [1, 1, 1, 1]

        if update:
            self.update_ocio_proc(force_update=True)

    def update_ocio_proc(
        self,
        proc_context: Optional[ProcessorContext] = None,
        transform: Optional[ocio.Transform] = None,
        channel: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        """
        Update one or more aspects of the OCIO GPU renderer. Parameters
        are cached, so not providing a parameter maintains the existing
        state. This will trigger a GL update IF the underlying OCIO ops
        in the processor have changed.

        :param proc_context: Processor context data
        :param transform: Optional main OCIO transform, to be applied
            from the current config's scene reference space.
        :param channel: ImagePlaneChannels value to toggle channel
            isolation.
        :param force_update: Set to True to update the viewport even
            when the processor has not been updated.
        """

        # Update processor parameters
        if proc_context is not None:
            self._ocio_proc_context = proc_context
        if transform is not None:
            self._ocio_tf = transform
        if channel is not None:
            self._update_ocio_channel_hot(channel)

        config = ocio.GetCurrentConfig()
        input_color_space = (
            self._ocio_proc_context.input_color_space
            if self._ocio_proc_context
            else None
        )
        scene_ref_name = (
            ReferenceSpaceManager.scene_reference_space().getName()
        )

        gpu_viewing_pipeline, cpu_viewing_pipeline = build_viewing_pipelines(
            config=config,
            input_color_space=input_color_space,
            transform=self._ocio_tf,
            exposure=self._ocio_exposure,
            gamma=self._ocio_gamma,
            channel_hot=self._ocio_channel_hot,
            scene_ref_name=scene_ref_name,
        )

        # Create GPU processor
        try:
            gpu_proc = config.getProcessor(
                gpu_viewing_pipeline, ocio.TRANSFORM_DIR_FORWARD
            )
        except ocio.Exception:
            # Config may have changed between transform creation and now. If this
            # doesn't error, CPU processor construction should succeed.
            return

        if gpu_proc.getCacheID() != self._ocio_proc_cache_id:
            # Update CPU processor
            cpu_proc = config.getProcessor(
                cpu_viewing_pipeline, ocio.TRANSFORM_DIR_FORWARD
            )
            self._ocio_proc = cpu_proc
            self._ocio_proc_cpu = cpu_proc.getDefaultCPUProcessor()

            # Update GPU processor shaders and textures
            shader_desc = ocio.GpuShaderDesc.CreateShaderDesc(
                language=ocio.GPU_LANGUAGE_GLSL_4_0
            )
            self._ocio_proc_cache_id = gpu_proc.getCacheID()
            ocio_gpu_proc = gpu_proc.getDefaultGPUProcessor()
            ocio_gpu_proc.extractGpuShaderInfo(shader_desc)

            self.ocio_pipeline.set_shader_desc(shader_desc)
            self.ocio_pipeline.allocate_textures()
            self._build_program()

            # Set initial dynamic property state
            self.ocio_pipeline.update_dynamic_property(
                ocio.DYNAMIC_PROPERTY_EXPOSURE, self._ocio_exposure
            )
            self.ocio_pipeline.update_dynamic_property(
                ocio.DYNAMIC_PROPERTY_GAMMA, self._ocio_gamma
            )

            self.update()

            # Log processor change after render
            message_queue.put_nowait(
                (self._ocio_proc_context, self._ocio_proc)
            )

        elif force_update:
            self.update()

            # The transform and processor has not changed, but other app components
            # which view it may have dropped the reference. Log processor to update
            # them as needed.
            if (
                self._ocio_proc is not None
                and self._ocio_proc_context is not None
            ):
                message_queue.put_nowait(
                    (self._ocio_proc_context, self._ocio_proc)
                )

    def exposure(self) -> float:
        """
        :return: Last set exposure dynamic property value
        """
        return self._ocio_exposure

    def update_exposure(self, value: float) -> None:
        """
        Update OCIO GPU renderer exposure. This is a dynamic property,
        implemented as a GLSL uniform, so can be updated without
        modifying the OCIO shader program or its dependencies.

        :param value: Exposure value in stops
        """
        self._ocio_exposure = value
        self.ocio_pipeline.update_dynamic_property(
            ocio.DYNAMIC_PROPERTY_EXPOSURE, value
        )
        self.update()

    def gamma(self) -> float:
        """
        :return: Last set gamma dynamic property value
        """
        return self._ocio_gamma

    def update_gamma(self, value: float) -> None:
        """
        Update OCIO GPU renderer gamma. This is a dynamic property,
        implemented as a GLSL uniform, so can be updated without
        modifying the OCIO shader program or its dependencies.

        .. note::
            Value is floor clamped at 0.001 to prevent zero division
            errors.

        :param value: Gamma value used like: pow(rgb, 1/gamma)
        """
        # Translate gamma to exponent, enforcing floor
        value = 1.0 / max(0.001, value)

        self._ocio_gamma = value
        self.ocio_pipeline.update_dynamic_property(
            ocio.DYNAMIC_PROPERTY_GAMMA, value
        )
        self.update()

    def enterEvent(self, event: QtCore.QEvent) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(True)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._mouse_pressed = True
        self._mouse_last_pos = event.position()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position()
        dpr = self.devicePixelRatioF()

        if self._mouse_pressed:
            offset = np.array([*(pos - self._mouse_last_pos).toTuple()]) * dpr
            self._mouse_last_pos = pos

            self.pan(offset, update=True)
        else:
            widget_w = self.width() * dpr
            widget_h = self.height() * dpr

            # Trace mouse position through the inverse MVP matrix to update
            # the sampled pixel (device pixels, matching the framebuffer).
            pixel_pos = self.camera.screen_to_image(
                pos.x() * dpr, pos.y() * dpr, widget_w, widget_h
            )

            # Broadcast sample position
            if (
                self._image_array is not None
                and 0 <= pixel_pos[0] < self.camera.image_size[0]
                and 0 <= pixel_pos[1] < self.camera.image_size[1]
            ):
                pixel_x = math.floor(pixel_pos[0])
                pixel_y = math.floor(pixel_pos[1])
                pixel_input = list(self._image_array[pixel_y, pixel_x])
                if len(pixel_input) < 3:
                    pixel_input += [0.0] * (3 - len(pixel_input))
                elif len(pixel_input) > 3:
                    pixel_input = pixel_input[:3]

                # Sample output pixel with CPU processor
                if self._ocio_proc_cpu is not None:
                    pixel_output = self._ocio_proc_cpu.applyRGB(pixel_input)
                else:
                    pixel_output = pixel_input.copy()

                self.sample_changed.emit(
                    pixel_x, pixel_y, *pixel_input, *pixel_output
                )
            else:
                # Out of image bounds
                self.sample_changed.emit(-1, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._mouse_pressed = False

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        dpr = self.devicePixelRatioF()
        w, h = self.width() * dpr, self.height() * dpr

        # Fit image to frame
        if h > w:
            min_scale = w / self.camera.image_size[0]
        else:
            min_scale = h / self.camera.image_size[1]

        # Fill frame with 1 pixel with 0.5 pixel overscan
        max_scale = max(w, h) * 1.5

        # Wheel away from the user (positive angleDelta) zooms in.
        delta = event.angleDelta().y() / 360.0 * self.camera.image_scale
        scale = min(max_scale, max(min_scale, self.camera.image_scale + delta))

        self.zoom(event.position(), scale, update=True, absolute=True)

    def pan(
        self, offset: np.ndarray, update: bool = True, absolute: bool = False
    ) -> None:
        """
        Pan the viewport by the specified offset in screen space.

        :param offset: Offset in pixels
        :param update: Whether to redraw the viewport
        :param absolute: When True, offset is an absolute position to
            translate the viewport from its origin.
        """
        self.camera.pan(offset, absolute=absolute)
        self._refresh_tex_interp()
        if update:
            self.update()

    def zoom(
        self,
        point: QtCore.QPoint,
        amount: float,
        update: bool = True,
        absolute: bool = False,
    ) -> None:
        """
        Zoom the viewport by the specified scale amount, centered on a point.

        :param point: Viewport position to center zoom on (logical pixels)
        :param amount: Zoom scale amount
        :param update: Whether to redraw the viewport
        :param absolute: When True, amount is an absolute scale to set.
        """
        dpr = self.devicePixelRatioF()
        center_offset = (
            np.array([*(point - self.rect().center()).toTuple()]) * dpr
        )
        self.camera.zoom(center_offset, amount, absolute=absolute)
        self._refresh_tex_interp()
        if update:
            self.update()

        if self._image_array is not None:
            self.scale_changed.emit(self.camera.image_scale / dpr)

    def fit(self, update: bool = True) -> None:
        """
        Pan and zoom so the image fits within the viewport and is centered.

        :param update: Whether to redraw the viewport
        """
        self.camera.fit()
        self._refresh_tex_interp()
        if update:
            self.update()

        if self._image_array is not None:
            self.scale_changed.emit(
                self.camera.image_scale / self.devicePixelRatioF()
            )

    def _install_shortcuts(self) -> None:
        """
        Setup supported keyboard shortcuts.
        """
        # R,G,B,A = view channel
        # C = view color
        for i, key in enumerate(("R", "G", "B", "A", "C")):
            channel_shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            channel_shortcut.activated.connect(
                partial(self.update_ocio_proc, channel=i)
            )
            self._shortcuts.append(channel_shortcut)

        # Number keys = Subscribe to transform @ slot
        for i in range(10):
            subscribe_shortcut = QtGui.QShortcut(
                QtGui.QKeySequence(str(i)), self
            )
            subscribe_shortcut.activated.connect(
                lambda slot=i: self.tf_subscription_requested.emit(slot)
            )
            self._shortcuts.append(subscribe_shortcut)

        # Ctrl + Number keys = Power of 2 scale: 1 = x1, 2 = x2, 3 = x4, ...
        for i in range(9):
            scale_shortcut = QtGui.QShortcut(
                QtGui.QKeySequence(f"Ctrl+{i + 1}"), self
            )
            scale_shortcut.activated.connect(
                lambda exponent=i: self.zoom(
                    self.rect().center(), float(2**exponent), absolute=True
                )
            )
            self._shortcuts.append(scale_shortcut)

        # F = fit image to viewport
        fit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F"), self)
        fit_shortcut.activated.connect(self.fit)
        self._shortcuts.append(fit_shortcut)

    def _compile_shader(
        self, glsl_src: str, shader_type: GL.GLenum
    ) -> Optional[GL.GLuint]:
        """
        Compile GLSL shader and return its object ID.

        :param glsl_src: Shader source code
        :param shader_type: Type of shader to be created, which is an
            enum adhering to the formatting ``GL_*_SHADER``.
        :return: Shader object ID, or None if shader compilation fails
        """
        self.makeCurrent()

        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, glsl_src)
        GL.glCompileShader(shader)

        compile_status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if not compile_status:
            compile_log = GL.glGetShaderInfoLog(shader)
            logger.error(
                "Shader program compile error: {log}".format(log=compile_log)
            )
            return None

        return shader

    def _build_program(self, force: bool = False) -> None:
        """
        This builds the initial shader program, and rebuilds its
        fragment shader whenever the OCIO GPU renderer changes.

        :param force: Whether to force a rebuild even if the OCIO
            shader cache ID has not changed.
        """
        if not self._gl_ready:
            return

        self.makeCurrent()

        # If new shader cache ID matches previous cache ID, existing program
        # can be reused.
        shader_cache_id = self._ocio_shader_cache_id
        if self.ocio_pipeline.has_shader_desc() and not force:
            shader_cache_id = self.ocio_pipeline.shader_cache_id()
            if self._ocio_shader_cache_id == shader_cache_id:
                return

        # Init shader program
        if not self._shader_program:
            self._shader_program = GL.glCreateProgram()

        # Vert shader only needs to be built once
        if not self._vert_shader:
            self._vert_shader = self._compile_shader(
                GLSL_VERT_SRC, GL.GL_VERTEX_SHADER
            )
            if not self._vert_shader:
                return

            GL.glAttachShader(self._shader_program, self._vert_shader)

        # Frag shader needs recompile each build (for OCIO changes)
        if self._frag_shader:
            GL.glDetachShader(self._shader_program, self._frag_shader)
            GL.glDeleteShader(self._frag_shader)

        frag_src = GLSL_FRAG_SRC
        if self.ocio_pipeline.has_shader_desc():
            # Inject OCIO shader block
            frag_src = GLSL_FRAG_OCIO_SRC_FMT.format(
                ocio_src=self.ocio_pipeline.shader_text()
            )
        self._frag_shader = self._compile_shader(
            frag_src, GL.GL_FRAGMENT_SHADER
        )
        if not self._frag_shader:
            return

        GL.glAttachShader(self._shader_program, self._frag_shader)

        # Link program
        GL.glBindAttribLocation(self._shader_program, 0, "in_position")
        GL.glBindAttribLocation(self._shader_program, 1, "in_texCoord")

        GL.glLinkProgram(self._shader_program)
        link_status = GL.glGetProgramiv(
            self._shader_program, GL.GL_LINK_STATUS
        )
        if not link_status:
            link_log = GL.glGetProgramInfoLog(self._shader_program)
            logger.error(
                "Shader program link error: {log}".format(log=link_log)
            )
            return

        # Store cache ID to detect reuse
        self._ocio_shader_cache_id = shader_cache_id

    def _refresh_tex_interp(self) -> None:
        """
        Use nearest interpolation when zoomed in past 1:1 so pixels are
        crisp; linear otherwise.
        """
        self.geometry.set_image_interp(self.camera.image_scale > 1.0)

    def cleanupGL(self) -> None:
        """
        Delete GL resources when the context is about to be destroyed
        (e.g. on viewer tab close), preventing GPU resource leaks.
        """
        if not self._gl_ready:
            return

        self.makeCurrent()

        self.ocio_pipeline.delete_gl()

        self.geometry.delete_gl()

        if self._shader_program is not None:
            GL.glDeleteProgram(self._shader_program)
            self._shader_program = None

        self._gl_ready = False
        self.doneCurrent()

    def _update_ocio_channel_hot(self, channel: int) -> None:
        """
        Update the OCIO GPU renderer's channel view to either isolate a
        specific channel or show them all.

        :param channel: ImagePlaneChannels value to toggle channel
            isolation.
        """
        self._ocio_channel_hot = next_channel_hot(
            self._ocio_channel_hot, channel
        )
