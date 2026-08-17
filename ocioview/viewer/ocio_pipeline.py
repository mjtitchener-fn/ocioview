# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

from __future__ import annotations

from typing import Any, Optional

from OpenGL import GL

import PyOpenColorIO as ocio


def build_viewing_pipelines(
    config: ocio.Config,
    input_color_space: str | None,
    transform: Optional[ocio.Transform],
    exposure: float,
    gamma: float,
    channel_hot: list[int],
    scene_ref_name: str,
) -> tuple[ocio.GroupTransform, ocio.GroupTransform]:
    """
    Build the GPU and CPU viewing pipelines for the image plane.

    The GPU pipeline is for viewport rendering (with dynamic exposure, a
    channel-isolation matrix, and dynamic gamma). The CPU pipeline mirrors it
    for pixel sampling but omits those viewport-only adjustments. Both are
    built relative to the current config's scene reference space.

    :param config: Current OCIO config
    :param input_color_space: Input color space name, or None/empty if unknown
    :param transform: Main viewing transform (scene-reference input), or None
    :param exposure: Dynamic exposure value (stops)
    :param gamma: Dynamic gamma exponent value
    :param channel_hot: 4-element channel-isolation flags
    :param scene_ref_name: Scene reference space name
    :return: (gpu_pipeline, cpu_pipeline)
    """
    has_scene_linear = config.hasRole(ocio.ROLE_SCENE_LINEAR)

    gpu_viewing_pipeline = ocio.GroupTransform()
    cpu_viewing_pipeline = ocio.GroupTransform()

    # Convert to scene linear space if input space is known
    if has_scene_linear and input_color_space:
        to_scene_linear = ocio.ColorSpaceTransform(
            src=input_color_space, dst=ocio.ROLE_SCENE_LINEAR
        )
        gpu_viewing_pipeline.appendTransform(to_scene_linear)
        cpu_viewing_pipeline.appendTransform(to_scene_linear)

    # Dynamic exposure adjustment (GPU only)
    gpu_viewing_pipeline.appendTransform(
        ocio.ExposureContrastTransform(exposure=exposure, dynamicExposure=True)
    )

    # Convert to the scene reference space (expected input for all transforms).
    if input_color_space:
        if has_scene_linear:
            to_scene_ref = ocio.ColorSpaceTransform(
                src=ocio.ROLE_SCENE_LINEAR, dst=scene_ref_name
            )
        else:
            to_scene_ref = ocio.ColorSpaceTransform(
                src=input_color_space, dst=scene_ref_name
            )
        gpu_viewing_pipeline.appendTransform(to_scene_ref)
        cpu_viewing_pipeline.appendTransform(to_scene_ref)

    # Main transform, or restore the input color space if known
    if transform is not None:
        gpu_viewing_pipeline.appendTransform(transform)
        cpu_viewing_pipeline.appendTransform(transform)
    elif input_color_space:
        from_scene_ref = ocio.ColorSpaceTransform(
            src=scene_ref_name, dst=input_color_space
        )
        gpu_viewing_pipeline.appendTransform(from_scene_ref)
        cpu_viewing_pipeline.appendTransform(from_scene_ref)

    # Channel view (GPU only)
    gpu_viewing_pipeline.appendTransform(
        ocio.MatrixTransform.View(
            channelHot=channel_hot,
            lumaCoef=config.getDefaultLumaCoefs(),
        )
    )

    # Dynamic gamma adjustment (GPU only)
    gpu_viewing_pipeline.appendTransform(
        ocio.ExposureContrastTransform(
            gamma=gamma, pivot=1.0, dynamicGamma=True
        )
    )

    return gpu_viewing_pipeline, cpu_viewing_pipeline


def next_channel_hot(current: list[int], channel: int) -> list[int]:
    """
    Compute the next channel-isolation state. The first three entries are the
    R/G/B hot flags (the 4th is alpha, left untouched).

    If ``channel`` is in 0..2 and we are currently showing all channels or a
    channel other than ``channel``, isolate that channel. Otherwise show all.
    """
    channel_hot = list(current)
    if channel < 3 and (all(channel_hot) or not channel_hot[channel]):
        for i in range(3):
            channel_hot[i] = 1 if i == channel else 0
    else:
        for i in range(3):
            channel_hot[i] = 1
    return channel_hot


def set_texture_interp(
    tex_type: GL.GLenum, interpolation: ocio.Interpolation
) -> None:
    """
    Set min/mag filtering for a texture based on an OCIO interpolation enum.
    The caller must have made the GL context current.
    """
    if interpolation == ocio.INTERP_NEAREST:
        GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    else:
        GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)


class OCIOGpuPipeline:
    """
    Owns the OCIO GPU resources for an image plane: the shader description,
    LUT textures (bound starting at GL texture unit 1, after the image
    texture at unit 0), and the dynamic-property uniforms. The owning widget
    keeps the shader program and feeds it this pipeline's shader text +
    bound resources.
    """

    def __init__(self, make_current) -> None:
        """
        :param make_current: Callable that makes the owning widget's GL
            context current (e.g. ``widget.makeCurrent``).
        """
        self._make_current = make_current
        self._shader_desc = None
        self._tex_start_index = 1  # Start after the image texture (unit 0)
        self._tex_ids = []
        self._uniform_ids = {}

    def set_shader_desc(self, shader_desc) -> None:
        """Set the current OCIO GPU shader description."""
        self._shader_desc = shader_desc

    def has_shader_desc(self) -> bool:
        """Whether a shader description is currently set."""
        return bool(self._shader_desc)

    def shader_cache_id(self) -> str:
        """OCIO shader cache id of the current shader description."""
        return self._shader_desc.getCacheID()

    def shader_text(self) -> str:
        """OCIO-generated GLSL shader text to inject into the fragment shader."""
        return self._shader_desc.getShaderText()

    def allocate_textures(self) -> None:
        """
        Iterate and allocate 1/2/3D textures needed by the current
        OCIO GPU processor. 3D LUTs become 3D textures and 1D LUTs
        become 1D or 2D textures depending on their size. Since
        textures have a hardware enforced width limitation, large LUTs
        are wrapped onto multiple rows.

        .. note::
            Each time this runs, the previous set of textures are
            deleted from GPU memory first.
        """
        if not self._shader_desc:
            return

        self._make_current()

        # Delete previous textures
        self.delete_textures()
        self.clear_uniforms()

        tex_index = self._tex_start_index

        # Process 3D textures
        for tex_info in self._shader_desc.get3DTextures():
            tex_data = tex_info.getValues()

            tex = GL.glGenTextures(1)
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)
            GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
            set_texture_interp(GL.GL_TEXTURE_3D, tex_info.interpolation)
            GL.glTexImage3D(
                GL.GL_TEXTURE_3D,
                0,
                GL.GL_RGB32F,
                tex_info.edgeLen,
                tex_info.edgeLen,
                tex_info.edgeLen,
                0,
                GL.GL_RGB,
                GL.GL_FLOAT,
                tex_data,
            )

            self._tex_ids.append(
                (
                    tex,
                    tex_info.textureName,
                    tex_info.samplerName,
                    GL.GL_TEXTURE_3D,
                    tex_index,
                )
            )
            tex_index += 1

        # Process 2D textures
        for tex_info in self._shader_desc.getTextures():
            tex_data = tex_info.getValues()

            internal_fmt = GL.GL_RGB32F
            fmt = GL.GL_RGB
            if tex_info.channel == self._shader_desc.TEXTURE_RED_CHANNEL:
                internal_fmt = GL.GL_R32F
                fmt = GL.GL_RED

            tex = GL.glGenTextures(1)
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)

            if tex_info.height > 1:
                tex_type = GL.GL_TEXTURE_2D
                GL.glBindTexture(tex_type, tex)
                set_texture_interp(tex_type, tex_info.interpolation)
                GL.glTexImage2D(
                    tex_type,
                    0,
                    internal_fmt,
                    tex_info.width,
                    tex_info.height,
                    0,
                    fmt,
                    GL.GL_FLOAT,
                    tex_data,
                )
            else:
                tex_type = GL.GL_TEXTURE_1D
                GL.glBindTexture(tex_type, tex)
                set_texture_interp(tex_type, tex_info.interpolation)
                GL.glTexImage1D(
                    tex_type,
                    0,
                    internal_fmt,
                    tex_info.width,
                    0,
                    fmt,
                    GL.GL_FLOAT,
                    tex_data,
                )

            self._tex_ids.append(
                (
                    tex,
                    tex_info.textureName,
                    tex_info.samplerName,
                    tex_type,
                    tex_index,
                )
            )
            tex_index += 1

    def delete_textures(self) -> None:
        """
        Delete all OCIO textures from the GPU.
        """
        self._make_current()

        for (
            tex,
            tex_name,
            sampler_name,
            tex_type,
            tex_index,
        ) in self._tex_ids:
            GL.glDeleteTextures([tex])
        del self._tex_ids[:]

    def use_textures(self, program) -> None:
        """
        Bind all OCIO textures to the shader program.
        """
        self._make_current()

        for (
            tex,
            tex_name,
            sampler_name,
            tex_type,
            tex_index,
        ) in self._tex_ids:
            GL.glActiveTexture(GL.GL_TEXTURE0 + tex_index)
            GL.glBindTexture(tex_type, tex)
            GL.glUniform1i(
                GL.glGetUniformLocation(program, sampler_name),
                tex_index,
            )

    def clear_uniforms(self) -> None:
        """
        Forget about the dynamic property uniforms needed for the
        previous OCIO shader build.
        """
        self._uniform_ids.clear()

    def use_uniforms(self, program) -> None:
        """
        Bind and/or update dynamic property uniforms needed for the
        current OCIO shader build.
        """
        if not self._shader_desc or not program:
            return

        self._make_current()

        for name, uniform_data in self._shader_desc.getUniforms():
            if name not in self._uniform_ids:
                uid = GL.glGetUniformLocation(program, name)
                self._uniform_ids[name] = uid
            else:
                uid = self._uniform_ids[name]

            if uniform_data.type == ocio.UNIFORM_DOUBLE:
                GL.glUniform1f(uid, uniform_data.getDouble())

    def update_dynamic_property(
        self, prop_type: ocio.DynamicPropertyType, value: Any
    ) -> None:
        """
        Update a specific OCIO dynamic property, which will be passed
        to the shader program as a uniform.

        :param prop_type: Property type to update. Only one dynamic
            property per type is supported per processor, so only the
            first will be updated if there are multiple.
        :param value: An appropriate value for the specific property
            type.
        """
        if not self._shader_desc:
            return

        if self._shader_desc.hasDynamicProperty(prop_type):
            dyn_prop = self._shader_desc.getDynamicProperty(prop_type)
            dyn_prop.setDouble(value)

    def delete_gl(self) -> None:
        """Free all OCIO GPU resources (textures + uniforms)."""
        self.delete_textures()
        self.clear_uniforms()
