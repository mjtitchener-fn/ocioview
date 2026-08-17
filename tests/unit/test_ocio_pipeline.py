# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenColorIO Project.

import PyOpenColorIO as ocio

from ocioview.viewer.ocio_pipeline import (
    build_viewing_pipelines,
    next_channel_hot,
)


def test_passthrough_pipeline_has_only_gpu_viewport_ops(raw_config):
    # No input color space, no main transform: the CPU pipeline is empty and
    # the GPU pipeline holds exactly exposure, channel view, and gamma.
    config = ocio.GetCurrentConfig()
    gpu, cpu = build_viewing_pipelines(
        config=config,
        input_color_space=None,
        transform=None,
        exposure=0.0,
        gamma=1.0,
        channel_hot=[1, 1, 1, 1],
        scene_ref_name="scene_ref",
    )
    assert len(cpu) == 0
    assert len(gpu) == 3
    transforms = list(gpu)
    assert isinstance(transforms[0], ocio.ExposureContrastTransform)
    assert isinstance(transforms[1], ocio.MatrixTransform)
    assert isinstance(transforms[2], ocio.ExposureContrastTransform)


def test_main_transform_is_appended_to_both(raw_config):
    config = ocio.GetCurrentConfig()
    main_tf = ocio.ExponentTransform([2.2, 2.2, 2.2, 1.0])
    gpu, cpu = build_viewing_pipelines(
        config=config,
        input_color_space=None,
        transform=main_tf,
        exposure=0.0,
        gamma=1.0,
        channel_hot=[1, 1, 1, 1],
        scene_ref_name="scene_ref",
    )
    assert len(cpu) == 1
    cpu_transforms = list(cpu)
    assert isinstance(cpu_transforms[0], ocio.ExponentTransform)
    assert len(gpu) == 4
    gpu_transforms = list(gpu)
    assert isinstance(gpu_transforms[1], ocio.ExponentTransform)


def test_known_input_color_space_adds_scene_reference_conversions(raw_config):
    config = ocio.GetCurrentConfig()
    gpu, cpu = build_viewing_pipelines(
        config=config,
        input_color_space="raw",
        transform=None,
        exposure=0.0,
        gamma=1.0,
        channel_hot=[1, 1, 1, 1],
        scene_ref_name="scene_ref",
    )
    cpu_transforms = list(cpu)
    assert len(cpu_transforms) == 2
    assert all(
        isinstance(cpu_transforms[i], ocio.ColorSpaceTransform)
        for i in range(2)
    )
    assert len(gpu) == 5


def test_gpu_exposure_and_gamma_are_dynamic(raw_config):
    config = ocio.GetCurrentConfig()
    gpu, _ = build_viewing_pipelines(
        config=config,
        input_color_space=None,
        transform=None,
        exposure=1.5,
        gamma=2.0,
        channel_hot=[1, 1, 1, 1],
        scene_ref_name="scene_ref",
    )
    gpu_transforms = list(gpu)
    exposure_tf = gpu_transforms[0]
    gamma_tf = gpu_transforms[2]
    assert exposure_tf.getExposure() == 1.5
    assert gamma_tf.getGamma() == 2.0
    assert gamma_tf.getPivot() == 1.0


def test_scene_linear_role_adds_scene_linear_conversions(raw_config):
    # With a scene_linear role defined and a known input color space, both
    # pipelines gain a to-scene-linear and a to-scene-reference conversion.
    config = ocio.Config.CreateRaw()
    config.setRole(ocio.ROLE_SCENE_LINEAR, "raw")
    gpu, cpu = build_viewing_pipelines(
        config=config,
        input_color_space="raw",
        transform=None,
        exposure=0.0,
        gamma=1.0,
        channel_hot=[1, 1, 1, 1],
        scene_ref_name="scene_ref",
    )
    # CPU: to-scene-linear, to-scene-ref, restore-from-scene-ref.
    cpu_transforms = list(cpu)
    assert len(cpu_transforms) == 3
    assert all(
        isinstance(cpu_transforms[i], ocio.ColorSpaceTransform)
        for i in range(3)
    )
    # GPU adds exposure, channel view, gamma -> 6.
    assert len(gpu) == 6


def test_next_channel_hot_isolates_channel_from_all():
    assert next_channel_hot([1, 1, 1, 1], 0) == [1, 0, 0, 1]
    assert next_channel_hot([1, 1, 1, 1], 1) == [0, 1, 0, 1]
    assert next_channel_hot([1, 1, 1, 1], 2) == [0, 0, 1, 1]


def test_next_channel_hot_toggles_back_to_all_when_already_isolated():
    assert next_channel_hot([1, 0, 0, 1], 0) == [1, 1, 1, 1]


def test_next_channel_hot_switches_between_isolated_channels():
    assert next_channel_hot([1, 0, 0, 1], 1) == [0, 1, 0, 1]


def test_next_channel_hot_index_out_of_range_shows_all():
    assert next_channel_hot([1, 0, 0, 1], 3) == [1, 1, 1, 1]
    assert next_channel_hot([0, 1, 0, 1], 4) == [1, 1, 1, 1]
