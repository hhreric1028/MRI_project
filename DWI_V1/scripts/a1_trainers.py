"""A1-100 trainer: B0 with one controlled intensity-augmentation change."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from recall_trainers import nnUNetTrainer_B0R2022
from screen100_trainers import Screen100Mixin


B0_INTENSITY_AUGMENTATION = {
    "gaussian_noise": {"variance": [0.0, 0.1], "apply_probability": 0.10},
    "gaussian_blur": {"sigma": [0.5, 1.0], "apply_probability": 0.20, "p_per_channel": 0.50},
    "multiplicative_brightness": {"range": [0.75, 1.25], "apply_probability": 0.15},
    "contrast": {"range": [0.75, 1.25], "apply_probability": 0.15, "preserve_range": True},
    "gamma_invert": {"range": [0.7, 1.5], "apply_probability": 0.10, "retain_stats": True},
    "gamma": {"range": [0.7, 1.5], "apply_probability": 0.30, "retain_stats": True},
}

A1_INTENSITY_AUGMENTATION = {
    "gaussian_noise": {"variance": [0.0, 0.1], "apply_probability": 0.15},
    # Deliberately identical to B0. A1 does not strengthen blur.
    "gaussian_blur": {"sigma": [0.5, 1.0], "apply_probability": 0.20, "p_per_channel": 0.50},
    "multiplicative_brightness": {"range": [0.85, 1.15], "apply_probability": 0.25},
    "contrast": {"range": [0.75, 1.25], "apply_probability": 0.30, "preserve_range": True},
    # Remove intensity inversion; use only physiologically plausible non-inverting gamma.
    "gamma_invert": {"range": [0.7, 1.5], "apply_probability": 0.0, "retain_stats": True},
    "gamma": {"range": [0.7, 1.5], "apply_probability": 0.50, "retain_stats": True},
}


def get_a1_intensity_preview_transform() -> BasicTransform:
    """The changed A1 intensity transforms only, for pre-training visual QA."""
    return ComposeTransforms(
        [
            RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
                apply_probability=0.15,
            ),
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.85, 1.15)), synchronize_channels=False, p_per_channel=1
                ),
                apply_probability=0.25,
            ),
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.30,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.50,
            ),
        ]
    )


class nnUNetTrainer_A1_DWIIntensityAug_Screen100(Screen100Mixin, nnUNetTrainer_B0R2022):
    """B0-100 with only the declared intensity augmentation probabilities/ranges changed."""

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        # This is nnUNet v2.8.1's default pipeline copied verbatim except for the
        # four A1 intensity settings annotated below. Keeping the full method here
        # makes accidental changes to spatial/mirror/segmentation transforms auditable.
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg="constant",
                padding_value_seg=-1,
            )
        )
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # A1 change: same mild variance, slightly higher low application probability.
        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
                apply_probability=0.15,
            )
        )
        # Unchanged B0 blur.
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.2,
            )
        )
        # A1 change: narrower brightness magnitude, modestly higher frequency.
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.85, 1.15)), synchronize_channels=False, p_per_channel=1
                ),
                apply_probability=0.25,
            )
        )
        # A1 change: same contrast range, higher frequency.
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.30,
            )
        )
        # Unchanged B0 low-resolution simulation.
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.25,
            )
        )
        # A1 change: no inverted-image gamma branch; one non-inverting branch only.
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.50,
            )
        )
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )
        transforms.append(RemoveLabelTansform(-1, 0))
        if is_cascaded:
            assert foreground_labels is not None
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=0.5,
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=0.5,
                    ),
                    apply_probability=0.2,
                )
            )
        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0,
                )
            )
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)


A1_TRAINERS = {"A1_DWIIntensityAug": nnUNetTrainer_A1_DWIIntensityAug_Screen100}
