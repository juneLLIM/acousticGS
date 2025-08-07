#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_symmetric(sym):
    """
    Extracts the upper-triangular elements of a symmetric 4x4 matrix.
    """
    indices = torch.triu_indices(4, 4, k=0, device=sym.device)
    return sym[:, indices[0], indices[1]]


def build_rotation(r):
    """
    Builds a 4D rotation matrix from 6 Lie algebra parameters
    using the exponential map (SO(4) group).
    This is a more stable and robust method than sequential rotations.
    """
    # r is a tensor of shape (N, 6), representing 6 rotation planes
    N = r.shape[0]

    # Create a batch of 4x4 skew-symmetric matrices B (elements of so(4))
    B = torch.zeros(N, 4, 4, device=r.device)

    # Populate the skew-symmetric matrices from the 6 parameters
    B[:, 0, 1], B[:, 1, 0] = -r[:, 0], r[:, 0]  # xy plane
    B[:, 0, 2], B[:, 2, 0] = -r[:, 1], r[:, 1]  # xz plane
    B[:, 0, 3], B[:, 3, 0] = -r[:, 2], r[:, 2]  # xt plane
    B[:, 1, 2], B[:, 2, 1] = -r[:, 3], r[:, 3]  # yz plane
    B[:, 1, 3], B[:, 3, 1] = -r[:, 4], r[:, 4]  # yt plane
    B[:, 2, 3], B[:, 3, 2] = -r[:, 5], r[:, 5]  # zt plane

    # The exponential map from the Lie algebra so(4) to the Lie group SO(4)
    return torch.matrix_exp(B)


def build_scaling_rotation(s, r):
    """Builds a combined scaling and rotation matrix L = R @ S."""
    R = build_rotation(r)
    S = torch.diag_embed(s)
    return R @ S


def safe_state(quiet=False):
    """
    Sets random seeds for reproducibility.
    [Refactored for simplicity]
    """
    if not quiet:
        print(
            f"[{datetime.now().strftime('%d/%m %H:%M:%S')}] Setting random seed to 0.")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # CUDNN 설정은 재현성에 중요
    torch.backends.cudnn.deterministic = True
