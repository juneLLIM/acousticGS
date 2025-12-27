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
import pytz
import numpy as np
import random
import torch.nn.functional as F
import torchaudio


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
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def build_rotation(rot, device="cpu"):

    # 3D rotation matrix from quaternion
    if rot.shape[-1] == 4:

        q = F.normalize(rot, dim=-1)

        R = torch.zeros((q.size(0), 3, 3), device=device)

        r, x, y, z = q.unbind(-1)

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    # 4D rotation matrix from dual quaternion
    elif rot.shape[-1] == 8:

        l = rot[..., :4]
        r = rot[..., 4:]

        q_l = F.normalize(l, dim=-1)
        q_r = F.normalize(r, dim=-1)

        a, b, c, d = q_l.unbind(-1)
        p, q, r, s = q_r.unbind(-1)

        R = torch.zeros((rot.size(0), 4, 4), device=device)

        R[..., 0, 0] = a*p + b*q + c*r + d*s
        R[..., 0, 1] = a*q - b*p - c*s + d*r
        R[..., 0, 2] = a*r + b*s - c*p - d*q
        R[..., 0, 3] = a*s - b*r + c*q - d*p

        R[..., 1, 0] = a*q - b*p + c*s - d*r
        R[..., 1, 1] = a*p + b*q - c*r - d*s
        R[..., 1, 2] = a*s - b*r - c*q + d*p
        R[..., 1, 3] = a*r + b*s + c*p + d*q

        R[..., 2, 0] = a*r - b*s - c*p + d*q
        R[..., 2, 1] = a*s + b*r + c*q + d*p
        R[..., 2, 2] = a*p - b*q + c*r - d*s
        R[..., 2, 3] = a*q + b*p - c*s - d*r

        R[..., 3, 0] = a*s + b*r - c*q - d*p
        R[..., 3, 1] = a*r - b*s + c*p - d*q
        R[..., 3, 2] = a*q + b*p + c*s + d*r
        R[..., 3, 3] = a*p - b*q - c*r + d*s

    # 5D rotation matrix from rotor vector
    elif rot.shape[-1] == 10:

        A = torch.zeros((rot.shape[0], 5, 5), device=device, dtype=rot.dtype)
        i, j = torch.triu_indices(5, 5, offset=1, device=device)
        A[:, i, j] = rot
        A[:, j, i] = -rot
        R = torch.matrix_exp(A)

    return R


def safe_state(silent, device="cpu"):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", f"\n[{now()}]\t"))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(device)


def now():
    return datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m/%d %H:%M:%S")


def now_str():
    return datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")


def save_audio(waveform, sr, save_path):
    """Save waveform audio file.
    """

    waveform = waveform.detach().cpu()

    # Normalize to -1.0 ~ 1.0
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    # Save as WAV file
    torchaudio.save(save_path, waveform.unsqueeze(0), sr)
