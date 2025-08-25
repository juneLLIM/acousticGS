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


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


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


def strip_lowerdiag(L):
    uncertainty = torch.zeros(
        (L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(rot):

    if rot.shape[-1] == 4:

        q = F.normalize(rot, dim=-1)

        r, x, y, z = q.unbind(-1)

        R = torch.stack([
            1 - 2 * (y*y + z*z), 2 * (x*y - r*z), 2 * (x*z + r*y),
            2 * (x*y + r*z), 1 - 2 * (x*x + z*z), 2 * (y*z - r*x),
            2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x*x + y*y)
        ], dim=1).view(-1, 3, 3)

    elif rot.shape[-1] == 8:

        l = rot[..., :4]
        r = rot[..., 4:]

        q_l = F.normalize(l, dim=-1)
        q_r = F.normalize(r, dim=-1)

        a, b, c, d = q_l.unbind(-1)
        p, q, r, s = q_r.unbind(-1)

        M_l = torch.stack([a, -b, -c, -d,
                           b, a, -d, c,
                           c, d, a, -b,
                           d, -c, b, a], dim=1).view(-1, 4, 4)
        M_r = torch.stack([p, q, r, s,
                           -q, p, -s, r,
                           -r, s, p, -q,
                           -s, -r, q, p], dim=1).view(-1, 4, 4)

        R = M_l @ M_r

    return R


def build_scaling_rotation(s, r):
    L = torch.diag_embed(s)
    R = build_rotation(r)

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", f" [{now()}]\n"))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def now():
    return datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m/%d %H:%M:%S")
