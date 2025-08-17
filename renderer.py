import torch
import numpy as np
import torch.nn as nn


class AVRRender(nn.Module):
    """ Audio signal rendering method
    """

    def __init__(self, networks_fn, config) -> None:
        super().__init__()

        self.network_fn = networks_fn
        self.n_samples = config.rendering.n_samples
        self.near = config.rendering.near
        self.far = config.rendering.far
        self.n_azi = config.rendering.n_azi
        self.n_ele = config.rendering.n_ele
        self.speed = config.rendering.speed
        self.fs = config.audio.fs
        self.pathloss = config.rendering.pathloss
        self.xyz_min = config.rendering.xyz_min
        self.xyz_max = config.rendering.xyz_max

    def forward(self, rays_o, position_tx, direction_tx=None):
        """Render audio signal for RAF dataset

        Parameters
        ----------
        rays_o : [bs, 3]
            position of the microphone, origin of the ray
        position_tx : [bs, 3]
            position of the speaker
        direction_tx : [bs, 3]
            speaker orientation
        mic_dir : [bs, 1]
            microphone orientation for spatial audio rendering

        Returns
        -------
        receive_sig: [bs, N_seqlen, 2]
            rendered audio signals in frequency domain
        """
        bs = position_tx.size(0)

        # get the pts 3d positions along each ray
        dir, _, _ = ray_directions(n_azi=self.n_azi, n_ele=self.n_ele)
        d_vals = torch.linspace(0., 1., self.n_samples).cuda(
        ) * (self.far - self.near) + self.near  # scale t with near and far
        ray_pts = rays_o.unsqueeze(1).unsqueeze(2) + (dir.unsqueeze(1) * (d_vals.unsqueeze(
            0).unsqueeze(2))).unsqueeze(0)  # [bs, n_azi*n_ele+2 (N_rays), N_samples, 3]

        # normalize the input
        network_pts = normalize_points(ray_pts.reshape(
            bs, -1, 3), self.xyz_min, self.xyz_max)
        # [bs, N_rays * N_samples, 3]
        network_view = -1 * \
            dir.unsqueeze(0).unsqueeze(2).expand(
                ray_pts.size()).reshape(bs, -1, 3)
        network_tx = normalize_points(position_tx.unsqueeze(1).expand(
            *network_pts.size()), self.xyz_min, self.xyz_max)
        if direction_tx is not None:
            network_dir_tx = direction_tx.unsqueeze(1).expand(
                *network_pts.size())  # [bs, N_rays * N_samples, 3]

        # get network output
        if direction_tx is not None:
            attn, signal = self.network_fn(
                network_pts.reshape(-1, 3), network_view, network_tx, network_dir_tx)
        else:
            attn, signal = self.network_fn(
                network_pts.reshape(-1, 3), network_view, network_tx)
        attn = attn.view(bs, -1, self.n_samples)  # [bs, N_rays, N_samples]
        # [bs, N_rays, N_samples, N_lenseq]
        signal = signal.view(bs, -1, self.n_samples, signal.size(-1))

        # bounce points to rx delay samples
        pts2rx_idx = self.fs * d_vals / self.speed  # [N_samples]
        shift_samples = torch.round(pts2rx_idx)  # [N_samples]
        # apply zero mask to the end of the signal
        zero_mask_tail = torch.where((torch.arange(
            signal.size(-1)-1, 0-1, -1).cuda().unsqueeze(0) - shift_samples.unsqueeze(1)) > 0, 1, 0).cuda()
        signal = signal * zero_mask_tail

        # tx to bounce points delay samples
        tx2pts_idx = torch.linalg.vector_norm(denormalize_points(
            network_tx - network_pts, self.xyz_min, self.xyz_max), dim=-1).reshape(*attn.shape) * self.fs / self.speed  # [bs, N_rays, N_samples]
        delay_samples = torch.clamp(torch.round(
            tx2pts_idx), min=0, max=signal.size(-1) - 1).unsqueeze(-1)
        range_tensor = torch.arange(signal.size(-1)).cuda()  # [N_lenseq]
        # [bs, N_rays, N_samples, N_lenseq]
        zero_mask_tx2pts = range_tensor >= delay_samples
        signal = signal * zero_mask_tx2pts  # [bs, N_rays, N_samples, N_lenseq]

        # apply 1/d attenuations in time domain by shifting the samples
        prev_part = int(0.1 / self.speed * self.fs)
        ideal_dis2rx = torch.arange(
            0, signal.size(-1)*2.5, device='cuda') / self.fs * self.speed
        # account for path loss term
        path_loss = self.pathloss / (ideal_dis2rx + 1e-3)
        path_loss[0:prev_part] = path_loss[prev_part+1]
        path_loss_all = torch.stack([path_loss[i:i+signal.size(-1)]
                                    for i in shift_samples.detach().cpu().numpy().astype(int)])

        # Apply fft, and phase shift
        fft_sig = torch.fft.rfft(signal.float() * path_loss_all, dim=-1)
        phase_shift = torch.exp(-1j*2*np.pi/signal.size(-1)*torch.arange(
            0, signal.size(-1)//2+1).cuda().unsqueeze(0)*pts2rx_idx.unsqueeze(1))
        shifted_signal = fft_sig * phase_shift

        # audio signal rendering for each ray
        batch_n_rays_signal = acoustic_render(
            attn, shifted_signal, d_vals)  # [bs, N_rays, N_lenseq]

        # combine signal
        receive_sig = torch.sum(batch_n_rays_signal, dim=-2)  # [bs, N_lenseq]

        # split into real and imagenary part for multi gpu training
        receive_sig = torch.cat([torch.real(receive_sig).unsqueeze(-1), torch.imag(
            receive_sig).unsqueeze(-1)], dim=-1)  # [bs, N_lenseq, 2]
        return receive_sig


def normalize_points(input_pts, xyz_min, xyz_max):
    return 2*(input_pts - xyz_min) / (xyz_max - xyz_min) - 1


def denormalize_points(input_pts, xyz_min, xyz_max):
    return (input_pts + 1) / 2 * (xyz_max - xyz_min) + xyz_min


def ray_directions(n_azi, n_ele, random_azi=True):
    """get the ray directions

    Parameters
    ----------
    n_azi : int, number of azimuth directions
    n_ele : int, number of elevation directions
    random_azi : bool, whether involve random azimuth selection, by default True

    Returns
    -------
    dir : torch.tensor [n_azi * n_ele + 2, 3]
    """

    # Azimuth direction
    azi_ray = torch.linspace(0, np.pi*2, n_azi+1)[:-1].cuda()
    # Randomlly add an angle shift
    azi_randadd = (np.pi*2 / n_azi) * torch.rand(n_azi).cuda()
    azi_ray = azi_ray + azi_randadd if random_azi else azi_ray

    # Elevation direction
    ele_ray = torch.linspace(
        0, 1, n_ele+2)[1:-1].cuda() + (0.5 / n_ele) * torch.rand(n_ele).cuda() * 0
    ele_ray = torch.acos(2 * ele_ray - 1)

    # Combined direction
    azi_ray, ele_ray = torch.meshgrid(azi_ray, ele_ray, indexing='ij')

    pts_x = torch.mul(torch.cos(azi_ray.flatten()),
                      torch.sin(ele_ray.flatten())).unsqueeze(1)
    pts_y = torch.mul(torch.sin(azi_ray.flatten()),
                      torch.sin(ele_ray.flatten())).unsqueeze(1)
    pts_z = torch.cos(ele_ray.flatten()).unsqueeze(1)

    dir = torch.cat((pts_x, pts_y, pts_z), dim=1)  # r     [n_azi * n_ele, 3]
    # [n_azi * n_ele + 2, 3]
    dir = torch.cat((dir, torch.tensor([[0, 0, 1], [0, 0, -1]]).cuda()), dim=0)
    return dir, azi_ray, ele_ray


def acoustic_render(attn, signal, r_vals):
    """acoustic volume rendering 

    Parameters
    ----------
    attn   : [bs, N_rays, N_samples]. 
    signal : [bs, N_rays, N_samples, N_lenseq]
    r_vals : [N_samples]. Integration distance.

    Return:
    ----------
    n_rays_signal : [batchsize, N_rays, N_lenseq]
        Time signal of a specific ray direction
    """
    def raw2alpha(raw, dists): return 1.-torch.exp(-raw*dists)

    bs, n_rays, n_samples, n_lenseq = signal.shape

    dists = r_vals[..., 1:] - r_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists.unsqueeze(0).repeat(n_rays, 1)

    alpha = raw2alpha(attn, dists.repeat(bs, 1, 1))  # [bs, N_rays, N_samples]
    att_i = torch.cumprod(torch.cat(
        [torch.ones((alpha[..., :1].shape)).cuda(), 1.-alpha + 1e-6], -1), -1)[..., :-1]

    # [bs, N_rays, N_lenseq]
    n_rays_signal = torch.sum(signal*(att_i*alpha)[..., None], -2)
    return n_rays_signal
