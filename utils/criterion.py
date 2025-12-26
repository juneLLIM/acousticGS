import torch
import torch.nn as nn
import auraloss


class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.spec_loss_weight = cfg.loss_weights.spec_loss_weight
        self.amplitude_loss_weight = cfg.loss_weights.amplitude_loss_weight
        self.angle_loss_weight = cfg.loss_weights.angle_loss_weight
        self.time_loss_weight = cfg.loss_weights.time_loss_weight
        self.energy_loss_weight = cfg.loss_weights.energy_loss_weight
        self.multi_stft_weight = cfg.loss_weights.multi_stft_loss_weight

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[
                                                               512, 256, 128, 64], win_lengths=[300, 150, 75, 30], hop_sizes=[60, 30, 8, 4])

    def forward(self, pred_time, gt_time):

        pred_freq = torch.fft.rfft(pred_time, dim=-1)
        gt_freq = torch.fft.rfft(gt_time, dim=-1)

        pred_spec = torch.abs(torch.stft(
            pred_time, n_fft=self.cfg.audio.n_fft, return_complex=True, window=torch.hann_window(self.cfg.audio.n_fft).to(pred_time.device)))
        gt_spec = torch.abs(torch.stft(
            gt_time, n_fft=self.cfg.audio.n_fft, return_complex=True, window=torch.hann_window(self.cfg.audio.n_fft).to(gt_time.device)))

        pred_spec_energy = torch.sum(pred_spec ** 2, dim=1)
        gt_spec_energy = torch.sum(gt_spec ** 2, dim=1)

        predict_energy = torch.log10(torch.flip(torch.cumsum(
            torch.flip(pred_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        predict_energy -= predict_energy[:, [0]]
        gt_energy = torch.log10(torch.flip(torch.cumsum(
            torch.flip(gt_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        gt_energy -= gt_energy[:, [0]]

        real_loss = self.l1_loss(torch.real(pred_freq), torch.real(gt_freq))
        imag_loss = self.l1_loss(torch.imag(pred_freq), torch.imag(gt_freq))
        spec_loss = (real_loss + imag_loss) * self.spec_loss_weight

        amplitude_loss = self.l1_loss(torch.abs(pred_freq), torch.abs(
            gt_freq)) * self.amplitude_loss_weight

        angle_loss = (self.l1_loss(torch.cos(torch.angle(pred_freq)), torch.cos(torch.angle(gt_freq))) +
                      self.l1_loss(torch.sin(torch.angle(pred_freq)), torch.sin(torch.angle(gt_freq)))) * self.angle_loss_weight

        time_loss = self.l1_loss(gt_time, pred_time) * self.time_loss_weight

        energy_loss = self.l1_loss(
            gt_energy, predict_energy) * self.energy_loss_weight

        multi_stft_loss = self.mrft_loss(gt_time.unsqueeze(
            1), pred_time.unsqueeze(1)) * self.multi_stft_weight

        total_loss = spec_loss + amplitude_loss + angle_loss + \
            time_loss + energy_loss + multi_stft_loss

        loss_dict = {
            "spec_loss": spec_loss,
            "amplitude_loss": amplitude_loss,
            "angle_loss": angle_loss,
            "time_loss": time_loss,
            "energy_loss": energy_loss,
            "multi_stft_loss": multi_stft_loss,
            "total_loss": total_loss
        }

        return loss_dict, gt_freq, pred_freq
