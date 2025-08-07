import torch
import torch.nn as nn
import auraloss


class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Add these lines to store audio config
        self.n_fft = cfg['n_fft']
        # Use hop_length from config or default
        self.hop_length = cfg.get('hop_length', self.n_fft // 4)

        self.spec_loss_weight = cfg['spec_loss_weight']
        self.amplitude_loss_weight = cfg['amplitude_loss_weight']
        self.angle_loss_weight = cfg['angle_loss_weight']
        self.time_loss_weight = cfg['time_loss_weight']
        self.energy_loss_weight = cfg['energy_loss_weight']
        self.multi_stft_weight = cfg['multistft_loss_weight']

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[
                                                               512, 256, 128, 64], win_lengths=[300, 150, 75, 30], hop_sizes=[60, 30, 8, 4])

    def forward(self, pred_sig, ori_sig):
        # Add a small epsilon for numerical stability
        epsilon = 1e-8

        # --- Time-Domain Conversion ---
        # Use a window function to prevent spectral leakage warnings and improve accuracy
        window = torch.hann_window(self.n_fft, device=pred_sig.device)
        pred_time = torch.istft(
            pred_sig, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        ori_time = torch.istft(ori_sig, n_fft=self.n_fft,
                               hop_length=self.hop_length, window=window)

        # --- Spectrogram Calculation ---
        pred_spec = torch.abs(pred_sig)
        ori_spec = torch.abs(ori_sig)

        # --- Loss Calculations ---
        # 1. Spectral Magnitude Loss (L1 on magnitudes)
        spec_loss = self.l1_loss(pred_spec, ori_spec) * self.spec_loss_weight

        # 2. Log-Spectral Amplitude Loss (stable version)
        amplitude_loss = self.l1_loss(torch.log(
            pred_spec + epsilon), torch.log(ori_spec + epsilon)) * self.amplitude_loss_weight

        # 3. Phase Loss
        angle_loss = self.l1_loss(torch.angle(
            pred_sig), torch.angle(ori_sig)) * self.angle_loss_weight

        # 4. Time-Domain Waveform Loss
        time_loss = self.l1_loss(pred_time, ori_time) * self.time_loss_weight

        # 5. Energy Loss (MSE on squared magnitudes)
        energy_loss = self.mse_loss(torch.sum(pred_time ** 2, dim=-1) + epsilon,
                                    torch.sum(ori_time ** 2, dim=-1) + epsilon) * self.energy_loss_weight

        # 6. Multi-Resolution STFT Loss
        # auraloss expects (batch, channels, time)
        multi_stft_loss = self.mrft_loss(pred_time.unsqueeze(
            1), ori_time.unsqueeze(1)) * self.multi_stft_weight

        return spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time
