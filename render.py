# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# import os
# import numpy as np
# from tqdm import tqdm
# from argparse import ArgumentParser
# import torchaudio

# from gaussian_model import GaussianModel
# from utils.config import load_config
# from datasets import WaveDataset
# from utils.general_utils import safe_state
# from utils.logger import plot_and_save_figure


# def render_audio_set(config, iteration):
#     # Get model and data paths from the config object.
#     model_path = config.data.model_path
#     data_path = config.data.source_path
#     print(f"Rendering audio for model: {model_path}")

#     # Load the Gaussian model.
#     # The constructor no longer takes arguments.
#     gaussians = GaussianModel()
#     checkpoint_path = os.path.join(
#         model_path, "chkpnt" + str(iteration) + ".pth")
#     (model_data, _) = torch.load(checkpoint_path)
#     # The restore method now expects the optimizer config.
#     gaussians.restore(model_data, config.optimizer)

#     # Load the dataset in evaluation mode.
#     seq_len = config.audio.seq_len
#     fs = config.audio.fs
#     render_dataset = WaveDataset(
#         base_folder=data_path, dataset_type='MeshRIR', eval=True, seq_len=seq_len, fs=fs)

#     # Create output directories.
#     render_path = os.path.join(model_path, "renders", f"iter_{iteration}")
#     plots_path = os.path.join(model_path, "plots", f"iter_{iteration}")
#     os.makedirs(render_path, exist_ok=True)
#     os.makedirs(plots_path, exist_ok=True)

#     # Begin the rendering loop.
#     with torch.no_grad():
#         for idx, batch in enumerate(tqdm(render_dataset, desc="Rendering audio progress")):
#             # Prepare data (unpacking corrected for MeshRIR dataset).
#             gt_wave_freq, position_rx, position_tx = batch
#             gt_waveform = torch.fft.irfft(gt_wave_freq, n=seq_len).cuda()
#             render_points = position_rx.cuda()

#             # Render the signal waveform.
#             rendered_signal = gaussians.render_signal_waveform(
#                 render_points, 0, (seq_len / fs), seq_len).squeeze(0)

#             # Save the rendered audio to a .wav file.
#             wav_path = os.path.join(render_path, f'render_{idx:04d}.wav')
#             torchaudio.save(wav_path, rendered_signal.unsqueeze(0).cpu(), fs)

#             # Convert signals to frequency domain for consistent visualization.
#             window = torch.hann_window(
#                 config.audio.n_fft, device=rendered_signal.device)
#             rendered_freq = torch.stft(
#                 rendered_signal, n_fft=config.audio.n_fft, hop_length=config.audio.hop_length, window=window, return_complex=True)
#             gt_freq = torch.stft(
#                 gt_waveform.squeeze(), n_fft=config.audio.n_fft, hop_length=config.audio.hop_length, window=window, return_complex=True)

#             # Save a comparison plot.
#             plot_path = os.path.join(plots_path, f'plot_{idx:04d}.png')
#             plot_and_save_figure(
#                 pred_sig=rendered_freq,
#                 ori_sig=gt_freq,
#                 pred_time=rendered_signal.squeeze(),
#                 ori_time=gt_waveform.squeeze(),
#                 position_rx=position_rx.squeeze(),
#                 position_tx=position_tx.squeeze(),
#                 mode_set="render",
#                 save_path=plot_path
#             )

#     print(
#         f"Rendering complete. Results saved to {render_path} and {plots_path}")


# if __name__ == "__main__":
#     # Set up a simplified argument parser.
#     parser = ArgumentParser(description="Rendering script for audio signals")
#     parser.add_argument("-m", "--model_path", required=True,
#                         type=str, help="Path to the model directory containing config.yml.")
#     parser.add_argument("--iteration", default=-1, type=int,
#                         help="Iteration number of the checkpoint to load.")
#     parser.add_argument("--quiet", action="store_true")
#     args = parser.parse_args()

#     # Initialize system state.
#     safe_state(args.quiet)

#     # Load the configuration file from the model's directory.
#     config_path = os.path.join(args.model_path, "config.yml")
#     config, _ = load_config(config_path)

#     # Update config with command-line arguments if necessary.
#     # This makes the script flexible.
#     if args.model_path:
#         config.data.model_path
