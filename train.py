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

import os
import torch
from tqdm import tqdm
import shutil

from gaussian_model import GaussianModel
from utils.general_utils import safe_state
from torch.utils.data import DataLoader
from datasets import WaveDataset
import wandb

from utils.criterion import Criterion
from utils.metric import metric_cal
from utils.logger import plot_and_save_figure
from utils.config import load_config
from utils.visualize import visualize_gaussian_xyt_3d


def log_and_visualize(iteration, losses, config, output_dir):
    """Handles all logging and visualization tasks."""
    # Log scalar loss values to WandB
    log_losses = {k: v for k, v in losses.items() if isinstance(
        v, (int, float)) or (hasattr(v, 'numel') and v.numel() == 1)}
    wandb.log(log_losses, step=iteration)

    # Every 100 iterations, calculate and log metrics and visualizations
    if iteration % 100 == 0:
        fs = config.audio.fs
        # Retrieve time-domain waveforms from the losses dictionary
        gt_waveform = losses['ori_time']
        rendered_signal = losses['pred_time']

        metrics = metric_cal(gt_waveform.squeeze().cpu().numpy(),
                             rendered_signal.squeeze().cpu().numpy(), fs)
        wandb.log({f"metrics/{k}": v for k,
                  v in metrics.items()}, step=iteration)

        vis_dir = os.path.join(output_dir, "train_vis")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"iter_{iteration}.png")

        # Call plotting function
        plot_and_save_figure(
            pred_sig=losses['pred_freq'].cpu(),
            ori_sig=losses['gt_freq'].cpu(),
            pred_time=rendered_signal.squeeze().cpu(),
            ori_time=gt_waveform.squeeze().cpu(),
            position_rx=losses['position_rx'].squeeze().cpu(),
            position_tx=losses['position_tx'].squeeze().cpu(),
            mode_set="train",
            save_path=save_path
        )


def training(config, config_path):
    # Setup output directory
    output_dir = config.data.model_path
    print("Output folder: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Save the config file for reproducibility
    shutil.copy(config_path, os.path.join(output_dir, "config.yml"))

    # Initialize WandB
    use_wandb = False
    try:
        wandb.login(timeout=5)
        wandb.init(
            project=config.logging.project_name,
            name=os.path.basename(output_dir),
            config=vars(config)
        )
        use_wandb = True
        print("W&B logging is enabled.")
    except Exception as e:
        print(
            f"W&B login failed: {e}\nTraining will proceed without W&B logging.")

    # Initialize model and data loader
    gaussians = GaussianModel()
    train_dataset = WaveDataset(base_folder=config.data.source_path,
                                dataset_type='MeshRIR', eval=False,
                                seq_len=config.audio.seq_len, fs=config.audio.fs)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # Setup loss function
    # Convert the config.audio namespace object to a dictionary using vars()
    # This allows the Criterion class to use dictionary-style access cfg['key']
    criterion = Criterion(vars(config.audio)).cuda()
    loss_names = ['spec', 'amp', 'angle', 'time',
                  'energy', 'stft', 'pred_time', 'ori_time']

    # Setup training
    first_iter = 0
    if config.logging.start_checkpoint:
        (model_params, first_iter) = torch.load(
            config.logging.start_checkpoint)
        gaussians.restore(model_params, config.optimizer)
    else:
        gaussians.create_random(
            count=config.model.initial_points, aabb=config.model.aabb)

    gaussians.training_setup(config.optimizer)

    # Training loop
    progress_bar = tqdm(
        range(first_iter, config.optimizer.iterations), desc="Training progress")
    for iteration, batch in enumerate(train_loader, start=first_iter):
        if iteration >= config.optimizer.iterations:
            break

        gaussians.update_learning_rate(iteration)

        # Prepare data
        gt_wave_freq, position_rx, position_tx = batch
        gt_waveform = torch.fft.irfft(
            gt_wave_freq, n=config.audio.seq_len).cuda()
        render_points = position_rx.cuda()

        # Render and compute loss
        rendered_waveform = gaussians.render_signal_waveform(
            render_points, 0, (config.audio.seq_len / config.audio.fs), config.audio.seq_len)

        # Convert waveforms to frequency domain before calculating loss
        window = torch.hann_window(
            config.audio.n_fft, device=rendered_waveform.device)
        rendered_freq = torch.stft(
            rendered_waveform, n_fft=config.audio.n_fft, hop_length=config.audio.hop_length, window=window, return_complex=True)
        gt_freq = torch.stft(
            gt_waveform, n_fft=config.audio.n_fft, hop_length=config.audio.hop_length, window=window, return_complex=True)

        loss_values = criterion(rendered_freq, gt_freq)
        losses = dict(zip(loss_names, loss_values))

        loss_components = ['spec', 'amp', 'angle', 'time', 'energy', 'stft']
        total_loss = sum(losses[name] for name in loss_components)
        total_loss.backward()

        # Densification
        with torch.no_grad():
            if iteration > config.densification.densify_from_iter and iteration < config.densification.densify_until_iter:
                grads = gaussians._xyzt.grad
                if grads is not None:
                    gaussians.add_densification_stats(grads)
                    if iteration % config.densification.densification_interval == 0:
                        gaussians.densify_and_prune(
                            config.densification.densify_grad_threshold,
                            config.densification.prune_opacity_threshold,
                            config.densification.prune_scale_threshold,
                            config.densification.densify_scale_threshold
                        )

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            gaussians.optimizer.param_groups[0]['params'], 1.0)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Reset any Gaussians that have become invalid (NaN)
            invalid_positions = torch.any(torch.isnan(gaussians._xyzt), dim=1)
            invalid_opacities = torch.any(
                torch.isnan(gaussians._opacity), dim=1)
            invalid_scales = torch.any(torch.isnan(gaussians._scaling), dim=1)
            invalid_rotations = torch.any(
                torch.isnan(gaussians._rotation), dim=1)

            invalid_gaussians_mask = invalid_positions | invalid_opacities | invalid_scales | invalid_rotations
            if torch.any(invalid_gaussians_mask):
                print(
                    f"\n[ITER {iteration}] Resetting {torch.sum(invalid_gaussians_mask).item()} invalid Gaussians.")
                gaussians.reset_invalid_gaussians(invalid_gaussians_mask)

            # Logging and visualization
            if use_wandb and iteration % 10 == 0:
                losses['total'] = total_loss
                losses['learning_rate'] = gaussians.optimizer.param_groups[0]['lr']
                # Pass additional data to the logging function
                losses['pred_freq'] = rendered_freq
                losses['gt_freq'] = gt_freq
                losses['position_rx'] = position_rx
                losses['position_tx'] = position_tx
                log_and_visualize(iteration, losses, config, output_dir)

            # Save model
            if iteration in config.logging.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                gaussians.save_ply(os.path.join(
                    output_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
            if iteration in config.logging.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(
                    output_dir, f"chkpnt{iteration}.pth"))

            # Visualize the 3D Gaussian distribution
            if iteration % 1000 == 0:
                vis_3d_dir = os.path.join(output_dir, "gaussian_vis_3d")
                os.makedirs(vis_3d_dir, exist_ok=True)
                save_vis_path = os.path.join(
                    vis_3d_dir, f"iter_{iteration}.png")
                visualize_gaussian_xyt_3d(gaussians, save_vis_path)

        progress_bar.update(1)
        progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})

    progress_bar.close()
    print("\nTraining complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # The config loader now returns the config object and the path to the loaded file
    config, config_path = load_config()

    if config.optimizer.iterations not in config.logging.save_iterations:
        config.logging.save_iterations.append(config.optimizer.iterations)

    print("Optimizing " + config.data.model_path)
    safe_state()

    # Pass both the config object and its path to the training function
    training(config, config_path)
