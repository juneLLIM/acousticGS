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
import wandb

from gaussian_model import GaussianModel
from torch.utils.data import DataLoader
from datasets import WaveDataset
from renderer import AVRRender
from utils.criterion import Criterion
from utils.general_utils import safe_state
from utils.metric import metric_cal
from utils.logger import plot_and_save_figure
from utils.config import load_config
from utils.visualize import visualize_gaussian_xyt_3d


# def log_and_visualize(iteration, losses, config, output_dir):
#     """Handles all logging and visualization tasks."""
#     # Log scalar loss values to WandB
#     log_losses = {k: v for k, v in losses.items() if isinstance(
#         v, (int, float)) or (hasattr(v, 'numel') and v.numel() == 1)}
#     wandb.log(log_losses, step=iteration)

#     # Every 100 iterations, calculate and log metrics and visualizations
#     if iteration % 100 == 0:
#         fs = config.audio.fs
#         # Retrieve time-domain waveforms from the losses dictionary
#         gt_waveform = losses['ori_time']
#         rendered_signal = losses['pred_time']

#         metrics = metric_cal(gt_waveform.squeeze().cpu().numpy(),
#                              rendered_signal.squeeze().cpu().numpy(), fs)
#         wandb.log({f"metrics/{k}": v for k,
#                   v in metrics.items()}, step=iteration)

#         vis_dir = os.path.join(output_dir, "train_vis")
#         os.makedirs(vis_dir, exist_ok=True)
#         save_path = os.path.join(vis_dir, f"iter_{iteration}.png")

#         # Call plotting function
#         plot_and_save_figure(
#             pred_sig=losses['pred_freq'].cpu(),
#             ori_sig=losses['gt_freq'].cpu(),
#             pred_time=rendered_signal.squeeze().cpu(),
#             ori_time=gt_waveform.squeeze().cpu(),
#             position_rx=losses['position_rx'].squeeze().cpu(),
#             position_tx=losses['position_tx'].squeeze().cpu(),
#             mode_set="train",
#             save_path=save_path
#         )


def training(config):

    # Setup output directory
    output_dir = config.path.output
    print("Output folder: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Save the config file for reproducibility
    shutil.copy(config.path.config, os.path.join(output_dir, "config.yml"))

    # Initialize WandB
    if config.logging.wandb:
        try:
            wandb.login(timeout=5)
            wandb.init(
                project=config.logging.project_name,
                name=os.path.basename(output_dir),
                config=vars(config)
            )
            print("W&B logging is enabled.")
        except Exception as e:
            config.logging.wandb = False
            print(
                f"W&B login failed: {e}\nTraining will proceed without W&B logging.")

    # Initialize model and renderer
    gaussians = GaussianModel(config)
    renderer = AVRRender(networks_fn=gaussians, config=config)

    # Setup data
    train_dataset = WaveDataset(config, eval=False)
    test_dataset = WaveDataset(config, eval=True)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Setup loss function
    criterion = Criterion(config)

    # Setup training
    first_iter = 0
    if config.path.checkpoint:
        (model_params, first_iter) = torch.load(
            config.path.checkpoint, weights_only=False)
        gaussians.restore(model_params)
    else:
        gaussians.create_random()

    # Setup iteration timing
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Training loop
    progress_bar = tqdm(range(first_iter, config.training.total_iterations),
                        desc="Training progress")
    for iteration, batch in enumerate(train_loader, start=first_iter+1):
        if iteration > config.training.total_iterations:
            break

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Prepare data
        gt_freq, position_rx, position_tx = batch

        # Render
        pred_freq = renderer(position_rx.cuda(), position_tx.cuda())

        # Compute loss
        loss_dict, gt_time, pred_time = criterion(pred_freq, gt_freq.cuda())
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        iter_end.record()

        gaussians.add_densification_stats(
            update_filter=torch.ones(gaussians.get_mean.shape[0], dtype=bool))  # tmp -> fix after culling

        with torch.no_grad():

            # Densification
            if iteration > config.densification.densify_from_iter and iteration < config.densification.densify_until_iter:

                if iteration % config.densification.densification_interval == 0:
                    gaussians.densify_and_prune()

                if iteration % config.densification.opacity_reset_interval == 0 or iteration == config.densification.densify_from_iter:
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # # Logging and visualization
            # if use_wandb and iteration % 10 == 0:
            #     losses['total'] = total_loss
            #     losses['learning_rate'] = gaussians.optimizer.param_groups[0]['lr']
            #     # Pass additional data to the logging function
            #     losses['pred_freq'] = rendered_freq
            #     losses['gt_freq'] = gt_freq
            #     losses['position_rx'] = position_rx
            #     losses['position_tx'] = position_tx
            #     log_and_visualize(iteration, losses, config, output_dir)

            # # Visualize the 3D Gaussian distribution
            # if iteration % 1000 == 0:
            #     vis_3d_dir = os.path.join(output_dir, "gaussian_vis_3d")
            #     os.makedirs(vis_3d_dir, exist_ok=True)
            #     save_vis_path = os.path.join(
            #         vis_3d_dir, f"iter_{iteration}.png")
            #     visualize_gaussian_xyt_3d(gaussians, save_vis_path)

            # Save model
            if iteration % config.training.save_freq == 0:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(
                    output_dir, f"chkpnt{iteration}.pth"))

        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})

    progress_bar.close()
    print("\nTraining complete.")
    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = load_config()

    print("-----------------------------------------------")
    print("Project name: " + config.logging.project_name)
    if config.path.checkpoint:
        print("Checkpoint path: " + config.path.checkpoint)
    else:
        print("No checkpoint path provided, starting from scratch.")
    safe_state(config.logging.short)
    print("-----------------------------------------------")

    training(config)
