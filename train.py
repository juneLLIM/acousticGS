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
from utils.criterion import Criterion
from utils.metric import metric_cal
from utils.config import load_config
from utils.visualize import visualize_all, visualize_geometry
from utils.general_utils import safe_state, now_str, save_audio


def training(config):

    print("-----------------------------------------------")
    print("Project name: " + config.logging.project_name)

    # Device setting
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = f'{config.path.output}/{now_str()}'
    print("Output folder: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Save the config file for reproducibility
    shutil.copy(config.path.config, os.path.join(output_dir, "config.yml"))

    # Initialize gaussian model
    gaussians = GaussianModel(config)

    # Setup data
    train_dataset = WaveDataset(
        config, eval=False, iteration=config.training.total_iterations)
    test_dataset = WaveDataset(config, eval=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    print(
        f"Train: {len(train_dataset)},\t Test: {len(test_dataset)}")
    print(f"Batchsize: {config.training.batchsize}")

    # Setup loss function
    criterion = Criterion(config)

    # Setup training
    first_iter = 0
    if config.path.checkpoint:
        print(f"Checkpoint path: {config.path.checkpoint}")
        (model_params, first_iter) = torch.load(
            config.path.checkpoint, weights_only=False)
        gaussians.restore(model_params)
    else:
        print("No checkpoint path provided, starting from scratch.")
        gaussians.initialize(position_tx=train_dataset.positions_tx[0])

    print(f"-----------------------------------------------")

    # Setup iteration timing
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Training loop
    print("Training start...")
    progress_bar = tqdm(range(first_iter, config.training.total_iterations),
                        desc="Training progress")

    # Compute first sample in testset
    gt_time, position_rx, position_tx = next(iter(test_loader))
    pred_time = gaussians(position_rx.to(device))
    loss_dict, gt_freq, pred_freq = criterion(
        pred_time, gt_time.to(device))

    # Saving path setup
    vis_dir = os.path.join(output_dir, "test_vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Call visualization function
    visualize_geometry(
        train_rx=train_dataset.positions_rx,
        train_tx=train_dataset.positions_tx,
        test_rx=test_dataset.positions_rx,
        test_tx=test_dataset.positions_tx,
        save_path=os.path.join(output_dir, "geometry.png"),
    )
    visualize_all(
        pred_freq=pred_freq[0, :],
        gt_freq=gt_freq[0, :],
        pred_time=pred_time[0, :],
        gt_time=gt_time[0, :],
        position_rx=position_rx[0, :],
        position_tx=position_tx[0, :],
        mode_set="test",
        save_dir=vis_dir,
        iteration=0,
        gaussians=gaussians,
        sr=config.audio.fs,
        coord_min=config.rendering.coord_min,
        coord_max=config.rendering.coord_max,
    )
    save_audio(
        waveform=gt_time[0, :],
        sr=config.audio.fs,
        save_path=os.path.join(vis_dir, f"audio_gt.wav")
    )

    for iteration, batch in enumerate(train_loader, start=first_iter+1):
        if iteration > config.training.total_iterations:
            break

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Prepare data
        gt_time, position_rx, position_tx = batch

        # Render
        pred_time = gaussians(position_rx.to(device))

        # Compute loss
        loss_dict, gt_freq, pred_freq = criterion(
            pred_time, gt_time.to(device))
        total_loss = loss_dict["total_loss"]
        gaussians.backward(total_loss)

        iter_end.record()

        with torch.no_grad():

            # Densification
            if iteration >= config.densification.densify_from_iter and iteration < config.densification.densify_until_iter:

                if iteration % config.densification.densification_interval == 0:
                    gaussians.densify_and_prune()

                if iteration % config.densification.opacity_reset_interval == 0 or iteration == config.densification.densify_from_iter:
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # Logging
            if iteration % config.logging.log_freq == 0:

                if config.logging.wandb:
                    # Log learning rate and loss to WandB
                    wandb.log(
                        {'learning_rate': gaussians.optimizer.param_groups[0]['lr']}, step=iteration)
                    wandb.log(
                        {f"loss/{k}": v for k, v in loss_dict.items()}, step=iteration)
                    wandb.log(
                        {'number of gaussians': gaussians.get_mean.shape[0]}, step=iteration)

            # Log test set metrics
            if iteration % config.logging.test_freq == 0:

                final_metric = {}

                for batch in test_loader:
                    gt_time, position_rx, position_tx = batch

                    pred_time = gaussians(position_rx.to(device))

                    metrics = metric_cal(gt_time.detach().cpu().numpy(),
                                         pred_time.detach().cpu().numpy(), config.audio.fs)

                    final_metric = {
                        k: final_metric.get(k, 0) + v * len(batch) for k, v in metrics.items()}

                final_metric = {
                    k: v / len(test_dataset) for k, v in final_metric.items()}

                if config.logging.wandb:
                    wandb.log(
                        {f"metrics/{k}": v for k, v in final_metric.items()}, step=iteration)

            # Visualizations
            if iteration % config.logging.viz_freq == 0:

                # Compute first sample in testset
                gt_time, position_rx, position_tx = next(iter(test_loader))
                pred_time = gaussians(position_rx.to(device))
                loss_dict, gt_freq, pred_freq = criterion(
                    pred_time, gt_time.to(device))

                # Call plotting function
                visualize_all(
                    pred_freq=pred_freq[0, :],
                    gt_freq=gt_freq[0, :],
                    pred_time=pred_time[0, :],
                    gt_time=gt_time[0, :],
                    position_rx=position_rx[0, :],
                    position_tx=position_tx[0, :],
                    mode_set="test",
                    save_dir=vis_dir,
                    iteration=iteration,
                    gaussians=gaussians,
                    sr=config.audio.fs,
                    coord_min=config.rendering.coord_min,
                    coord_max=config.rendering.coord_max,
                )

            # Save model
            if iteration % config.logging.save_freq == 0:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(
                    output_dir, f"chkpnt{iteration}.pth"))

        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({"Loss": f"{total_loss.item():.{7}f}"})

    progress_bar.close()
    print("\nTraining complete.")
    if config.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    config = load_config()

    safe_state(silent=not config.logging.log,
               device=torch.device(config.device))
    print()

    # Initialize WandB
    if config.logging.wandb:
        try:
            wandb.login(timeout=5)
            wandb.init(
                project=config.logging.project_name,
                name=os.path.basename(config.path.output),
                config=vars(config)
            )
            print("W&B logging is enabled.")
        except Exception as e:
            config.logging.wandb = False
            print(
                f"W&B login failed: {e}\nTraining will proceed without W&B logging.")

    training(config)
