
import os
import time
import torch
from torch.utils.data import DataLoader, Subset
from gaussian_model import GaussianModel
from datasets import WaveDataset
from utils.criterion import Criterion
from utils.metric import metric_cal
from utils.config import load_config
from utils.visualize import visualize_all


def inference(config):
    print(f"-----------------------------------------------")
    # Device setting
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Config file
    print(f"Using config file: {config.path.config}")

    # Initialize gaussian model
    gaussians = GaussianModel(config)

    # Setup data
    test_dataset = WaveDataset(config, eval=True)
    indices = list(range(min(config.logging.num_samples, len(test_dataset))))
    test_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batchsize, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Test set size: {len(test_dataset)}")
    print(f"Batchsize: {config.training.batchsize}")

    # Load weights from checkpoint if provided
    if config.path.checkpoint:
        print(f"Checkpoint path: {config.path.checkpoint}")
        (model_params, first_iter) = torch.load(
            config.path.checkpoint, weights_only=False)
        gaussians.restore(model_params)
    else:
        print("No checkpoint path provided, using randomly initialized model.")

    print(f"-----------------------------------------------")

    # Inference and save results
    total_time = 0
    first_iter_time = None
    final_metric = {}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            gt_time, position_rx, position_tx = batch

            start_time = time.time()
            pred_time = gaussians(position_rx.to(device))
            batch_time = time.time() - start_time

            total_time += batch_time

            if i == 0:
                first_iter_time = batch_time

            metrics = metric_cal(gt_time.detach().cpu().numpy(),
                                 pred_time.detach().cpu().numpy(), config.audio.fs)

            final_metric = {
                k: final_metric.get(k, 0) + v * len(position_rx) for k, v in metrics.items()}

            # Visualizations
            if config.logging.viz:

                output_dir = "output/inference"
                print("Visualization folder: {}".format(output_dir))
                os.makedirs(output_dir, exist_ok=True)

                criterion = Criterion(config)
                loss_dict, gt_freq, pred_freq = criterion(
                    pred_time, gt_time.to(device))

                for j in range(len(batch)):

                    visualize_all(
                        pred_freq=pred_freq[j],
                        gt_freq=gt_freq[j],
                        pred_time=pred_time[j],
                        gt_time=gt_time[j],
                        position_rx=position_rx[j],
                        position_tx=position_tx[j],
                        mode_set="test",
                        save_dir=output_dir,
                        iteration=i * config.training.batchsize + j,
                        gaussians=gaussians,
                        sr=config.audio.fs,
                        coord_min=config.rendering.coord_min,
                        coord_max=config.rendering.coord_max,
                    )

        final_metric = {
            k: v / len(test_dataset) for k, v in final_metric.items()}

    # Print results
    print("[Inference results]")
    print("   ".join([f"{k}: {v:.3f}" for k, v in final_metric.items()]))
    print()
    print("[Inference time]")
    print(f"   {len(test_dataset)} samples generated in {total_time * 1000:.2f} ms.")
    print(
        f"   First iteration time:\t{first_iter_time * 1000:.2f} ms.")
    print("[Including first]")
    print(
        f"   Average time per sample:\t{total_time / len(test_dataset) * 1000:.2f} ms.")
    print(
        f"   Average time per iteration:\t{total_time / len(test_loader) * 1000:.2f} ms.")
    if len(test_loader) > 1:
        time_wo_first = total_time - first_iter_time
        print("[Excluding first]")
        print(
            f"   Average time per sample:\t{time_wo_first / (len(test_dataset) - config.training.batchsize) * 1000:.2f} ms.")
        print(
            f"   Average time per iteration:\t{time_wo_first / (len(test_loader) - 1) * 1000:.2f} ms.")


if __name__ == "__main__":
    config = load_config()
    inference(config)
