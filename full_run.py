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
from argparse import ArgumentParser
import time
import subprocess
import sys


def run_full_pipeline(args):
    # Executes the complete pipeline for an audio scene: training, rendering, and evaluation.
    model_output_path = os.path.join(args.output_path, args.model_name)
    print(f"Starting full pipeline for model: {args.model_name}")
    print(f"Data source: {args.data_path}")
    print(f"Output will be saved to: {model_output_path}")

    # --- 1. Training ---
    if not args.skip_training:
        print("\n--- Starting Training ---")
        start_time = time.time()

        # Construct the training command as a list for subprocess.
        train_cmd = [
            sys.executable, "train.py",
            "-s", args.data_path,
            "-m", model_output_path,
            "--iterations", str(args.iterations)
        ]

        print(f"Executing: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd)

        # Check if the command failed and exit the pipeline if it did.
        if result.returncode != 0:
            print(
                f"--- Training failed with exit code {result.returncode}. Stopping pipeline. ---")
            sys.exit(1)

        train_timing = (time.time() - start_time) / 60.0
        print(f"--- Training Finished in {train_timing:.2f} minutes ---\n")

        with open(os.path.join(model_output_path, "timing.txt"), 'w') as file:
            file.write(f"Training time: {train_timing:.2f} minutes\n")

    # --- 2. Rendering ---
    if not args.skip_rendering:
        print("\n--- Starting Rendering ---")

        # Render the final model checkpoint.
        render_iterations = [args.iterations]

        for iter_num in render_iterations:
            render_cmd = [
                sys.executable, "render.py",
                "-m", model_output_path,
                "-s", args.data_path,
                "--iteration", str(iter_num)
            ]
            print(f"Executing: {' '.join(render_cmd)}")
            result = subprocess.run(render_cmd)

            if result.returncode != 0:
                print(
                    f"--- Rendering failed with exit code {result.returncode}. Stopping pipeline. ---")
                sys.exit(1)

        print("--- Rendering Finished ---\n")

    # --- 3. Metrics Evaluation ---
    if not args.skip_metrics:
        print("\n--- Starting Metrics Evaluation ---")

        # Evaluate the final model checkpoint.
        eval_iterations = [args.iterations]

        for iter_num in eval_iterations:
            metrics_cmd = [
                sys.executable, "metrics.py",
                "-m", model_output_path,
                "-s", args.data_path,
                "-i", str(iter_num)
            ]
            print(f"Executing: {' '.join(metrics_cmd)}")
            result = subprocess.run(metrics_cmd)

            if result.returncode != 0:
                print(
                    f"--- Metrics evaluation failed with exit code {result.returncode}. Stopping pipeline. ---")
                sys.exit(1)

        print("--- Metrics Evaluation Finished ---\n")


if __name__ == "__main__":
    # Set up the argument parser.
    parser = ArgumentParser(
        description="Full evaluation script for audio rendering pipeline")

    # Path arguments.
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to the audio dataset (e.g., MeshRIR).")
    parser.add_argument("--output_path", default="./output",
                        type=str, help="Parent directory for all output models.")
    parser.add_argument("--model_name", required=True, type=str,
                        help="Name for the model and its output folder.")

    # Control arguments to skip parts of the pipeline.
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the training step.")
    parser.add_argument("--skip_rendering", action="store_true",
                        help="Skip the rendering step.")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Skip the metrics evaluation step.")

    # Training arguments.
    parser.add_argument("--iterations", default=30000,
                        type=int, help="Number of training iterations.")

    args = parser.parse_args()

    # Ensure the output path exists.
    os.makedirs(args.output_path, exist_ok=True)

    # Run the full pipeline.
    run_full_pipeline(args)
