import os
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils.general_utils import save_audio, build_rotation


def visualize_all(pred_freq, gt_freq, pred_time, gt_time, position_rx, position_tx, mode_set, save_dir, iteration, gaussians, sr, coord_min, coord_max):
    """Visualize and save various aspects of the acoustic scene for target receiver including signals, sound fields, Gaussian distributions, and audio waveforms.
    """
    # 1. Signal Plots (Time & Frequency Domain)
    visualize_signal(
        pred_freq=pred_freq,
        gt_freq=gt_freq,
        pred_time=pred_time,
        gt_time=gt_time,
        mode_set=mode_set,
        save_path=os.path.join(save_dir, f"signal_iter_{iteration}.png")
    )

    # 2. Sound Field Video (Time Domain)
    visualize_sound_field_video(
        gaussians=gaussians,
        position_rx=position_rx,
        position_tx=position_tx,
        coord_min=coord_min,
        coord_max=coord_max,
        sr=sr,
        save_path=os.path.join(save_dir, f"field_iter_{iteration}.gif")
    )

    # 3. Gaussian Spatial Distribution Video
    visualize_gaussian_spatial_video(
        gaussians=gaussians,
        position_rx=position_rx,
        position_tx=position_tx,
        coord_min=coord_min,
        coord_max=coord_max,
        sr=sr,
        save_path=os.path.join(save_dir, f"g_spatial_iter_{iteration}.gif")
    )

    # 4. Gaussian Stft-Path Distribution Plot
    visualize_gaussian_stft_path(
        gaussians=gaussians,
        position_rx=position_rx,
        position_tx=position_tx,
        sr=sr,
        save_path=os.path.join(save_dir, f"g_stft_iter_{iteration}.png")
    )

    # 5. Audio Waveform
    save_audio(
        waveform=pred_time,
        sr=sr,
        save_path=os.path.join(save_dir, f"audio_iter_{iteration}.wav")
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def cleanup():
    plt.close("all")
    gc.collect()
    torch.cuda.empty_cache()


def to_numpy(tensor):
    """Helper function to convert tensor to numpy array safely."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def plot_source(ax, rx, tx):
    rx = to_numpy(rx)
    tx = to_numpy(tx)
    ax.scatter(rx[0], rx[1], rx[2],
               c="#303030", marker='.', label='Receiver (Microphone)', zorder=1000, depthshade=False, s=150)
    ax.scatter(tx[0], tx[1], tx[2],
               c="#303030", marker='*', label='Transmitter (Speaker)', zorder=1000, depthshade=False, s=150)


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------


def visualize_geometry(train_rx, train_tx, test_rx, test_tx, save_path):
    """Visualize and save the geometry of receiver and transmitter positions.
    Modified from https://github.com/sh01k/MeshRIR/blob/main/irutilities.py
    """

    train_rx = to_numpy(train_rx)
    train_tx = to_numpy(train_tx)
    test_rx = to_numpy(test_rx)
    test_tx = to_numpy(test_tx)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Train Data
    ax.scatter(train_rx[:, 0], train_rx[:, 1], train_rx[:, 2],
               c='b', marker='.', label='Train Receiver (Microphone)', alpha=0.2)
    ax.scatter(train_tx[0, 0], train_tx[0, 1], train_tx[0, 2],
               c='b', marker='*', label='Train Transmitter (Speaker)', alpha=0.5, s=50)

    # Plot Test Data
    ax.scatter(test_rx[:, 0], test_rx[:, 1], test_rx[:, 2],
               c='r', marker='.', label='Test Receiver (Microphone)', alpha=0.2)
    ax.scatter(test_tx[0, 0], test_tx[0, 1], test_tx[0, 2],
               c='r', marker='*', label='Test Transmitter (Speaker)', alpha=0.5, s=50)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()

    # Set aspect ratio to be equal based on data range
    all_data = np.concatenate([train_rx, train_tx, test_rx, test_tx], axis=0)
    x_range = all_data[:, 0].max() - all_data[:, 0].min()
    y_range = all_data[:, 1].max() - all_data[:, 1].min()
    z_range = all_data[:, 2].max() - all_data[:, 2].min()
    ax.set_box_aspect((x_range, y_range, z_range))

    plt.title("Geometry of Receiver and Transmitter Positions")
    plt.savefig(save_path, dpi=300)
    cleanup()


def visualize_signal(pred_freq, gt_freq, pred_time, gt_time, mode_set, save_path):
    """Visualize and save the predicted and ground truth signals in both time and frequency domains.
    Modified from https://github.com/penn-waves-lab/AVR/blob/main/utils/logger.py
    """

    pred_freq = to_numpy(pred_freq)
    gt_freq = to_numpy(gt_freq)
    pred_time = to_numpy(pred_time)
    gt_time = to_numpy(gt_time)

    plt.figure(1, figsize=(16, 12))
    plt.suptitle(f"Signal Visualization ({mode_set} set)", fontsize=16)

    # Time domain
    plt.subplot(231)
    plt.title("Time Domain")
    plt.plot(pred_time, label='Predicted')
    plt.plot(gt_time, alpha=0.5, label='Ground Truth')
    handles, labels = plt.gca().get_legend_handles_labels()

    # Phase
    plt.subplot(232)
    plt.title("Phase")
    plt.plot(np.angle(pred_freq))
    plt.plot(np.angle(gt_freq), alpha=0.5)

    # Legend
    plt.subplot(233)
    plt.axis('off')
    plt.legend(handles, labels, loc='lower left', fontsize=15)

    # Magnitude
    plt.subplot(234)
    plt.title("Magnitude")
    plt.plot(np.abs(pred_freq))
    plt.plot(np.abs(gt_freq), alpha=0.5)
    plt.ylim(0)

    # Real part
    plt.subplot(235)
    plt.title("Real")
    plt.plot(np.real(pred_freq))
    plt.plot(np.real(gt_freq), alpha=0.5)

    # Imaginary part
    plt.subplot(236)
    plt.title("Imaginary")
    plt.plot(np.imag(pred_freq))
    plt.plot(np.imag(gt_freq), alpha=0.5)

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300)
    cleanup()


def visualize_sound_field_video(gaussians, position_rx, position_tx, coord_min, coord_max, sr, save_path):
    """Visualize and save the sound field video in 3D space (Time domain) using scatter plot.
    Modified from https://github.com/sh01k/MeshRIR/blob/main/irutilities.py
    """
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    # Get device
    device = gaussians.device

    # Define grid
    resolution = 1
    grid_vals = torch.arange(coord_min, coord_max, resolution, device='cpu')
    grid_points = torch.cartesian_prod(grid_vals, grid_vals, grid_vals)

    # Time settings
    n_frames = 50
    seq_len = gaussians.seq_len
    dt = max(1, seq_len // n_frames)
    seq_idxs = list(range(0, seq_len, dt))
    n_frames = len(seq_idxs)

    # Process in iterations to save memory
    pred_ir_list = []

    with torch.no_grad():
        for i in range(0, grid_points.shape[0]):
            pred_ir = gaussians(grid_points[None, i, :].to(device))

            # Keep only target time indices
            pred_ir = pred_ir[:, seq_idxs]
            pred_ir_list.append(to_numpy(pred_ir))

    pred_ir = np.concatenate(pred_ir_list, axis=0)
    grid_points = to_numpy(grid_points)

    # Sound pressure range
    vmin, vmax = -0.1, 0.1

    # Draw plots
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add Colorbar
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Sound Pressure', shrink=0.8, pad=0.1)

    # Set static elements once
    ax.set_xlim(coord_min, coord_max)
    ax.set_ylim(coord_min, coord_max)
    ax.set_zlim(coord_min, coord_max)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plot_source(ax, position_rx, position_tx)
    ax.legend()

    # Update function for animation
    scatter = None

    def update(frame):
        nonlocal scatter

        # Remove previous scatter plot if it exists
        if scatter is not None:
            scatter.remove()
            scatter = None

        # Calculate actual time
        seq_idx = seq_idxs[frame]
        current_time = seq_idx / sr

        # Filter by magnitude for visualization clarity
        data = pred_ir[:, frame]
        mask = np.abs(data) > vmax * 0.15
        if np.sum(mask) > 0:
            scatter = ax.scatter(grid_points[mask, 0], grid_points[mask, 1], grid_points[mask, 2],
                                 c=data[mask], cmap='RdBu', alpha=0.5, s=100, marker='s', vmin=vmin, vmax=vmax)

        ax.set_title(f'Sound Field 3D (Time: {current_time:.3f}s)')

        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames)
    anim.save(save_path, writer='pillow', fps=10)
    cleanup()


def visualize_gaussian_spatial_video(gaussians, position_rx, position_tx, coord_min, coord_max, sr, save_path, f_band=[125, 250, 500, 1000, 2000, 4000]):
    """Visualize and save the spatial distribution video of Gaussians with 3-sigma boundaries for each frequency band.
    """

    # Extract parameters
    device = gaussians.device
    with torch.no_grad():
        XYZ = gaussians.get_xyz
        T = gaussians.get_t
        F = gaussians.get_f
        S = gaussians.get_scaling
        O = gaussians.get_opacity
        R = build_rotation(gaussians._rotation[:, :4]).to(device)
        O = to_numpy(O)
        features = gaussians.get_features
        seq_len = gaussians.seq_len
        total_time = seq_len / sr
        t_len = gaussians.t_len
        span = gaussians.span

        # Denormalize xyztf
        XYZ = XYZ / 2 * span
        T = (T + 1) / 2 * total_time
        F = (F + 1) / 2 * (sr / 2)

        # Denormalize scaling
        S_xyz = S[:, :3] / 2 * span

        if gaussians.gaussian_version == 1:
            S_t, S_f = None, None
        elif gaussians.gaussian_version == 2:
            S_t, S_f = S[:, 3], None
        elif gaussians.gaussian_version == 3:
            S_t, S_f = None, S[:, 3]
        elif gaussians.gaussian_version == 4:
            S_t, S_f = S[:, 3], S[:, 4]

        if S_t is not None:
            S_t = S_t / 2 * total_time
        if S_f is not None:
            S_f = S_f / 2 * (sr / 2)

        # Calculate colors based on DC Component Magnitude of Complex SH Feature
        features = to_numpy(torch.abs(features[:, 0]))
        feat_min, feat_max = 0, 2

    # Setup subplots for each frequency band
    n_bands = len(f_band)
    n_cols = (n_bands + 1) // 2
    n_rows = 2

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.subplots_adjust(right=0.80)

    # Feature Magnitude Colorbar
    cbar_ax_feat = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    sm_feat = plt.cm.ScalarMappable(
        cmap='coolwarm', norm=plt.Normalize(vmin=feat_min, vmax=feat_max))
    sm_feat.set_array([])
    fig.colorbar(sm_feat, cax=cbar_ax_feat, shrink=0.8,
                 label='Feature Magnitude')

    # Opacity Colorbar
    cbar_ax_op = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm_op = plt.cm.ScalarMappable(
        cmap='gray_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm_op.set_array([])
    fig.colorbar(sm_op, cax=cbar_ax_op, shrink=0.8, label='Opacity')

    axes = []

    for i, freq in enumerate(f_band):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        ax.set_xlim(coord_min, coord_max)
        ax.set_ylim(coord_min, coord_max)
        ax.set_zlim(coord_min, coord_max)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"{freq} Hz")
        plot_source(ax, position_rx, position_tx)
        axes.append(ax)

    # Time settings
    n_frames = 50
    dt = max(1, t_len // n_frames)
    t_idxs = list(range(0, t_len, dt))
    n_frames = len(t_idxs)

    # Sphere for ellipsoid construction
    u = torch.linspace(0, 2 * np.pi, 8, device=device)
    v = torch.linspace(0, np.pi, 8, device=device)
    x_sphere = torch.outer(torch.cos(u), torch.sin(v))
    y_sphere = torch.outer(torch.sin(u), torch.sin(v))
    z_sphere = torch.outer(torch.ones_like(u), torch.cos(v))
    sphere_points = torch.stack(
        [x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])

    lines_dict = {freq: [] for freq in f_band}

    def update(frame):
        # Calculate actual time
        current_time = t_idxs[frame] / t_len * total_time
        fig.suptitle(
            f"Gaussian Spatial Distribution (Time: {current_time:.3f}s)")

        # Remove previous lines
        for freq in f_band:
            for line in lines_dict[freq]:
                line.remove()
            lines_dict[freq].clear()

        # Calculate squared Mahalanobis distance for time
        if S_t is not None:
            dist_t_sq = ((T - current_time) / (S_t + 1e-12)) ** 2
        else:
            dist_t_sq = 0

        # Iterate over frequency bands
        for ax, freq in zip(axes, f_band):

            # Calculate squared Mahalanobis distance for frequency
            if S_f is not None:
                dist_f_sq = ((F - freq) / (S_f + 1e-12)) ** 2
            else:
                f_min = freq / np.sqrt(2)
                f_max = freq * np.sqrt(2)
                dist_f_sq = torch.where((F >= f_min) & (F <= f_max), torch.tensor(
                    0.0, device=device), torch.tensor(100.0, device=device))

            # Calculate remaining squared radius for spatial dimensions
            dist_xyz_sq = 9 - dist_t_sq - dist_f_sq
            mask = dist_xyz_sq > 0
            idxs = torch.where(mask)[0]

            if len(idxs) > 0:
                chunk_size = 5000
                num_points = len(idxs)
                n_grid = x_sphere.shape[0]

                # Collect segments and colors for all chunks
                collected_segments = []
                collected_colors = []

                for i in range(0, num_points, chunk_size):
                    chunk_end = min(i + chunk_size, num_points)
                    chunk_idxs = idxs[i:chunk_end]

                    # (K, 1, 1)
                    r_factors = torch.sqrt(dist_xyz_sq[chunk_idxs])

                    # (K, 3, 1) * (1, 3, M) -> (K, 3, M)
                    scaled_spheres = S_xyz[chunk_idxs, :,
                                           None] * sphere_points[None, :, :]

                    # (K, 3, 3) @ (K, 3, M) -> (K, 3, M)
                    rotated_spheres = R[chunk_idxs] @ scaled_spheres
                    # (K, 3, 1) + (K, 3, M) -> (K, 3, M)

                    points = XYZ[chunk_idxs, :, None] + \
                        r_factors[:, None, None] * rotated_spheres

                    points = to_numpy(points)
                    chunk_idxs = to_numpy(chunk_idxs)

                    # (K, 3, 8, 8) -> (K, 8, 8, 3)
                    points = points.reshape(
                        len(chunk_idxs), 3, n_grid, n_grid).transpose(0, 2, 3, 1)

                    # Extract horizontal (rows) and vertical (cols) line segments
                    rows = points.reshape(-1, n_grid, 3)
                    cols = points.transpose(0, 2, 1, 3).reshape(-1, n_grid, 3)

                    # Concatenate all segments
                    all_segments = np.concatenate([rows, cols], axis=0)

                    # Apply colors and opacity uniformly
                    sphere_colors = sm_feat.to_rgba(
                        features[chunk_idxs].flatten())
                    sphere_colors[:, 3] = O[chunk_idxs].flatten()

                    # Repeat color array for each line in the sphere (n_grid)
                    c_repeated = np.repeat(sphere_colors, n_grid, axis=0)
                    all_colors = np.concatenate(
                        [c_repeated, c_repeated], axis=0)

                    collected_segments.append(all_segments)
                    collected_colors.append(all_colors)

                # Merge all chunk data into one collection to reduce Matplotlib overhead
                if collected_segments:
                    final_segments = np.concatenate(collected_segments, axis=0)
                    final_colors = np.concatenate(collected_colors, axis=0)

                    # Create and add collection
                    lc = Line3DCollection(
                        final_segments, colors=final_colors, linewidth=0.1)
                    ax.add_collection(lc)
                    lines_dict[freq].append(lc)

        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames)
    anim.save(save_path, writer='pillow', fps=10)
    cleanup()


def visualize_gaussian_stft_path(gaussians, position_rx, position_tx, sr, save_path):
    """Visualize and save the STFT(tf)-path(xyz) distribution of Gaussians.
    """
    device = gaussians.device

    # Extract parameters
    with torch.no_grad():
        XYZ = gaussians.get_xyz
        T = gaussians.get_t
        F = gaussians.get_f
        S = gaussians.get_scaling
        O = gaussians.get_opacity
        R = build_rotation(gaussians._rotation[:, :4]).to(device)
        features = gaussians.get_features

        # Denormalize
        span = gaussians.span
        seq_len = gaussians.seq_len
        total_time = seq_len / sr

        XYZ = XYZ / 2 * span
        T = (T + 1) / 2 * total_time
        F = (F + 1) / 2 * (sr / 2)

        # Denormalize scaling
        S_xyz = S[:, :3] / 2 * span

        if gaussians.gaussian_version == 1:
            S_t, S_f = None, None
        elif gaussians.gaussian_version == 2:
            S_t, S_f = S[:, 3], None
        elif gaussians.gaussian_version == 3:
            S_t, S_f = None, S[:, 3]
        elif gaussians.gaussian_version == 4:
            S_t, S_f = S[:, 3], S[:, 4]

        if S_t is not None:
            S_t = S_t / 2 * total_time
        else:
            S_t = torch.zeros_like(T)

        if S_f is not None:
            S_f = S_f / 2 * (sr / 2)
        else:
            S_f = torch.zeros_like(F)

        # Calculate colors based on DC Component Magnitude of Complex SH Feature
        features = to_numpy(torch.abs(features[:, 0]))
        feat_min, feat_max = 0, 2

        # Vector from tx to rx
        rx_t = position_rx.to(device)
        tx_t = position_tx.to(device)
        vec_tx_rx = rx_t - tx_t
        dist_tx_rx = torch.norm(vec_tx_rx)
        unit_vec = vec_tx_rx / (dist_tx_rx + 1e-8)  # (3,)

        # Project spatial covariance to path axis
        v = R.transpose(1, 2) @ unit_vec

        # Projected radius (1-sigma)
        r_p = torch.sqrt(torch.sum((v * S_xyz)**2, dim=1))  # (N,)

        # To numpy
        XYZ = to_numpy(XYZ)
        T = to_numpy(T).flatten()
        F = to_numpy(F).flatten()
        O = to_numpy(O).flatten()
        r_p = to_numpy(r_p).flatten()
        S_t = to_numpy(S_t).flatten()
        S_f = to_numpy(S_f).flatten()

        tx = to_numpy(position_tx)
        dist_tx_rx = float(to_numpy(dist_tx_rx))

    # Calculate Path Projection Center
    vec_tx_g = XYZ - tx
    unit_vec_np = to_numpy(unit_vec)
    path_proj = np.dot(vec_tx_g, unit_vec_np)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter low opacity for clarity
    mask = O > 0.01
    idxs = np.where(mask)[0]

    # Create RGBA colors based on feature magnitude and opacity
    norm = plt.Normalize(vmin=0, vmax=2)
    cmap = plt.cm.coolwarm

    # Sphere for ellipsoid construction (8x8 grid)
    u = np.linspace(0, 2 * np.pi, 8)
    v = np.linspace(0, np.pi, 8)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    # Flatten for easier manipulation: (3, 64)
    sphere_points = np.stack(
        [x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
    n_grid = 8

    if len(idxs) > 0:
        chunk_size = 5000
        num_points = len(idxs)

        collected_segments = []
        collected_colors = []

        for i in range(0, num_points, chunk_size):
            chunk_idxs = idxs[i:i+chunk_size]

            # Centers: (K, 3) -> (Path, Time, Freq)
            centers = np.stack(
                [path_proj[chunk_idxs], T[chunk_idxs], F[chunk_idxs]], axis=1)

            # Radii: (K, 3) -> (r_p, S_t, S_f) * 3 (for 3-sigma)
            radii = np.stack(
                [r_p[chunk_idxs], S_t[chunk_idxs], S_f[chunk_idxs]], axis=1) * 3

            # Transform sphere points
            # (K, 3, 1) * (1, 3, M) -> (K, 3, M)
            scaled_spheres = radii[:, :, None] * sphere_points[None, :, :]

            # Translate
            # (K, 3, 1) + (K, 3, M) -> (K, 3, M)
            points = centers[:, :, None] + scaled_spheres

            # Reshape to grid for line generation
            # (K, 3, 8, 8) -> (K, 8, 8, 3)
            points = points.reshape(
                len(chunk_idxs), 3, n_grid, n_grid).transpose(0, 2, 3, 1)

            # Generate segments
            rows = points.reshape(-1, n_grid, 3)
            cols = points.transpose(0, 2, 1, 3).reshape(-1, n_grid, 3)
            all_segments = np.concatenate([rows, cols], axis=0)

            # Colors
            c_vals = features[chunk_idxs]
            o_vals = O[chunk_idxs]

            rgba = cmap(norm(c_vals))
            rgba[:, 3] = np.clip(o_vals, 0, 1)

            # Repeat colors for segments
            c_repeated = np.repeat(rgba, 2 * n_grid, axis=0)

            collected_segments.append(all_segments)
            collected_colors.append(c_repeated)

        if collected_segments:
            final_segments = np.concatenate(collected_segments, axis=0)
            final_colors = np.concatenate(collected_colors, axis=0)

            lc = Line3DCollection(
                final_segments, colors=final_colors, linewidth=0.1)
            ax.add_collection(lc)

    # Mark Tx and Rx positions on the Path axis
    ax.plot([0, 0], [0, total_time], [0, 0], 'k--', label='Tx Plane')
    ax.plot([dist_tx_rx, dist_tx_rx], [0, total_time],
            [0, 0], 'r--', label='Rx Plane')

    ax.set_xlabel('Path (Tx -> Rx) [m]')
    ax.set_ylabel('Time [s]')
    ax.set_zlabel('Frequency [Hz]')
    ax.set_title(
        'Gaussian Tx -> Rx Path-Time-Frequency Distribution')

    # Set limits
    ax.set_xlim(path_proj.min(), path_proj.max())
    ax.set_ylim(0, total_time)
    ax.set_zlim(0, sr/2)

    # Feature Magnitude Colorbar
    cbar_ax_feat = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    sm_feat = plt.cm.ScalarMappable(
        cmap='coolwarm', norm=plt.Normalize(vmin=feat_min, vmax=feat_max))
    sm_feat.set_array([])
    fig.colorbar(sm_feat, cax=cbar_ax_feat, shrink=0.8,
                 label='Feature Magnitude')

    # Opacity Colorbar
    cbar_ax_op = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    sm_op = plt.cm.ScalarMappable(
        cmap='gray_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm_op.set_array([])
    fig.colorbar(sm_op, cax=cbar_ax_op, shrink=0.8, label='Opacity')

    ax.legend()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure
    plt.savefig(save_path, dpi=300)
    cleanup()
