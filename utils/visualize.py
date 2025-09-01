# Reference: https://github.com/sh01k/MeshRIR/blob/main/irutilities.py

from pathlib import Path
import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch


def log_inference_figure(gt_time_sig, pred_time_sig, metrics, save_dir=None):
    """show the estimated and ground truth signal, show the signal metric on the figure

    Parameters
    ----------
    gt_time_sig : np.array
        ground truth signal
    pred_time_sig : np.array
        estimated signal
    metric : dictionary
        metric infomation
    save_dir : string, optional
        image save directory, by default None (not save)
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gt_time_sig, c='b')
    ax.plot(pred_time_sig, c='r', alpha=0.8)
    ax.set_ylim(-np.max(np.abs(gt_time_sig))*1,
                np.max(np.abs(gt_time_sig))*1)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    font_size = 26
    text_pos_x = 0.65
    text_pos_y = 0.10

    plt.text(text_pos_x, 0.50 - text_pos_y, f"Angle err: {metrics['Angle']:.2f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.44 - text_pos_y, f"Amp. err: {metrics['Amplitude']:.3f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.38 - text_pos_y, f"Env. err: {metrics['Envelope']:.3f}", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.32 - text_pos_y, f"T60 err: {metrics['T60'] * 100:.2f}%", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.26 - text_pos_y, f"C50 err: {metrics['C50']:.2f} db", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)
    plt.text(text_pos_x, 0.20 - text_pos_y, f"EDT err: {metrics['EDT']:.3f} s", transform=plt.gca(
    ).transAxes, verticalalignment='top', fontsize=font_size)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()

    if save_dir is not None:
        # Change the path and file name as needed
        plt.savefig(save_dir, dpi=300, pad_inches=0)
    plt.close("all")

    return


def plot_and_save_figure(pred_freq, gt_freq, pred_time, gt_time, position_rx, position_tx, mode_set, save_path):

    plt.figure(1, figsize=(16, 12))
    plt.suptitle(f"{mode_set} set")
    plt.subplot(231)
    plt.title("Real")
    plt.plot(np.real(pred_freq.detach().cpu().numpy().flatten()))
    plt.plot(np.real(gt_freq.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)

    plt.subplot(234)
    plt.title("Imaginary")
    plt.plot(np.imag(pred_freq.detach().cpu().numpy().flatten()))
    plt.plot(np.imag(gt_freq.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)

    plt.subplot(232)
    plt.plot(pred_time.detach().cpu().numpy().flatten())
    plt.plot(gt_time.detach().cpu().numpy().flatten(), alpha=0.5)

    plt.subplot(235)
    plt.scatter(position_rx[0], position_rx[1], c='b')
    plt.scatter(position_tx[0], position_tx[1], c='r')
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(233)
    plt.plot(np.abs(pred_freq.detach().cpu().numpy().flatten()))
    plt.plot(np.abs(gt_freq.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)
    plt.ylim(0)

    plt.subplot(236)
    plt.plot(np.angle(pred_freq.detach().cpu().numpy().flatten()))
    plt.plot(np.angle(gt_freq.type(
        torch.complex64).flatten().cpu().numpy()), alpha=0.5)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close("all")


def loadIR(sessionPath):
    """Load impulse response (IR) data

    Parameters
    ------
    sessionPath: Path to IR folder

    Returns
    ------
    pos_mic: Microphone positions of shape (numMic, 3)
    pos_src: Source positions of shape (numSrc, 3)
    fullIR: IR data of shape (numSrc, numMic, irLen)
    """
    pos_mic = np.load(sessionPath.joinpath("pos_mic.npy"))
    pos_src = np.load(sessionPath.joinpath("pos_src.npy"))

    numMic = pos_mic.shape[0]

    allIR = []
    irIndices = []
    for f in sessionPath.iterdir():
        if not f.is_dir():
            if f.stem.startswith("ir_"):
                allIR.append(np.load(f))
                irIndices.append(int(f.stem.split("_")[-1]))

    assert (len(allIR) == numMic)
    numSrc = allIR[0].shape[0]
    irLen = allIR[0].shape[-1]
    fullIR = np.zeros((numSrc, numMic, irLen))
    for i, ir in enumerate(allIR):
        assert (ir.shape[0] == numSrc)
        assert (ir.shape[-1] == irLen)
        fullIR[:, irIndices[i], :] = ir

    return pos_mic, pos_src, fullIR


def sortIR(pos, ir, numXY, posX=None, posY=None):
    """Sort IR data into 2D rectangular shape
    """
    if (posX is None) or (posY is None):
        posX = np.unique(pos[:, 0].round(4))
        posY = np.unique(pos[:, 1].round(4))
    sortIdx = np.zeros((numXY[0], numXY[1]), dtype=int)

    for i in range(numXY[1]):
        xIdx = np.where(np.isclose(pos[:, 1], posY[i]))[0]
        sorter = np.argsort(pos[xIdx, 0])
        xIdxSort = xIdx[sorter]
        sortIdx[:, i] = xIdxSort

    sortPos = pos[sortIdx, :]
    sortIR = ir[:, sortIdx, :]
    return sortPos, sortIR, sortIdx


def sortIR3(pos, ir, numXYZ, posX=None, posY=None, posZ=None):
    """Sort IR data into 3D cuboid shape
    """
    if (posX is None) & (posY is None) & (posZ is None):
        posX = np.unique(pos[:, 0].round(4))
        posY = np.unique(pos[:, 1].round(4))
        posZ = np.unique(pos[:, 2].round(4))
    sortIdx = np.zeros((numXYZ[0], numXYZ[1], numXYZ[2]), dtype=int)

    for i in range(numXYZ[2]):
        for j in range(numXYZ[1]):
            xIdx = np.where(np.isclose(
                pos[:, 1], posY[j]) & np.isclose(pos[:, 2], posZ[i]))[0]
            sorter = np.argsort(pos[xIdx, 0])
            xIdxSort = xIdx[sorter]
            sortIdx[:, j, i] = xIdxSort

    sortPos = pos[sortIdx, :]
    sortIR = ir[:, sortIdx, :]
    return sortPos, sortIR, sortIdx


def extract_plane(pos, ir, z):
    """Extract IR data on the plane at z
    """
    z_list = pos[:, 2]
    pos_z_idx = np.where(z_list == z)[0].tolist()

    pos_z = pos[pos_z_idx, :]
    ir_z = ir[:, pos_z_idx, :]

    return pos_z, ir_z


def reverbParams(ir, samplerate):
    """Compute reverberation parameters
    Returns
    ------
    t60: Reverberation time RT60
    energy: Energy decay curve
    line: Regression line of energy decay curve
    """
    t = np.arange(ir.shape[0]) / samplerate
    energy = 10.0 * np.log10(np.cumsum(ir[::-1]**2)[::-1]/np.sum(ir**2))

    # Linear regression parameters for computing RT60
    init_db = -5
    end_db = -25
    factor = 3.0

    energy_init = energy[np.abs(energy - init_db).argmin()]
    energy_end = energy[np.abs(energy - end_db).argmin()]
    init_sample = np.where(energy == energy_init)[0][0]
    end_sample = np.where(energy == energy_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / samplerate
    y = energy[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]
    line = slope * t + intercept

    db_regress_init = (init_db - intercept) / slope
    db_regress_end = (end_db - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)

    return t60, energy, line


def irPlots(ir, samplerate):
    """Plot impulse response
    """
    t = np.arange(ir.shape[0]) / samplerate

    rt60, energy_curve, energy_line = reverbParams(ir, samplerate)
    print("RT60 (ms): ", '{:.1f}'.format(rt60*1000))

    f_spec, t_spec, spec = signal.spectrogram(ir, samplerate, nperseg=512)

    # IR
    plt.plot(t, ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Energy decay curve
    # plt.plot(t, energy_curve)
    # plt.plot(t, energy_line, linestyle="--")
    # plt.ylim(-70, 5)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Energy (dB)')
    # plt.show()

    # Spectrogram
    # color = plt.pcolormesh(t_spec, f_spec, 20*np.log10(spec), vmin=-250, shading='auto')
    # cbar=plt.colorbar(color)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # cbar.set_label('Power (dB)')
    # plt.show()


def plotWave(x, y, ir, tIdx=None):
    """Plot instantaneous pressure distribution
    """
    if tIdx is None:
        tIdx, _ = findPeak(ir, 0)
        print("Time (sample):", tIdx)

    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    color = plt.pcolormesh(
        xx, yy, ir[:, :, tIdx].T, cmap='RdBu', shading='auto')
    ax.set_aspect('equal')
    cbar = plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()


def plotWaveFronts(x, ir, samplerate, xy='x'):
    """Plot impulse responses along the line at x
    """
    tIdxMin, tIdxMax = findPeak(ir)
    t = np.arange(tIdxMin, tIdxMax)/samplerate
    if xy == 'x':
        ir_plt = np.squeeze(ir[:, 0, tIdxMin:tIdxMax])
    elif xy == 'y':
        ir_plt = np.squeeze(ir[0, :, tIdxMin:tIdxMax])
    else:
        raise ValueError()

    xx, yy = np.meshgrid(t, x)
    fig, ax = plt.subplots()
    color = plt.pcolormesh(xx, yy, ir_plt, cmap='RdBu', shading='auto')
    cbar = plt.colorbar(color)
    cbar.set_label('Amplitude')
    plt.xlabel('Time (s)')
    if xy == 'x':
        plt.ylabel('x (m)')
    elif xy == 'y':
        plt.ylabel('y (m)')
    plt.show()


def findPeak(ir, preBuffer=100, tailBuffer=100):
    """Find time sample of peak amplitude
    """
    peakIdx = np.argmax(np.abs(ir), axis=-1)
    minPeakIdx = np.min(peakIdx)
    maxPeakIdx = np.max(peakIdx)
    return minPeakIdx-preBuffer, maxPeakIdx+tailBuffer


def drawGeometry(posSrc, posMic):
    """Plot geometry of sources and microphones 
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(posMic[:, 0], posMic[:, 1], posMic[:, 2], marker='.')
    ax.scatter3D(posSrc[:, 0], posSrc[:, 1], posSrc[:, 2], marker='*')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.show()


def movWave(sessionPath, x, y, ir, samplerate, start=None, end=None, downSampling=None):
    """Generate movie of pressure field
    """
    if (start is None) or (end is None):
        start, end = findPeak(ir)

    if downSampling is not None:
        ir = signal.resample_poly(ir, up=1, down=downSampling, axis=-1)
        samplerate = samplerate // downSampling

    maxVal = np.max(np.abs(ir))
    # maxVal = 0.2

    plt.rcParams["font.size"] = 14

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(ir[..., start].T, vmin=-maxVal, vmax=maxVal,
                    cmap='RdBu', origin='lower', interpolation='none')
    cbar = fig.colorbar(cax)
    ax.set_xticks(np.arange(0, x.shape[0], 4))
    ax.set_xticklabels(x[0::4])
    ax.set_yticks(np.arange(0, y.shape[0], 4))
    ax.set_yticklabels(y[0::4])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def animate(i):
        cax.set_array(ir[..., start+i].T)
        currentTime = (i+start) / samplerate
        ax.set_title("Sample: " + str(i) + ", Time: " +
                     "{:.3f}".format(currentTime), fontsize=14)
        return cax,

    anim = animation.FuncAnimation(
        fig=fig, func=animate, interval=15, frames=end-start-1, blit=True)
    # anim = animation.FuncAnimation(fig, animate, interval=200, frames=end-start-1, blit=True)

    plt.show()

    # anim.save(sessionPath.joinpath("wave_mov.mp4"), writer='ffmpeg', fps=15, bitrate=1800)
    anim.save(sessionPath.joinpath("wave_mov.gif"),
              writer='imagemagick', fps=15, bitrate=1800)


def visualize_gaussian_xyt_3d(gaussians, save_path=None):
    # Visualize the distribution of Gaussians in 3D spacetime (X-Y-T).
    # Position is (x, y, t), while color and alpha represent opacity.
    xyzt = gaussians.get_xyzt.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy().flatten()

    x, y, t = xyzt[:, 0], xyzt[:, 1], xyzt[:, 3]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Manually create RGBA colors from the colormap and opacities
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=opacities.min(), vmax=opacities.max())
    rgba_colors = cmap(norm(opacities))
    rgba_colors[:, 3] = opacities  # Set the alpha channel

    # Create a 3D scatter plot with the combined RGBA colors
    scatter = ax.scatter(x, y, t, c=rgba_colors, s=15, edgecolors='none')

    # Create a colorbar that represents the opacity values
    cbar = fig.colorbar(cm.ScalarMappable(
        norm=norm, cmap=cmap), ax=ax, shrink=0.6)
    cbar.set_label('Opacity')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Time Delay (t)")
    ax.set_title("3D Gaussian Distribution in Spacetime (XY-T)")
    ax.grid(True)

    # Set a better viewing angle for 3D visualization.
    ax.view_init(elev=20., azim=-65)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"3D visualization saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    sessionName = "S32-M441_npy"  # "S1-M3969_npy"
    sessionPath = Path(__file__).parent.joinpath(sessionName)

    # Load files
    posMic, posSrc, ir = loadIR(sessionPath)

    # Sampling rate
    samplerate = 48000
    srcIdx = 0
    micIdx = 0
    print("Source position (m): ", posSrc[srcIdx, :])
    print("Mic position (m): ", posMic[micIdx, :])

    # Geometry
    drawGeometry(posSrc, posMic)

    # IR plots
    ir_plt = ir[srcIdx, micIdx, :]
    irPlots(ir_plt, samplerate)

    # Extract plane
    z = 0.0
    posMic_z, ir_z = extract_plane(posMic, ir, z)
    posMicX = np.unique(posMic_z[:, 0].round(4))
    posMicY = np.unique(posMic_z[:, 1].round(4))
    numXY = (posMicX.shape[0], posMicX.shape[0])
    posMicXY, irXY, _ = sortIR(posMic_z, ir_z, numXY, posMicX, posMicY)

    # Lowpass filter
    maxFreq = 600
    h = signal.firwin(numtaps=64, cutoff=maxFreq, fs=samplerate)
    irXY_lp = signal.filtfilt(h, 1, irXY[srcIdx, :, :, :], axis=-1)

    # Wave image
    plotWave(posMicX, posMicY, irXY_lp)
    # plotWaveFronts(posMicX, irXY_lp, samplerate)

    # Wave movie
    # movWave(sessionPath, posMicX, posMicY, irXY_lp, samplerate)
