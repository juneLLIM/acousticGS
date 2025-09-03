import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import librosa
import math


class WaveDataset(Dataset):  # Renamed from WaveLoader
    def __init__(self, config, eval=False, iteration=None):
        """DataLoader initializations, can load three different sets together

        Parameters
        ----------
        eval : bool, optional
            flag to determine training or testing set
        """

        self.wave_chunks = []
        self.positions_rx = []
        self.positions_tx = []
        self.rotations_tx = []

        self.wave_max = float('-inf')
        self.wave_min = float('inf')
        self.position_max = np.array(
            [float('-inf'), float('-inf'), float('-inf')])
        self.position_min = np.array(
            [float('inf'), float('inf'), float('inf')])

        self.dataset_type = config.path.dataset_type
        self.eval = eval

        # load three different datasets seperately
        if self.dataset_type == 'MeshRIR':
            self.load_mesh_rir(config.path.data, eval,
                               config.audio.seq_len, config.audio.fs)
        elif self.dataset_type == 'RAF':
            self.load_raf(config.path.data, eval,
                          config.audio.seq_len, config.audio.fs)
        elif self.dataset_type == 'Simu':
            self.load_simu(config.path.data, eval,
                           config.audio.seq_len, config.audio.fs)
        else:
            raise ValueError("Unsupported dataset type")

        # Convert lists to tensors for faster processing in __getitem__
        self.wave_chunks = torch.tensor(
            np.array(self.wave_chunks), dtype=torch.float32)
        self.positions_rx = torch.tensor(
            np.array(self.positions_rx), dtype=torch.float32)
        self.positions_tx = torch.tensor(
            np.array(self.positions_tx), dtype=torch.float32)
        if self.rotations_tx:
            self.rotations_tx = torch.tensor(
                np.array(self.rotations_tx), dtype=torch.float32)

        self.iteration = iteration if iteration else len(self.wave_chunks)
        self.length = len(self.wave_chunks)

    def load_mesh_rir(self, base_folder, eval, seq_len, fs=24000):
        """ Load MeshRIR datasets
        """
        down_sample_rate = 48000 // fs
        self.default_st_idx = int(9100 / down_sample_rate)

        if eval:
            wave_folder = os.path.join(base_folder, 'test')
        else:
            wave_folder = os.path.join(base_folder, 'train')

        filenames = [f for f in os.listdir(wave_folder) if f.endswith('.npy')]
        filenames.sort()

        rx_pos = np.load(os.path.join(base_folder, 'pos_mic.npy'))
        tx_pos = np.load(os.path.join(base_folder, 'pos_src.npy'))[0]

        for filename in filenames:
            audio_data = np.load(os.path.join(wave_folder, filename))[
                0, ::down_sample_rate]  # first resample the IR data
            # index the IR data.
            audio_data = audio_data[self.default_st_idx:self.default_st_idx+seq_len]

            file_ind = int(filename.split('_')[1].split('.')[0])
            position_rx = rx_pos[file_ind]
            position_tx = tx_pos

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(audio_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)

    def load_simu(self, base_folder, eval, seq_len, fs):
        """ Load simulation datasets
        """
        filenames = [f for f in os.listdir(base_folder) if f.endswith('.npz')]
        filenames.sort()

        if eval:
            filenames = filenames[int(0.9 * len(filenames)):]  # testing
        else:
            filenames = filenames[:int(0.9 * len(filenames))]  # training

        for filename in filenames:
            meta_data = np.load(os.path.join(base_folder, filename))
            audio_data = meta_data['ir'][:seq_len]

            position_rx = meta_data['position_rx']
            position_tx = meta_data['position_tx']

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(audio_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)

    def load_raf(self, base_folder, eval, seq_len, fs):
        """ Load RAF datasets
        """
        folderpaths = glob.glob(f"{base_folder}/*")
        folderpaths.sort()

        if eval:
            folderpaths = glob.glob(f"{base_folder}/test/*")
        else:
            folderpaths = glob.glob(f"{base_folder}/train/*")
        folderpaths.sort()

        for folderpath in folderpaths:
            rir_path = os.path.join(folderpath, "rir.wav")
            audio_data, _ = librosa.load(rir_path, sr=None, mono=True)
            audio_data = audio_data[:seq_len * int(48000 / fs):int(48000 / fs)]

            position_rx = self.load_position(
                os.path.join(folderpath, "rx_pos.txt"))
            position_tx, rotation_tx = self.load_tx_info(
                os.path.join(folderpath, "tx_pos.txt"))

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(audio_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)
            self.rotations_tx.append(rotation_tx)

    def load_position(self, file_path):
        position = []
        with open(file_path, 'r') as file:
            for line in file:
                position.extend([float(num) for num in line.split(',')])
        return np.array(position)[[0, 2, 1]]

    def load_tx_info(self, file_path):
        tx_info = []
        with open(file_path, 'r') as file:
            for line in file:
                tx_info.extend([float(num) for num in line.split(',')])
        tx_info = np.array(tx_info)
        rotation_tx = tx_info[:4]
        rotation_tx = quaternion_to_direction_vector(rotation_tx)
        position_tx = np.array(tx_info[4:])[[0, 2, 1]]
        return position_tx, rotation_tx

    def update_min_max(self, audio_data, position_rx):
        self.wave_max = max(self.wave_max, audio_data.max())
        self.wave_min = min(self.wave_min, audio_data.min())
        self.position_max = np.maximum(self.position_max, position_rx)
        self.position_min = np.minimum(self.position_min, position_rx)

    def __len__(self):
        return self.iteration

    def __getitem__(self, idx):
        idx = idx % self.length
        wave_signal = self.wave_chunks[idx]
        position_rx = self.positions_rx[idx]
        position_tx = self.positions_tx[idx]

        if not self.eval and self.dataset_type == 'RAF':
            position_rx = position_rx + torch.randn_like(position_rx) * 0.1
            position_tx = position_tx + torch.randn_like(position_tx) * 0.1

        if self.dataset_type == 'RAF':
            rotation_tx = self.rotations_tx[idx]
            return wave_signal, position_rx, position_tx, rotation_tx
        else:
            return wave_signal, position_rx, position_tx


def quaternion_to_direction_vector(q):
    """Convert a quaternion to direction vectors in Cartesian coordinates

    Parameters
    ----------
    q : Quaternion, given as a Tensor [x, y, z, w].

    Returnsdata
    -------
    Direction vectors as pts_x, pts_y, pts_z
    """

    x, y, z, w = q

    # Convert quaternion to forward direction vector
    fwd_x = 2 * (x*z + w*y)
    fwd_y = 2 * (y*z - w*x)
    fwd_z = 1 - 2 * (x*x + y*y)

    # Normalize the vector (in case it's not exactly 1 due to numerical precision)
    norm = math.sqrt(fwd_x**2 + 0**2 + fwd_z**2)

    return np.array([-fwd_x / norm, -fwd_z / norm, 0])


class DummyDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        self.data = self.batchify(args)

    def __getitem__(self, index):
        return None, *[batch[index] for batch in self.data]

    def batchify(self, *args):
        return [torch.stack(arg) for arg in args]
