"""generating input and target"""

import numpy as np
import torch.utils.data as data


class SineWave(data.Dataset):
    def __init__(self, time_length=200, freq_range=3):
        self.time_length = time_length
        self.freq_range = freq_range

    def __len__(self):
        return 200

    def __getitem__(self, item):
        freq = np.random.randint(1, self.freq_range + 1)
        const_signal = np.repeat(freq / self.freq_range + 0.25, self.time_length)
        const_signal = np.expand_dims(const_signal, axis=1)
        # print(const_signal.shape)
        t = np.arange(0, self.time_length*0.025, 0.025)
        target = np.sin(freq * t)
        target = np.expand_dims(target, axis=1)
        return const_signal, target


class Torus(data.Dataset):
    def __init__(self, time_length=50, freq_range=3):
        self.time_length = time_length
        self.freq_range = freq_range

    def __len__(self):
        return 200

    def __getitem__(self, item):
        freq1 = np.random.randint(1, self.freq_range + 1)
        freq2 = 2
        const_signal1 = np.repeat(freq1 / self.freq_range + 0.25, self.time_length)
        const_signal2 = np.repeat(freq2 / self.freq_range + 0.25, self.time_length)
        const_signal1 = np.expand_dims(const_signal1, axis=1)
        const_signal2 = np.expand_dims(const_signal2, axis=1)
        const_signal = np.concatenate((const_signal1, const_signal2), axis=1)
        # print(const_signal.shape)
        t = np.arange(0, self.time_length * 0.3, 0.3)
        target = 0.6 * np.sin(1.8 * freq1 * t) + 0.8 * np.sin(0.3 * freq2 * t)
        target = np.expand_dims(target, axis=1)
        return const_signal, target
