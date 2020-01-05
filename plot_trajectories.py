"""plot trajectories and fixed points in the PCA space."""

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from analyzer import FixedPoint
from model import RecurrentNeuralNetwork


class Torus(object):
    def __init__(self, time_length=50, freq_range=3):
        self.time_length = time_length
        self.freq_range = freq_range

    def __len__(self):
        return 200

    def getitem(self):
        freq1 = np.random.randint(1, self.freq_range + 1)
        freq2 = 2
        const_signal1 = np.repeat(freq1 / self.freq_range + 0.25, self.time_length)
        const_signal2 = np.repeat(freq2 / self.freq_range + 0.25, self.time_length)
        const_signal1 = np.expand_dims(const_signal1, axis=1)
        const_signal2 = np.expand_dims(const_signal2, axis=1)
        const_signal = np.concatenate((const_signal1, const_signal2), axis=1)
        t = np.arange(0, self.time_length*0.3, 0.3)
        target = 0.6 * np.sin(2.2 * freq1 * t) + 0.8 * np.sin(0.5 * freq2 * t)
        target = np.expand_dims(target, axis=1)
        return const_signal, target


def main(activation):
    os.makedirs('figures', exist_ok=True)
    time_length = 300

    eval_dataset = Torus(freq_range=3, time_length=300)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentNeuralNetwork(n_in=2, n_out=1, n_hid=300, device=device,
                                   activation=activation, sigma=0, use_bias=True).to(device)

    model_path = f'trained_model/torus_{activation}/epoch_2000.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    analyzer = FixedPoint(model=model, device=device, max_epochs=200000)

    hidden_list_list = np.zeros([5 * (time_length-100), model.n_hid])
    fixed_point_list = np.zeros([5, model.n_hid])
    i = 0
    while i < 5:
        signal_numpy, target = eval_dataset.getitem()
        signal = torch.from_numpy(np.array([signal_numpy]))

        hidden = torch.zeros(1, 300)
        hidden = hidden.to(device)
        signal = signal.float().to(device)
        with torch.no_grad():
            hidden_list, _, _ = model(signal, hidden)

        const_signal = signal[0, 80, :]
        const_signal = const_signal.float().to(device)

        fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, 200], const_signal, view=True)
        if not result_ok:
            continue

        hidden_list_list[i * (time_length-100):(i + 1) * (time_length-100), ...] = hidden_list.cpu().numpy()[:, 100:, :]
        fixed_point_list[i] = fixed_point.detach().cpu().numpy()
        i += 1

    pca = PCA(n_components=3)
    pca.fit(hidden_list_list)

    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=45, azim=134)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    print(hidden_list_list.shape)
    print(fixed_point_list.shape)
    pc_trajectory = pca.transform(hidden_list_list)
    pc_fixed_point = pca.transform(fixed_point_list)

    for i in range(5):
        ax.plot(pc_trajectory.T[0, i * time_length:(i + 1) * time_length],
                pc_trajectory.T[1, i * time_length:(i + 1) * time_length],
                pc_trajectory.T[2, i * time_length:(i + 1) * time_length], color='royalblue')
    ax.scatter(pc_fixed_point.T[0], pc_fixed_point.T[1], pc_fixed_point.T[2], color='red', marker='x')
    plt.title('trajectory')
    plt.savefig(f'figures/torus_trajectory_{activation}.png', dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
