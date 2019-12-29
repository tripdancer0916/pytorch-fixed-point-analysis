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


def main(activation):
    os.makedirs('figures', exist_ok=True)
    freq_range = 51
    time_length = 40

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentNeuralNetwork(n_in=1, n_out=1, n_hid=200, device=device,
                                   activation=activation, sigma=0, use_bias=True).to(device)

    model_path = f'trained_model/{activation}/epoch_1000.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    analyzer = FixedPoint(model=model, device=device, max_epochs=200000)

    hidden_list_list = np.zeros([30 * time_length, model.n_hid])
    fixed_point_list = np.zeros([15, model.n_hid])
    i = 0
    while i < 15:
        freq = np.random.randint(10, freq_range + 1)
        const_signal = np.repeat(freq / freq_range + 0.25, time_length)
        const_signal = np.expand_dims(const_signal, axis=1)
        const_signal_tensor = torch.from_numpy(np.array([const_signal]))

        hidden = torch.zeros(1, 200)
        hidden = hidden.to(device)
        const_signal_tensor = const_signal_tensor.float().to(device)
        with torch.no_grad():
            hidden_list, _, _ = model(const_signal_tensor, hidden)

        fixed_point, result_ok = analyzer.find_fixed_point(torch.unsqueeze(hidden_list[:, 20, :], dim=0).to(device),
                                                           const_signal_tensor)
        if not result_ok:
            continue

        hidden_list_list[i * time_length:(i + 1) * time_length, ...] = hidden_list.cpu().numpy()[:, ...]
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

    for i in range(15):
        ax.plot(pc_trajectory.T[0, i * time_length:(i + 1) * time_length],
                pc_trajectory.T[1, i * time_length:(i + 1) * time_length],
                pc_trajectory.T[2, i * time_length:(i + 1) * time_length], color='royalblue')
    ax.scatter(pc_fixed_point.T[0], pc_fixed_point.T[1], pc_fixed_point.T[2], color='red', marker='x')
    plt.title('trajectory')
    plt.savefig(f'figures/trajectory_{activation}.png', dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
