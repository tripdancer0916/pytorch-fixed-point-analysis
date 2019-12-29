"""compare fixed points"""

import argparse
import os

import numpy as np
import torch

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

    analyzer = FixedPoint(model=model, device=device)

    # compare fixed points
    freq = 17
    const_signal1 = np.repeat(freq / freq_range + 0.25, time_length)
    const_signal1 = np.expand_dims(const_signal1, axis=1)
    const_signal_tensor1 = torch.from_numpy(np.array([const_signal1]))

    hidden = torch.zeros(1, 200)
    hidden = hidden.to(device)
    const_signal_tensor1 = const_signal_tensor1.float().to(device)
    with torch.no_grad():
        hidden_list, _, _ = model(const_signal_tensor1, hidden)

    # different time of same trajectory.
    fixed_point1, _ = analyzer.find_fixed_point(torch.unsqueeze(hidden_list[:, 20, :], dim=0).to(device),
                                                const_signal_tensor1, view=True)
    fixed_point2, _ = analyzer.find_fixed_point(torch.unsqueeze(hidden_list[:, 15, :], dim=0).to(device),
                                                const_signal_tensor1)

    print('distance between 2 fixed point start from different IC; different time of same trajectory.')
    print(torch.norm(fixed_point1 - fixed_point2).item())

    # same time of different trajectories.
    freq = 18
    const_signal2 = np.repeat(freq / freq_range + 0.25, time_length)
    const_signal2 = np.expand_dims(const_signal2, axis=1)
    const_signal_tensor2 = torch.from_numpy(np.array([const_signal2]))

    hidden = torch.zeros(1, 200)
    hidden = hidden.to(device)
    const_signal_tensor2 = const_signal_tensor2.float().to(device)
    with torch.no_grad():
        hidden_list, _, _ = model(const_signal_tensor2, hidden)

    fixed_point3, _ = analyzer.find_fixed_point(torch.unsqueeze(hidden_list[:, 20, :], dim=0).to(device),
                                                const_signal_tensor2)
    print('distance between 2 fixed point start from different IC; same time of different trajectories.')
    print(torch.norm(fixed_point1 - fixed_point3).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
