"""fixed point analysis"""

import argparse
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

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

    freq = 17
    const_signal = np.repeat(freq / freq_range + 0.25, time_length)
    const_signal = np.expand_dims(const_signal, axis=1)
    const_signal_tensor = torch.from_numpy(np.array([const_signal]))

    analyzer = FixedPoint(model=model, device=device)

    hidden = torch.zeros(1, 200)
    hidden = hidden.to(device)
    const_signal_tensor = const_signal_tensor.float().to(device)
    with torch.no_grad():
        hidden_list, _, _ = model(const_signal_tensor, hidden)

    fixed_point, _ = analyzer.find_fixed_point(torch.unsqueeze(hidden_list[:, 20, :], dim=0).to(device),
                                               const_signal_tensor, view=True)

    # linear approximation around fixed point
    jacobian = analyzer.calc_jacobian(fixed_point, const_signal_tensor)

    # eigenvalue decomposition
    w, v = np.linalg.eig(jacobian)
    w_real = list()
    w_im = list()
    for eig in w:
        w_real.append(eig.real)
        w_im.append(eig.imag)
    plt.scatter(w_real, w_im)
    plt.xlabel(r'$Re(\lambda)$')
    plt.ylabel(r'$Im(\lambda)$')
    plt.savefig(f'figures/{activation}_eigenvalues.png', dpi=100)

    eig_freq = list()
    dynamics_freq = list()
    for i in range(20):
        freq = np.random.randint(1, freq_range + 1)
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

        jacobian = analyzer.calc_jacobian(fixed_point, const_signal_tensor)
        w, v = np.linalg.eig(jacobian)
        max_index = np.argmax(abs(w))
        eig_freq.append(abs(w[max_index].imag))
        dynamics_freq.append(freq)

    plt.figure()
    plt.scatter(eig_freq, dynamics_freq)
    plt.xlabel(r'$|Im(\lambda_{max})|$')
    plt.ylabel(r'$\omega$')
    plt.title('relationship of frequency')
    plt.savefig(f'figures/freq_{activation}.png', dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
