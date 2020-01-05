"""fixed point analysis"""

import argparse
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

from analyzer import FixedPoint
from model import RecurrentNeuralNetwork


class Torus(object):
    def __init__(self, time_length=50, freq_range=3):
        self.time_length = time_length
        self.freq_range = freq_range

    def __len__(self):
        return 200

    def getitem(self, freq1):
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
    freq_range = 3
    time_length = 300

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentNeuralNetwork(n_in=2, n_out=1, n_hid=300, device=device,
                                   activation=activation, sigma=0, use_bias=True).to(device)

    model_path = f'trained_model/torus_{activation}/epoch_2000.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    eval_dataset = Torus(freq_range=3, time_length=300)

    analyzer = FixedPoint(model=model, device=device, max_epochs=800000)

    freq1 = 1
    signal_numpy, target = eval_dataset.getitem(freq1)
    signal = torch.from_numpy(np.array([signal_numpy]))

    hidden = torch.zeros(1, 300)
    hidden = hidden.to(device)
    signal = signal.float().to(device)
    with torch.no_grad():
        hidden_list, _, _ = model(signal, hidden)

    const_signal = signal[0, 80, :]
    const_signal = const_signal.float().to(device)

    fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, 200], const_signal, view=True)

    # linear approximation around fixed point
    jacobian = analyzer.calc_jacobian(fixed_point, const_signal)

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
    plt.savefig(f'figures/torus_{activation}_freq_{freq1}_eigenvalues.png', dpi=100)

    eig_freq = list()
    dynamics_freq = list()
    for i in range(3):
        freq1 = i + 1
        signal_numpy, target = eval_dataset.getitem(freq1)
        signal = torch.from_numpy(np.array([signal_numpy]))

        hidden = torch.zeros(1, 300)
        hidden = hidden.to(device)
        signal = signal.float().to(device)
        with torch.no_grad():
            hidden_list, _, _ = model(signal, hidden)

        const_signal = signal[0, 80, :]
        const_signal = const_signal.float().to(device)

        fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, 200], const_signal, view=False)

        jacobian = analyzer.calc_jacobian(fixed_point, const_signal)
        w, v = np.linalg.eig(jacobian)
        max_index = np.argmax(abs(w))
        eig_freq.append(abs(w[max_index].imag))
        dynamics_freq.append(freq1)

    plt.figure()
    plt.scatter(eig_freq, dynamics_freq)
    plt.xlabel(r'$|Im(\lambda_{max})|$')
    plt.ylabel(r'$\omega$')
    plt.title('relationship of frequency')
    plt.savefig(f'figures/torus_freq_{activation}.png', dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
