"""search fixed points!!!"""

import argparse
import os

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use('Agg')

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
        t = np.arange(0, self.time_length * 0.3, 0.3)
        target = 0.6 * np.sin(2.2 * freq1 * t) + 0.8 * np.sin(0.5 * freq2 * t)
        target = np.expand_dims(target, axis=1)
        return const_signal, target


class FlipFlop(object):
    def __init__(self, time_length, u_fast_mean=4, u_slow_mean=16):
        self.time_length = time_length
        self.u_fast_mean = u_fast_mean
        self.u_slow_mean = u_slow_mean

    def __len__(self):
        return 200

    def getitem(self):
        # input signal
        u_fast_signal = np.zeros(self.time_length)
        u_slow_signal = np.zeros(self.time_length)
        fast_signal_timing = np.random.poisson(self.u_fast_mean, 100)
        slow_signal_timing = np.random.poisson(self.u_slow_mean, 100)
        slow_signal_timing[0] = 10
        u_fast_signal[0] = np.random.choice([-1, 1])
        u_slow_signal[0] = np.random.choice([-1, 1])
        count = 0
        index = 0
        while index < self.time_length:
            index += max(0, fast_signal_timing[count])
            count += 1
            if index < self.time_length:
                u_fast_signal[index] = np.random.choice([-1, 1])

        count = 0
        index = 0
        while index < self.time_length:
            index += max(0, slow_signal_timing[count])
            count += 1
            if index < self.time_length:
                u_slow_signal[index] = np.random.choice([-1, 1])

        # target
        fast_signal_record = np.zeros(self.time_length)
        slow_signal_record = np.zeros(self.time_length)
        fast_signal_record[0] = u_fast_signal[0]
        temporal_memory = u_slow_signal[0]
        for index in range(1, self.time_length):
            if u_fast_signal[index] == 0:
                fast_signal_record[index] = fast_signal_record[index - 1]
            else:
                fast_signal_record[index] = u_fast_signal[index]

            if u_slow_signal[index] == 0:
                slow_signal_record[index] = slow_signal_record[index - 1]
            else:
                slow_signal_record[index] = temporal_memory
                temporal_memory = u_slow_signal[index]

        target_signal = np.zeros(self.time_length)
        for index in range(self.time_length):
            target_signal[index] = slow_signal_record[index] * fast_signal_record[index]

        fast_signal = np.expand_dims(u_fast_signal, axis=1)
        slow_signal = np.expand_dims(u_slow_signal, axis=1)
        input_signal = np.concatenate((fast_signal, slow_signal), axis=1)
        target_signal = np.expand_dims(target_signal, axis=1)

        return input_signal, target_signal


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # save path
    os.makedirs('results', exist_ok=True)
    save_path = f'results/{cfg["MODEL"]["NAME"]}'
    os.makedirs(save_path, exist_ok=True)

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    eval_dataset = FlipFlop(time_length=100,
                            u_fast_mean=cfg['DATALOADER']['FAST_MEAN'],
                            u_slow_mean=cfg['DATALOADER']['SLOW_MEAN_1'])

    cfg['MODEL']['SIGMA'] = 0.0
    model = RecurrentNeuralNetwork(n_in=2, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   activation=cfg['MODEL']['ACTIVATION'], sigma=cfg['MODEL']['SIGMA'],
                                   use_bias=cfg['MODEL']['USE_BIAS']).to(device)

    model_path = f'trained_model/{cfg["MODEL"]["NAME"]}/{cfg["MODEL"]["NAME"]}_epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    analyzer = FixedPoint(model=model, device=device, max_epochs=230000)

    for trial in range(100):
        signal_numpy, target = eval_dataset.getitem()
        signal = torch.from_numpy(np.array([signal_numpy]))

        hidden = torch.zeros(1, cfg['MODEL']['SIZE'])
        hidden = hidden.to(device)
        signal = signal.float().to(device)
        with torch.no_grad():
            hidden_list, _, _ = model(signal, hidden)

        const_signal = torch.tensor([0, 0])
        const_signal = const_signal.float().to(device)

        fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, 50], const_signal, view=True)

        # dynamics = hidden_list.cpu().numpy()[0, 50:, :]
        fixed_point = fixed_point.detach().cpu().numpy()

        print(fixed_point)

        np.savetxt(os.path.join(save_path, f'fixed_point_{trial}.txt'), fixed_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
