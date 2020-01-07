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


class ThreeBitFlipFlopMixed(object):
    def __init__(self, time_length, u1_mean=10, u2_mean=10, u3_mean=10):
        self.time_length = time_length
        self.u1_mean = u1_mean
        self.u2_mean = u2_mean
        self.u3_mean = u3_mean

    def __len__(self):
        return 200

    def getitem(self):
        # input signal
        u1_signal = np.zeros(self.time_length)
        u2_signal = np.zeros(self.time_length)
        u3_signal = np.zeros(self.time_length)
        u1_signal_timing = np.random.poisson(self.u1_mean, self.time_length)
        u2_signal_timing = np.random.poisson(self.u2_mean, self.time_length)
        u3_signal_timing = np.random.poisson(self.u3_mean, self.time_length)

        u1_signal[0] = np.random.choice([-1, 1])
        u2_signal[0] = np.random.choice([-1, 1])
        u3_signal[0] = np.random.choice([-1, 1])
        count = 0
        index = 0
        while index < self.time_length:
            index += max(0, u1_signal_timing[count])
            count += 1
            if index < self.time_length:
                u1_signal[index] = np.random.choice([-1, 1])
        count = 0
        index = 0
        while index < self.time_length:
            index += max(0, u2_signal_timing[count])
            count += 1
            if index < self.time_length:
                u2_signal[index] = np.random.choice([-1, 1])
        count = 0
        index = 0
        while index < self.time_length:
            index += max(0, u3_signal_timing[count])
            count += 1
            if index < self.time_length:
                u3_signal[index] = np.random.choice([-1, 1])

        # target
        u1_signal_record = np.zeros(self.time_length)
        u2_signal_record = np.zeros(self.time_length)
        u3_signal_record = np.zeros(self.time_length)
        u1_signal_record[0] = u1_signal[0]
        u2_signal_record[0] = u2_signal[0]
        u3_signal_record[0] = u3_signal[0]
        for index in range(1, self.time_length):
            if u1_signal[index] == 0:
                u1_signal_record[index] = u1_signal_record[index - 1]
            else:
                u1_signal_record[index] = u1_signal[index]
            if u2_signal[index] == 0:
                u2_signal_record[index] = u2_signal_record[index - 1]
            else:
                u2_signal_record[index] = u2_signal[index]
            if u3_signal[index] == 0:
                u3_signal_record[index] = u3_signal_record[index - 1]
            else:
                u3_signal_record[index] = u3_signal[index]

        z1_target = np.zeros(self.time_length)
        z2_target = np.zeros(self.time_length)
        z3_target = np.zeros(self.time_length)
        for index in range(self.time_length):
            z1_target[index] = u1_signal_record[index] * u2_signal_record[index]
            z2_target[index] = u2_signal_record[index] * u3_signal_record[index]
            z3_target[index] = u1_signal_record[index] * u2_signal_record[index] * u3_signal_record[index]

        u1_signal = np.expand_dims(u1_signal, axis=1)
        u2_signal = np.expand_dims(u2_signal, axis=1)
        u3_signal = np.expand_dims(u3_signal, axis=1)
        input_signal = np.concatenate((u1_signal, u2_signal), axis=1)
        input_signal = np.concatenate((input_signal, u3_signal), axis=1)
        z1_target = np.expand_dims(z1_target, axis=1)
        z2_target = np.expand_dims(z2_target, axis=1)
        z3_target = np.expand_dims(z3_target, axis=1)
        target_signal = np.concatenate((z1_target, z2_target, z3_target), axis=1)

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

    eval_dataset = ThreeBitFlipFlopMixed(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                         u1_mean=10, u2_mean=10, u3_mean=10)

    cfg['MODEL']['SIGMA'] = 0.0
    alpha = np.ones(cfg['MODEL']['SIZE'])
    model = RecurrentNeuralNetwork(n_in=3, n_out=3, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_weight=alpha,
                                   activation=cfg['MODEL']['ACTIVATION'], sigma=cfg['MODEL']['SIGMA'],
                                   use_bias=cfg['MODEL']['USE_BIAS']).to(device)

    model_path = f'trained_model/{cfg["MODEL"]["NAME"]}/{cfg["MODEL"]["NAME"]}_epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    analyzer = FixedPoint(model=model, device=device, max_epochs=140000)

    for trial in range(100):
        signal_numpy, target = eval_dataset.getitem()
        signal = torch.from_numpy(np.array([signal_numpy]))

        hidden = torch.zeros(1, cfg['MODEL']['SIZE'])
        hidden = hidden.to(device)
        signal = signal.float().to(device)
        with torch.no_grad():
            hidden_list, _, _ = model(signal, hidden)

        const_signal = torch.tensor([0, 0, 0])
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
