"""training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable

from dataset import FlipFlop
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # save path
    os.makedirs('trained_model', exist_ok=True)
    save_path = f'trained_model/{cfg["MODEL"]["NAME"]}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    model = RecurrentNeuralNetwork(n_in=2, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   activation=cfg['MODEL']['ACTIVATION'], sigma=cfg['MODEL']['SIGMA'],
                                   use_bias=cfg['MODEL']['USE_BIAS']).to(device)

    train_dataset = FlipFlop(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                             u_fast_mean=cfg['DATALOADER']['FASE_MEAN'],
                             u_slow_mean=cfg['DATALOADER']['SLOW_MEAN_1'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])

    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        if epoch == 1000:
            train_dataset = FlipFlop(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                     u_fast_mean=cfg['DATALOADER']['FASE_MEAN'],
                                     u_slow_mean=cfg['DATALOADER']['SLOW_MEAN_2'])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                           num_workers=2, shuffle=True,
                                                           worker_init_fn=lambda x: np.random.seed())
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target, = data
            inputs, target, = inputs.float(), target.float()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)
            # print(inputs.shape)

            hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            hidden_list, output, hidden = model(inputs, hidden)

            loss = torch.nn.MSELoss()(output[:, 10:], target[:, 10:])
            loss.backward()
            optimizer.step()

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            print(f'Train Epoch: {epoch}, Loss: {loss.item():.6f}')
            print('output', output[0, 10:, 0].cpu().detach().numpy())
            print('target', target[0, 10:, 0].cpu().detach().numpy())
            torch.save(model.state_dict(), os.path.join(save_path, f'{cfg["MODEL"]["NAME"]}_epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
