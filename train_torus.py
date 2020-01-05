"""training models"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.append('../')

from torch.autograd import Variable

from dataset import Torus
from model import RecurrentNeuralNetwork


def main(activation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    os.makedirs('trained_model', exist_ok=True)
    save_path = f'trained_model/torus_{activation}'
    os.makedirs(save_path, exist_ok=True)

    sigma = 0.1

    model = RecurrentNeuralNetwork(n_in=2, n_out=1, n_hid=300, device=device,
                                   activation=activation, sigma=sigma, use_bias=True).to(device)

    train_dataset = Torus(freq_range=3, time_length=60)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50,
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, weight_decay=0.0001)

    for epoch in range(2001):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target, = data
            inputs, target, = inputs.float(), target.float()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)
            # print(inputs.shape)

            hidden = torch.zeros(50, 300)
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            hidden_list, output, hidden = model(inputs, hidden)

            loss = torch.nn.MSELoss()(output[:, 5:], target[:, 5:])
            loss.backward()
            optimizer.step()

        if epoch > 0 and epoch % 100 == 0:
            print(f'Train Epoch: {epoch}, Loss: {loss.item():.6f}')
            print('output', output[0, :, 0].cpu().detach().numpy())
            print('target', target[0, :, 0].cpu().detach().numpy())
            torch.save(model.state_dict(), os.path.join(save_path, f'sigma_{sigma}_epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('--activation', type=str, default='tanh')
    args = parser.parse_args()
    # print(args)
    main(args.activation)
