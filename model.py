"""define recurrent neural networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 activation='relu', sigma=0.05, use_bias=True):
        super(RecurrentNeuralNetwork, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=use_bias)
        self.w_out = nn.Linear(n_hid, n_out, bias=False)

        self.activation = activation
        self.sigma = sigma
        self.device = device

    def forward(self, input_signal, hidden):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):

            pre_activates = self.w_in(input_signal[t]) + self.w_hh(hidden)

            if self.activation == 'relu':
                hidden = F.relu(pre_activates)
            else:
                hidden = torch.tanh(pre_activates)

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output
        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden
