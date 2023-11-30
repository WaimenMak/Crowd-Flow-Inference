# -*- coding: utf-8 -*-
# @Time    : 26/11/2023 15:43
# @Author  : mmai
# @FileName: Diffusion
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Velocity_Model(nn.Module):
    def __init__(self, num_timesteps_input, scalar):
        super(Velocity_Model, self).__init__()
        # MLP
        self.linear1 = torch.nn.Linear(2*num_timesteps_input, 64)
        # batch normalization
        # self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = torch.nn.Linear(64, 32)
        # self.bn2 = nn.BatchNorm1d(32)
        self.linear3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.x_scalar = scalar

    def forward(self, upstream, downstream):
        # upstream : [batch_size, num_timesteps_input], only from one node
        # downstream : [batch_size, num_timesteps_input]
        with torch.no_grad():
            upstream = self.x_scalar.transform(upstream)
            downstream = self.x_scalar.transform(downstream)
        x = torch.cat((upstream, downstream), dim=1)
        x = self.linear1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        out = self.linear3(x)
        # out = self.relu(out)
        return out

class Velocity_Model_NAM(nn.Module):
    def __init__(self, num_timesteps_input, scalar):
        super(Velocity_Model_NAM, self).__init__()
        # MLP
        self.linear11 = torch.nn.Linear(num_timesteps_input, 32)
        self.linear12 = torch.nn.Linear(32, 1)

        self.linear21 = torch.nn.Linear(num_timesteps_input, 32)
        self.linear22 = torch.nn.Linear(32, 1)

        self.linear3 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
        self.x_scalar = scalar

    def forward(self, upstream, downstream):
        # upstream : [batch_size, num_timesteps_input], only from one node
        # downstream : [batch_size, num_timesteps_input]
        with torch.no_grad():
            upstream = self.x_scalar.transform(upstream)
            downstream = self.x_scalar.transform(downstream)
        x_up = self.linear11(upstream)
        x_up = self.relu(x_up)
        x_up = self.linear12(x_up)
        x_up = torch.sigmoid(x_up)

        x_down = self.linear21(downstream)
        x_down = self.relu(x_down)
        x_down = self.linear22(x_down)
        x_down = torch.sigmoid(x_down)

        x = torch.cat((x_up, x_down), dim=1)
        out = self.linear3(x)
        # out = self.relu(out)
        return out
class Diffusion_Model(torch.nn.Module):
    def __init__(self, num_nodes, num_timesteps_input, scalar, time_units=10):
        """

        :param num_nodes: number of ancestor nodes
        :param num_timesteps_input: time steps of input
        """
        super(Diffusion_Model, self).__init__()
        # self.velocity_model = Velocity_Model(num_timesteps_input, scalar=scalar)
        self.velocity_model = Velocity_Model_NAM(num_timesteps_input, scalar=scalar)
        self.num_timesteps_input = num_timesteps_input
        self.num_nodes = num_nodes
        self.time_units = time_units
        # self.alpha = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)
        self.alpha = nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)
        self.device = device
        self.to(self.device)
        self.L = torch.FloatTensor([45]).to(device)  # 50m

    @staticmethod
    def diffusion_sequence(F, n):
        indices = torch.arange(n.item()-1, -1, -1)
        result = F * torch.pow(1 - F, indices)
        return result

    def init_velocity_model(self, x_train, x_val, model):
        # pretrain the model on a sythentic dataset with speed label
        # x_train : [batch_size, num_timesteps_input, num_nodes]
        # construct the label for x_train
        # y_train : [batch_size, 1]
        x_train = torch.FloatTensor(x_train)
        x_val = torch.FloatTensor(x_val)
        x = torch.cat((x_train, x_val), dim=0)
        upstream_flows = x[:, :, :-1].reshape(-1, self.num_timesteps_input)
        downstream_flows = x[:, :, -1].reshape(-1, self.num_timesteps_input, 1).repeat(1, 1, self.num_nodes).reshape(-1, self.num_timesteps_input)
        # velocity label coming from a gaussian distribution with mean 1.5 and std 0.01
        y = torch.normal(mean=1.5, std=0.1, size=[x.shape[0], 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = torch.nn.MSELoss()
        # upstream_flows = self.x_scalar.transform(upstream_flows)
        # downstream_flows = self.x_scalar.transform(downstream_flows)
        model.train()
        for iter in range(100):
            pred = model(upstream_flows, downstream_flows)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Pretrain Iteration: {}, Loss: {}'.format(iter, loss.item()))
    def diffusion_sample(self, upstream_flow, T, T_idx):
        # upstream flow : [batch_size, num_timesteps_input], flow from one node
        # T : [batch_size, 1], travel time
        # T_idx : [batch_size, 1], travel time index

        #diffustion coefficient
        F = 1/(1 + self.alpha*T) # [batch_size, 1]
        # F = 1/(1 + T) # [batch_size, 1]
        total_time_steps = upstream_flow.shape[1]
        # construct the diffusion matrix according to T_idx
        sequences = []

        for i in range(F.size(0)):
            F_sequence = self.diffusion_sequence(F[i], torch.max((total_time_steps - T_idx[i]), torch.FloatTensor([1])))
            sequences.append(F_sequence)

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0) # [batch_size, num_timesteps_input]
        if padded_sequences.shape[1] < upstream_flow.shape[1]:
            pad_zero = torch.zeros(upstream_flow.shape[0], upstream_flow.shape[1] - padded_sequences.shape[1]).to(self.device)
            padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)

        prediction = padded_sequences * upstream_flow # [batch_size, num_timesteps_input]

        prediction = torch.sum(prediction, dim=1, keepdim=True) # [batch_size]

        return prediction

    def diffusion_batch(self, upstream_flow, T, T_idx):
        pass

    def velocity(self, upstream_flow, downstream_flow):
        # upstream flow : [batch_size, num_timesteps_input], flow from one node
        # downstream flow : [batch_size, num_timesteps_input], flow from one node
        with torch.no_grad():
            v = self.velocity_model(upstream_flow, downstream_flow)

        return v

    def forward(self, x):
        # upstream flows : [batch_size, num_timesteps_input, num_nodes]
        # downstream flows : [batch_size, num_timesteps_input, num_nodes] # duplicate of the downstream flows
        upstream_flows = x[:, :, :-1].reshape(-1, num_input_timesteps, num_nodes)
        downstream_flows = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
        pred = []
        for n in range(self.num_nodes):
            v = self.velocity_model(upstream_flows[:, :, n],
                                    downstream_flows[:, :, n]) # v : [batch_size, 1]
            # v = torch.FloatTensor([1.5]).repeat(upstream_flows.shape[0], 1).to(self.device)
            T = torch.divide(self.L, v) + 1e-5 # T : [batch_size, 1]
            #round T to the time interval
            # with torch.no_grad():
            T_idx = torch.round(T/self.time_units)
            # < 0  = 0, > num_timesteps_input = num_timesteps_input-1
            T_idx = T_idx.masked_fill(T_idx < 0, 0)
            T_idx = T_idx.masked_fill(T_idx > self.num_timesteps_input, self.num_timesteps_input-1)
            # T_idx = 0 * torch.ones_like(T)
            # diffusion
            oupt = self.diffusion_sample(upstream_flows[:, :, n], T, T_idx)
            pred.append(oupt)

        # TODO: sum the predictions from all the nodes with transition probability
        pred = torch.sum(torch.cat(pred, dim=1), dim=1, keepdim=True) # [batch_size, num_nodes] --> [batch_size, 1]

        return pred

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, generate_insample_dataset, StandardScaler

    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = './sc sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)
    up = data_dict['./sc sensor/crossroad1'][:,0,0].reshape(-1,1) # shape (ts, num_nodes)
    down = data_dict['./sc sensor/crossroad1'][:,1,1].reshape(-1,1) # shape (ts, 1)

    # x_train : [batch_size, num_timesteps_input, num_nodes], x_train contain the downstream flow at last row.
    x_train, y_train, x_val, y_val, x_test, y_test = generate_insample_dataset(up, down)
    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] - 1 # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(np.concatenate([x_train, x_val], axis=0),
                                np.concatenate([y_train, y_val], axis=0), batch_size=7)
    train_dataloader = DataLoader(train_dataset, batch_size=7)
    # set seed
    torch.manual_seed(10)
    #normalization
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())
    # train
    model = Diffusion_Model(num_nodes=1, num_timesteps_input=x_train.shape[1], scalar=x_scalar)
    model.init_velocity_model(x_train, x_val, model.velocity_model)
    model.velocity_model.requires_grad_(True)
    model.alpha.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    # x, y = next(iter(train_dataloader))
    x = torch.FloatTensor(x_train[2].reshape(1, num_input_timesteps, 2))
    y = torch.FloatTensor(y_train[2].reshape(1, 8, 1))
    for epoch in range(15000):
        l = []
        # for i, (x, y) in enumerate(train_dataloader):
            # training loop x: [batch_size, num_timesteps_input, num_nodes]
            # if epoch % 50 == 0:
            #     model.alpha.requires_grad_(True)
            #     model.velocity_model.requires_grad_(False)
            # else:
            #     model.alpha.requires_grad_(False)
            #     model.velocity_model.requires_grad_(True)
        pred = model(x)
        loss = loss_fn(pred, y[:, 0, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l.append(loss.item())
            # early stopping
        if epoch % 500 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))


    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    print('*************')

with torch.no_grad():
    for i, (x, y) in enumerate(train_dataloader):
        x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
        x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
        pred = model(x)
        loss = loss_fn(pred, y[:, 0, :])
        v = model.velocity_model(x_up[...,0], x_down[...,0])
        print('Train Prediction: {}'.format(pred))
        print('Train Ground Truth: {}'.format(y[:, 0, :]))
        print('Train Loss: {}'.format(loss.item()))
        print('Train Velocity: {}'.format(v))
        print('*************')

    for i, (x, y) in enumerate(test_dataloader):
        x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
        x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
        pred = model(x)
        loss = loss_fn(pred, y[:, 0, :])
        v = model.velocity_model(x_up[...,0], x_down[...,0])
        print('Test Prediction: {}'.format(pred))
        print('Test Ground Truth: {}'.format(y[:, 0, :]))
        print('Test Loss: {}'.format(loss.item()))
        print('Test Velocity: {}'.format(v))
        print('*************')

# print model parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
