# -*- coding: utf-8 -*-
# @Time    : 02/04/2025 16:01
# @Author  : mmai
# @FileName: LSTM_UQ
# @Software: PyCharm

import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from src.Diffusion_Network4_UQ import NegativeBinomialDistributionLoss

class SimpleLSTM_UQ(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_nodes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * num_nodes)
        self.sigma_fc = nn.ModuleList([nn.Linear(hidden_size, 32), nn.LayerNorm(32),
                                       nn.ReLU(), nn.Linear(32, num_nodes)]) # for the parameter of the negative binomial distribution, for each node

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the output from the last time step
        inpt = lstm_out[:, -1, :]
        for layer in self.sigma_fc:
            inpt = layer(inpt)
        sigma = torch.log(1 + torch.exp(inpt.squeeze())) # [batch_size, num_nodes]
        return output, sigma

if __name__ == '__main__':
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    import dgl
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import numpy as np
    import random
    import time
    import torch
    from dgl.data.utils import load_graphs
    import pickle

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = '../sc_sensor'

    # out of distribution
    # dataset_name = "crossroad"
    dataset_name = "train_station"
    # dataset_name = "maze"
    # dataset_name = "edinburgh"
    if dataset_name == "crossroad":
        train_sc = ['../sc_sensor/crossroad2']
        test_sc = ['../sc_sensor/crossroad1', '../sc_sensor/crossroad11', '../sc_sensor/crossroad13']
    elif dataset_name == "train_station":
        train_sc = ['../sc_sensor/train1']
        test_sc = ['../sc_sensor/train2']
    elif dataset_name == "maze":
        train_sc = ['sc_sensor/maze19']
        test_sc = ['sc_sensor/maze13', 'sc_sensor/maze4']
    elif dataset_name == "edinburgh":
        train_sc = ['26Aug']
        test_sc = ['27Aug']

    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
        # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    if dataset_name == "maze":
        with open("../sc_sensor/maze/flow_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
    elif dataset_name == "edinburgh":
        with open("../sc_sensor/edinburgh/flow_data_edinburgh.pkl", "rb") as f:
            data_dict = pickle.load(f)
    else:
        df_dict = process_sensor_data(parent_dir, df_dict)
        data_dict = gen_data_dict(df_dict)
        data_dict = seperate_up_down(data_dict)

    pred_horizon = 7 # 3, 5
    lags = 5
    if dataset_name == "edinburgh":
        pred_horizon = 2
        lags = 6

    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=lags, horizons=pred_horizon, shuffle=True)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=lags,
    #                                                                              horizons=pred_horizon,
    #                                                                              portion=0.7,
    #                                                                              shuffle=True)

    g_data = load_graphs('../graphs/4graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]
    elif dataset_name == "maze":
        g = g_data[0][2]
    elif dataset_name == "edinburgh":
        g = g_data[0][3]

    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(x_train,
                                y_train, batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # set seed
    torch.manual_seed(1)
    #normalization
    # x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
    #                           std=np.concatenate([x_train, x_val]).std())

    # train
    model = SimpleLSTM_UQ(input_size=num_nodes, hidden_size=64, output_size=pred_horizon-1, num_layers=2, num_nodes=num_nodes)  # out_size: prediction horizon
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    nll_loss_fn = NegativeBinomialDistributionLoss()

    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()
    g = dgl.add_self_loop(g)

    # for epoch in range(1500):
    for epoch in range(300):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            # x = x.permute(1, 2, 0)
            # y = y.permute(1, 2, 0).reshape(y.shape[1], -1)
            first_step = y[:, 0, :]
            y = y.reshape(y.shape[0], -1)
            pred, sigma = model(x)
            pred_first_step = pred.reshape(pred.shape[0], pred_horizon-1, num_nodes)[:, 0, :] # [batch_size, num_nodes]
            mse = loss_fn(pred, y)
            nll_loss = nll_loss_fn(pred_first_step[:, dst_idx].clamp(min=0), first_step[:, dst_idx], sigma[:, dst_idx]) # input size should be [num_nodes, batch_size] or [batch_size, num_nodes]
            # nll_loss = nll_loss_fn(pred_first_step.clamp(min=0), first_step, sigma)
            loss = mse + nll_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    if dataset_name == "crossroad":
        torch.save(model.state_dict(), f'../checkpoint/lstm/lstm_uq_crossroad_lags{lags}_hor{pred_horizon}.pth')
    elif dataset_name == "train_station":
        torch.save(model.state_dict(), f'../checkpoint/lstm/lstm_uq_trainstation_lags{lags}_hor{pred_horizon}.pth')
    elif dataset_name == "maze":
        torch.save(model.state_dict(), f'../checkpoint/lstm/lstm_uq_maze_lags{lags}_hor{pred_horizon}.pth')
    elif dataset_name == "edinburgh":
        torch.save(model.state_dict(), f'../checkpoint/lstm/lstm_uq_edinburgh_lags{lags}_hor{pred_horizon}.pth')
    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=y_test.shape[0])
    test_dataloader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    print('*************')

    test_loss = []
    train_loss = []
    multi_steps_train_loss = []
    multi_steps_test_loss = []
    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            # g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            # pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
            # x = x.permute(1, 2, 0)
            y = y.reshape(y.shape[0], -1)
            pred, sigma = model(x)
            pred = pred.reshape(pred.shape[0], pred_horizon-1, num_nodes).permute(2, 0, 1)
            loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            train_loss.append(loss.item())

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_train_loss.append(multisteps_loss.item())

            print('Train Prediction: {}'.format(pred[dst_idx,:, 0]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            print('Train Multi-Steps Loss: {}'.format(multisteps_loss.item()))

            # print("Probabilities: {}".format(model.g.ndata['alpha'][[0,3],...]))

            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            # g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            # pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
            # x = x.permute(1, 2, 0)
            y = y.reshape(y.shape[0], -1)
            pred, sigma = model(x)
            pred = pred.reshape(pred.shape[0], pred_horizon-1, num_nodes).permute(2, 0, 1)
            loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            test_loss.append(loss.item())


            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_test_loss.append(multisteps_loss.item())

            print('Test Prediction: {}'.format(pred[dst_idx, :, 0]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            print('Test Multi-Steps Loss: {}'.format(multisteps_loss.item()))

            print('*************')

        # print('Training Time: {}'.format(total_time))
        print('Total Train Loss: {}'.format(np.mean(train_loss)))
        print('Multi-Steps Train Loss: {}'.format(np.mean(multi_steps_train_loss)))
        print('Total Test Loss: {}'.format(np.mean(test_loss)))
        print('Multi-Steps Test Loss: {}'.format(np.mean(multi_steps_test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model)))  # 1025
