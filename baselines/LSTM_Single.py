# -*- coding: utf-8 -*-
# @Time    : 28/02/2024 05:49
# @Author  : mmai
# @FileName: LSTM_Single
# @Software: PyCharm

import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_nodes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * num_nodes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use the output from the last time step
        return output

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

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = '../sc_sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)

    # dataset_name = "crossroad"
    dataset_name = "train_station"
    # train_sc = ['../sc_sensor/crossroad3', '../sc_sensor/crossroad8', '../sc_sensor/crossroad2', '../sc_sensor/crossroad5']
    # test_sc = ['../sc_sensor/crossroad1', '../sc_sensor/crossroad11', '../sc_sensor/crossroad13']
    train_sc = ['../sc_sensor/train6', '../sc_sensor/train7', '../sc_sensor/train2']
    test_sc = ['../sc_sensor/train1']

    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)
    pred_horizon = 5 # 3, 5
    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=5, horizons=pred_horizon, shuffle=True)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=5,
    #                                                                              horizons=pred_horizon,
    #                                                                              portion=0.03,
    #                                                                              shuffle=True)

    g_data = load_graphs('../graphs/graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]

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
    # model = SimpleLSTM(input_size=num_nodes, hidden_size=64, output_size=pred_horizon-1, num_layers=2, num_nodes=num_nodes)  # out_size: prediction horizon
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()
    g = dgl.add_self_loop(g)

    models = [SimpleLSTM(input_size=1, hidden_size=64, output_size=pred_horizon-1, num_layers=2, num_nodes=1) for i in range(len(dst_idx))]  # out_size: prediction horizon
    optimizers = [torch.optim.Adam(models[i].parameters(), lr=0.001, weight_decay=1e-5) for i in range(len(dst_idx))]
    loss_fn = torch.nn.MSELoss()


    for epoch in range(100):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            # x = x.permute(1, 2, 0)
            # y = y.permute(1, 2, 0).reshape(y.shape[1], -1)
            # y = y.reshape(y.shape[0], -1)
            pred_list = []
            for dst_id, model, optimizer in zip(dst_idx, models, optimizers):
                pred = model(x[..., dst_id].reshape(-1, num_input_timesteps, 1))
                loss = loss_fn(pred, y[..., dst_id])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_list.append(pred)

            loss = loss_fn(torch.stack(pred_list, dim=-1), y[..., dst_idx])
            l.append(loss.item())

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    # if dataset_name == "crossroad":
    #     torch.save(model.state_dict(), '../checkpoint/lstm/lstm_crossroad.pth')
    # if dataset_name == "train_station":
    #     torch.save(model.state_dict(), '../checkpoint/lstm/lstm_trainstation.pth')
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
            # y = y.reshape(y.shape[0], -1)
            pred_list = []
            for dst_id, model, optimizer in zip(dst_idx, models, optimizers):
                pred = model(x[..., dst_id].reshape(-1, num_input_timesteps, 1))
                pred_list.append(pred)
            # pred = model(x)
            # pred = pred.reshape(pred.shape[0], pred_horizon-1, num_nodes).permute(2, 0, 1)
            loss = loss_fn(torch.stack(pred_list, dim=-1).permute(2, 0, 1)[..., 0], g.ndata['label'][dst_idx,:, 0])
            # loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            train_loss.append(loss.item())

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(torch.stack(pred_list, dim=-1).permute(2, 0, 1), g.ndata['label'][dst_idx, :, :])
            multi_steps_train_loss.append(multisteps_loss.item())


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
            pred_list = []
            for dst_id, model, optimizer in zip(dst_idx, models, optimizers):
                pred = model(x[..., dst_id].reshape(-1, num_input_timesteps, 1))
                pred_list.append(pred)

            loss = loss_fn(torch.stack(pred_list, dim=-1).permute(2, 0, 1)[..., 0], g.ndata['label'][dst_idx,:, 0])
            # loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            train_loss.append(loss.item())

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(torch.stack(pred_list, dim=-1).permute(2, 0, 1), g.ndata['label'][dst_idx, :, :])
            multi_steps_test_loss.append(multisteps_loss.item())


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
