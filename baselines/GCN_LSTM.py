# -*- coding: utf-8 -*-
# @Time    : 07/07/2024 11:39
# @Author  : mmai
# @FileName: GCN_LSTM
# @Software: PyCharm


import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn.pytorch import GraphConv
class GCN_LSTMRegressor(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, lstm_hidden_size, num_layers, g):
        super(GCN_LSTMRegressor, self).__init__()
        # self.conv1 = GraphConv(in_feats, h_feats, norm='both', weight=True)
        # self.conv2 = GraphConv(h_feats, h_feats, norm='both', weight=True)
        self.conv1 = GraphConv(1, h_feats, norm='both', weight=True)
        self.conv2 = GraphConv(h_feats, h_feats, norm='both', weight=True)
        self.lstm = nn.LSTM(h_feats * g.num_nodes(), lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, out_feats * g.num_nodes())
        self.out_feats = out_feats
        sigma = g.edata['distance'].std()
        self.distance = torch.exp(-g.edata['distance'] ** 2 / (sigma ** 2))

    def forward(self, g, in_feat):
        # h = self.conv1(g, in_feat, edge_weight=self.distance)
        h1_lst = []
        for i in range(in_feat.shape[-1]):
            h = self.conv1(g, in_feat[..., i].unsqueeze(-1), edge_weight=self.distance)
            h = F.relu(h)
            h1_lst.append(h)
        # h = F.relu(h)
        h = torch.stack(h1_lst, dim=-1) # [num_nodes, batch_size, hidden_size, num_timesteps_input]
        # h = self.conv2(g, h, edge_weight=self.distance)
        # h = h.unsqueeze(0)  # Add batch dimension for LSTM
        h = h.permute(1, 3, 0, 2)  # [batch_size, num_timesteps_input, num_nodes, hidden_size]
        h = h.reshape(h.shape[0], h.shape[1], -1)  # [num_timesteps_input, batch_size, num_nodes*hidden_size]
        h, (hn, cn) = self.lstm(h)
        h = self.fc(h)  # [batch_size, num_timesteps_input, num_nodes*out_feats]
        h = h[:, -1, :].reshape(-1, g.num_nodes(), self.out_feats).permute(1, 0, 2)  # [num_nodes, batch_size, out_feats]
        return h

    def inference(self, g, in_feat):
        h = self.forward(g, in_feat)
        return h.clamp(min=0)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import numpy as np
    import dgl
    import random
    import pickle
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
    # dataset_name = "train_station"
    dataset_name = "maze"
    if dataset_name == "crossroad":
        train_sc = ['../sc_sensor/crossroad2']
        test_sc = ['../sc_sensor/crossroad1', '../sc_sensor/crossroad11', '../sc_sensor/crossroad13']
    elif dataset_name == "train_station":
        train_sc = ['../sc_sensor/train1']
        test_sc = ['../sc_sensor/train2']
    elif dataset_name == "maze":
        train_sc = ['sc_sensor/maze0']
        test_sc = ['sc_sensor/maze13', 'sc_sensor/maze4']

    # Loop through each subdirectory in the parent directory
    if dataset_name == "maze":
        with open("../sc_sensor/maze/flow_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
    else:
        df_dict = process_sensor_data(parent_dir, df_dict)
        data_dict = gen_data_dict(df_dict)
        data_dict = seperate_up_down(data_dict)

    pred_horizon = 7 # 3, 5
    lags = 5
    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=lags, horizons=pred_horizon, shuffle=True)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=5,
    #                                                                              horizons=pred_horizon,
    #                                                                              portion=0.03,
    #                                                                              shuffle=True)

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
    x_scalar = None

    #processing graph data
    # src = np.array([0, 2])
    # dst = np.array([3, 1])
    g_data = load_graphs('../graphs/graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]
    elif dataset_name == "maze":
        g = g_data[0][2]
    g = dgl.add_self_loop(g)
    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(x_train,
                                y_train, batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    # train
    # model = GCN(in_size=num_input_timesteps, hid_size=128, out_size=pred_horizon-1, scalar=x_scalar, g=g)  # out_size: prediction horizon
    model = GCN_LSTMRegressor(in_feats=num_input_timesteps, h_feats=32, out_feats=pred_horizon-1, lstm_hidden_size=32, num_layers=2, g=g)  # out_size: prediction horizon
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()
    # g = dgl.add_self_loop(g)
    # train
    for epoch in range(500):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model(g, g.ndata['feature']) # [num_dst, batch_size]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            # early stopping
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    if dataset_name == "crossroad":
        torch.save(model.state_dict(), f'../checkpoint/gcn/gcnlstm_crossroad_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "train_station":
        torch.save(model.state_dict(), f'../checkpoint/gcn/gcnlstm_trainstation_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "maze":
        torch.save(model.state_dict(), f'../checkpoint/gcn/gcnlstm_maze_lags{lags}_hor{pred_horizon}.pth')

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
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
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
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
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