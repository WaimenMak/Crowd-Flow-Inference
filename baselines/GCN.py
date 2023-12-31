# -*- coding: utf-8 -*-
# @Time    : 09/12/2023 11:30
# @Author  : mmai
# @FileName: GCN
# @Software: PyCharm


import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, scalar):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
        self.x_scalar = scalar

    def forward(self, g, features):
        # with torch.no_grad():
        #     features = self.x_scalar.transform(features)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=g.edata['distance'])
        return h

    def inference(self, g, features):
        # with torch.no_grad():
        #     features = self.x_scalar.transform(features)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=g.edata['distance'])
        return h.clamp(min=0)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import numpy as np
    import dgl
    import random

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = '../sc_sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)
    # in distributino
    # x_train, y_train, x_val, y_val, x_test, y_test = generate_insample_dataset_ver2(data_dict)


    # out of distribution
    train_sc = ['../sc_sensor/crossroad4', '../sc_sensor/crossroad2', '../sc_sensor/crossroad3', '../sc_sensor/crossroad1']
    test_sc = ['../sc_sensor/crossroad5']
    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)

    # data_dict['../sc_sensor/crossroad4'] = data_dict['../sc_sensor/crossroad4'][..., :4]
    # for k in data_dict.keys():  # debug
    #     data_dict[k] = data_dict[k][:,[0,3]]

    # x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=5)
    x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
                                                                                 lags=5,
                                                                                 shuffle=False)

    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(np.concatenate([x_train, x_val], axis=0),
                                np.concatenate([y_train, y_val], axis=0), batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # set seed
    torch.manual_seed(1)
    #normalization
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())

    #processing graph data
    # src = np.array([0, 2])
    # dst = np.array([3, 1])

    src = np.array([0, 0, 0, 3, 3, 3, 5, 5, 5, 6, 6, 6])
    dst = np.array([4, 2, 7, 1, 4, 7, 2, 7, 1, 2, 4, 1])

    g = dgl.graph((src, dst))
    g.edata['distance'] = torch.FloatTensor([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]) # 50m

    # train
    model = GCN(in_size=num_input_timesteps, hid_size=128, out_size=1, scalar=x_scalar)  # out_size: prediction horizon
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    src, dst = g.edges()
    g = dgl.add_self_loop(g)
    for epoch in range(1500):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model(g, g.ndata['feature']) # [num_dst, batch_size]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst, :, 0], g.ndata['label'][dst, :, 0]) # [num_dst, batch_size], one-step prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            # early stopping
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    print('*************')

    test_loss = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
            loss = loss_fn(pred[dst, :, 0], g.ndata['label'][dst, :, 0])

            print('Train Prediction: {}'.format(pred[dst]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
            loss = loss_fn(pred[dst, :, 0], g.ndata['label'][dst, :, 0])

            test_loss.append(loss.item())

            print('Test Prediction: {}'.format(pred[dst]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            print('*************')

        print('Total Test Loss: {}'.format(np.mean(test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model)))  # 1025