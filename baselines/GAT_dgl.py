# -*- coding: utf-8 -*-
# @Time    : 13/12/2023 15:31
# @Author  : mmai
# @FileName: GAT
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # self.out_dim1 =
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # self.fc2 = nn.Linear(self.out_dim1, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}


    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z' : edges.src['z'], 'e' : edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def forward(self, h):
        # 公式 (1)
        z = self.fc(h)
        # z = self.fc2(z)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=2)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs, dim=3), dim=3)

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, scalar):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads, merge="mean")
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        # self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim, out_dim, 1, merge="mean")
        self.x_scalar = scalar

    def forward(self, h):
        with torch.no_grad():
            h = self.x_scalar.transform(h)
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

from dgl.nn.pytorch import GATConv
class GATRegressor(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_heads):
        super(GATRegressor, self).__init__()
        self.gat1 = GATConv(in_feats, h_feats, num_heads, feat_drop=0.6, attn_drop=0.6)
        self.gat2 = GATConv(h_feats * num_heads, out_feats, 1, feat_drop=0.6, attn_drop=0.6)

    def forward(self, g, inputs):
        # bg: batched graph
        # inputs: node features of shape (batch_size, num_nodes, in_feats)
        h = self.gat1(g, inputs)
        h = h.reshape(h.shape[0], h.shape[1], -1) # flatten the output of multi-head attention
        h = self.gat2(g, h)
        return h.squeeze()

    def inference(self, g, inputs):
        self.eval()
        h = self.forward(g, inputs).squeeze()
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
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())

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

    # src, dst = g.edges()
    # src_idx = np.unique(src)
    # dst_idx = np.unique(dst)
    # adj_mat = g.adj()

    # train
    model = GATRegressor(in_feats=num_input_timesteps, h_feats=32, out_feats=pred_horizon-1, num_heads=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()

    # g = dgl.add_self_loop(g)
    for epoch in range(1000):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # y = y.permute(2, 0, 1)
            pred = model(g, g.ndata['feature']).squeeze() # inpt : [batch_size, node, num_timesteps_input]
            # loss = loss_fn(pred, y[:, 0, :])
            # loss = loss_fn(pred[dst_idx, :, :], y[dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            # early stopping
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    if dataset_name == "crossroad":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat2_crossroad_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "train_station":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat2_trainstation_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "maze":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat2_maze_lags{lags}_hor{pred_horizon}.pth')

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=y_test.shape[0])
    test_dataloader = DataLoader(test_dataset, batch_size=y_test.shape[0])

    test_loss = []
    train_loss = []
    multi_steps_train_loss = []
    multi_steps_test_loss = []
    model.eval()
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # y = y.permute(2, 0, 1)
            pred = model.inference(g, g.ndata['feature']) # [num_dst, batch_size]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, 0], g.ndata['label'][dst_idx, :, 0])
            train_loss.append(loss.item())

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_train_loss.append(multisteps_loss.item())

            print('Train Loss: {}'.format(loss.item()))
            print('Train Multi-Steps Loss: {}'.format(multisteps_loss.item()))
            # print('Train Prediction: {}'.format(pred[dst_idx]))
            # print('Train Ground Truth: {}'.format(y[dst_idx,:, 0]))
            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # y = y.permute(2, 0, 1)
            pred = model.inference(g, g.ndata['feature'])
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, 0], g.ndata['label'][dst_idx, :, 0])

            test_loss.append(loss.item())
            multisteps_loss = loss_fn(pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_test_loss.append(multisteps_loss.item())

        print('Total Train Loss: {}'.format(np.mean(train_loss)))
        print('Multi-Steps Train Loss: {}'.format(np.mean(multi_steps_train_loss)))
        print('Total Test Loss: {}'.format(np.mean(test_loss)))
        print('Multi-Steps Test Loss: {}'.format(np.mean(multi_steps_test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model)))