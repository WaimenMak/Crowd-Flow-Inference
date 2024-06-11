# -*- coding: utf-8 -*-
# @Time    : 13/12/2023 18:43
# @Author  : mmai
# @FileName: GAT
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
class GATLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, nodes):
        super(GATLayer, self).__init__()
        self.adj = adj
        self.nodes = nodes
        # self.bc = bc
        self.feat = input_dim  # seq_len * features
        self.output_dim = output_dim
        self.hidden_dim = 32
        self.mask = torch.zeros([self.nodes, self.nodes])
        self.mask[self.adj.row, self.adj.col] = 1

        # self.W = nn.Parameter(torch.FloatTensor(size=(input_dim, output_dim))) # could be change to RNN encoder
        self.mlp = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
                                 nn.ReLU(), nn.Linear(self.hidden_dim, output_dim)) # [bc, node, output_dim*2] --> [bc, node, output_dim]

        self.atten_W = nn.Parameter(torch.FloatTensor(size=(2*self.output_dim, 1)))
        self.leaky_relu = nn.LeakyReLU(0.1)
        # nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.atten_W, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.atten_W.data, gain=1.414)
        # nn.init.normal_(self.W)
        # nn.init.normal_(self.atten_W)

    def edge_attention_concatenate(self, z):
        bc = z.shape[0]
        b = z.repeat([1, 1, self.nodes]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        c = z.repeat([1, self.nodes, 1]).reshape([bc, self.nodes, self.nodes, self.output_dim]) #
        e = torch.cat([b,c],dim=3).reshape(bc, -1, 2*self.output_dim) # [bc, node*node, output_dim*2]
        # mask = mask.repeat([self.bc, 1, 1])
        atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(bc, self.nodes, self.nodes)) * self.mask.unsqueeze(0) #[bc, node, node] batch attention scores
        # atten_mat = self.leaky_relu(torch.matmul(e, self.atten_W).reshape(bc, self.nodes, self.nodes)) # not just consider neighbors
        atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat

    def edge_attention_innerprod(self, z):
        atten_mat = torch.bmm(z, z.transpose(2, 1)) #[bc node feat] * [bc feat node]
        mask = torch.zeros([self.nodes, self.nodes])
        mask[self.adj.row, self.adj.col] = 1
        atten_mat = self.leaky_relu(atten_mat) * mask.unsqueeze(0)
        # atten_mat = self.leaky_relu(atten_mat)
        # atten_mat.data.masked_fill_(torch.eq(atten_mat, 0), -float(1e16))

        return atten_mat



    def forward(self, h):
        '''
        h: [bc, node, seq_len*feature]
        output_dim is the out dimenstion after original vector multiplied by W
        :param graph:
        :param data: [bc, seq, node, feature]  bc: shape[0], node: shape[2]
        :return:
        '''
        # h = data.transpose(0, 2, 1, 3).view(self.bc, self.nodes, -1) #[bc, node, feat]
        bc = h.shape[0]
        # z = torch.matmul(h.reshape(bc*self.nodes, self.feat), self.W)  # z = Wh  #[bc*node, output_dim]
        z = self.mlp(h.reshape(bc*self.nodes, self.feat))  #[bc*node, output_dim]
        z = z.reshape(bc, self.nodes, self.output_dim)  # [bc, nodes, output_dim]
        # atten_mat = self.edge_attention_concatenate(z)
        atten_mat = self.edge_attention_innerprod(z)
        # normallization
        # atten_mat = F.normalize(atten_mat, p=1, dim=2)
        atten_mat = F.softmax(atten_mat, dim=2)
        # atten_mat = self.mask.unsqueeze(0) * atten_mat
        h_agg = torch.matmul(atten_mat, z)  # [bc, node, output_dim]
        return h_agg

class MultiHeadGATLayer(nn.Module):
    def __init__(self, adj, input_dim, output_dim, nodes, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.merge = merge
        for i in range(num_heads):
            self.heads.append(GATLayer(adj, input_dim, output_dim, nodes))

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=2)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, seq_len, feature_size, hidden_dim, out_dim, nodes, num_heads):
        super(GAT, self).__init__()
        self._output_dim = out_dim
        self.layer1 = MultiHeadGATLayer(g, seq_len*feature_size, hidden_dim, nodes, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, nodes, 1)

    def forward(self, h):
        '''

        :param h:  [bc, node, feat]
        :return:  [bc, node, output_dim]
        '''
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h.permute(1, 0, 2) #return [bc, node, output_dim] --> [node, bc, output_dim]

    def inference(self, h):
        '''

        :param h:  [bc, node, feat]
        :return:  [bc, node, output_dim]
        '''
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h.permute(1, 0, 2).clamp(min=0) #return [bc, node, output_dim] --> [node, bc, output_dim]

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

    dataset_name = "crossroad"
    # dataset_name = "train_station"
    # dataset_name = "maze"
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


    src, dst = g.edges()
    src_idx = np.unique(src)
    dst_idx = np.unique(dst)
    adj_mat = g.adj()

    # train
    model = GAT(g=adj_mat, seq_len=num_input_timesteps, feature_size=1, hidden_dim=32, out_dim=pred_horizon-1, nodes=num_nodes, num_heads=3)  # out_size: prediction horizon
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()


    # g = dgl.add_self_loop(g)
    for epoch in range(1000):
        l = []
        for i, (x, y) in enumerate(train_dataloader):
            # g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            # g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            y = y.permute(2, 0, 1)
            pred = model(x.permute(0, 2, 1)) # inpt : [batch_size, node, num_timesteps_input]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, :], y[dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l.append(loss.item())
            # early stopping
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    if dataset_name == "crossroad":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat_crossroad_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "train_station":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat_trainstation_lags{lags}_hor{pred_horizon}.pth')
    if dataset_name == "maze":
        torch.save(model.state_dict(), f'../checkpoint/gat/gat_maze_lags{lags}_hor{pred_horizon}.pth')

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
            y = y.permute(2, 0, 1)
            pred = model.inference(x.permute(0, 2, 1)) # inpt : [batch_size, node, num_timesteps_input]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, 0], y[dst_idx, :, 0])
            train_loss.append(loss.item())

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(pred[dst_idx, :, :], y[dst_idx, :, :])
            multi_steps_train_loss.append(multisteps_loss.item())

            print('Train Loss: {}'.format(loss.item()))
            print('Train Multi-Steps Loss: {}'.format(multisteps_loss.item()))
            # print('Train Prediction: {}'.format(pred[dst_idx]))
            # print('Train Ground Truth: {}'.format(y[dst_idx,:, 0]))
            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            y = y.permute(2, 0, 1)
            pred = model.inference(x.permute(0, 2, 1)) # inpt : [batch_size, node, num_timesteps_input]
            # loss = loss_fn(pred, y[:, 0, :])
            loss = loss_fn(pred[dst_idx, :, 0], y[dst_idx, :, 0])

            test_loss.append(loss.item())
            multisteps_loss = loss_fn(pred[dst_idx, :, :], y[dst_idx, :, :])
            multi_steps_test_loss.append(multisteps_loss.item())

        print('Total Train Loss: {}'.format(np.mean(train_loss)))
        print('Multi-Steps Train Loss: {}'.format(np.mean(multi_steps_train_loss)))
        print('Total Test Loss: {}'.format(np.mean(test_loss)))
        print('Multi-Steps Test Loss: {}'.format(np.mean(multi_steps_test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model)))
