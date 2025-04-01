# -*- coding: utf-8 -*-
# @Time    : 18/12/2023 22:03
# @Author  : mmai
# @FileName: MA
# @Software: PyCharm

import numpy as np
import time
import torch
class Moving_Average():
    def __init__(self, horizons=0):
        self.horizons = horizons
    def inference(self, data):
        pred = []
        pred.append(torch.mean(data, dim=-1))
        data = torch.cat([data, pred[0].unsqueeze(-1)], dim=-1)
        for i in range(1, self.horizons):
            # pred.append(torch.mean(data[..., i:], dim=-1))
            pred.append(torch.mean(data, dim=-1))
            data = torch.cat([data, pred[-1].unsqueeze(-1)], dim=-1)
        return torch.stack(pred, dim=-1) # [num_nodes, batch_size]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import dgl
    import random
    import pickle
    from dgl.data.utils import load_graphs

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = '../sc_sensor'


    # dataset_name = "maze"
    dataset_name = "edinburgh"
    if dataset_name == "crossroad":
        train_sc = ['sc_sensor/crossroad2']
        test_sc = ['sc_sensor/crossroad1', 'sc_sensor/crossroad11', 'sc_sensor/crossroad13']
    elif dataset_name == "train_station":
        train_sc = ['sc_sensor/train13']
        test_sc = ['sc_sensor/train2']
    elif dataset_name == "maze":
        train_sc = ['sc_sensor/maze0']
        # train_sc = ['sc_sensor/maze2']
        test_sc = ['sc_sensor/maze13', 'sc_sensor/maze4']
    elif dataset_name == "edinburgh":
        train_sc = ['26Aug']
        test_sc = ['27Aug']

    # Load data
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

    # dataset_name = "crossroad"
    # dataset_name = "train_station"
    # train_sc = ['../sc_sensor/crossroad3', '../sc_sensor/crossroad8', '../sc_sensor/crossroad2', '../sc_sensor/crossroad5']
    # test_sc = ['../sc_sensor/crossroad1', '../sc_sensor/crossroad11', '../sc_sensor/crossroad13']
    # train_sc = ['../sc_sensor/train1']
    # test_sc = ['../sc_sensor/train2']
    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    # data_dict['./sc_sensor/crossroad4'] = data_dict['./sc_sensor/crossroad4'][..., :4]
    # for k in data_dict.keys():  # debug
    #     data_dict[k] = data_dict[k][:,[0,3]]

    pred_horizon = 7 # 3, 5
    lags = 5
    if dataset_name == "edinburgh":
        pred_horizon = 2
        lags = 6

    # x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=lags, horizons=pred_horizon, shuffle=True)
    x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
                                                                                 lags=lags,
                                                                                 horizons=pred_horizon,
                                                                                 portion=0.7,
                                                                                 shuffle=True)

    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(x_train,
                                y_train, batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # set seed
    #normalization
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())

    #processing graph data
    # src = np.array([0])
    # dst = np.array([1])

    # src = np.array([0, 2])
    # dst = np.array([3, 1])
    g_data = load_graphs('../graphs/4graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]
    elif dataset_name == "maze":
        g = g_data[0][2]
    elif dataset_name == "edinburgh":
        g = g_data[0][3]


    # g = dgl.graph((src, dst))
    # g.edata['distance'] = torch.FloatTensor([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]) # 50m
    # g.edata['distance'] = torch.FloatTensor([50, 50]) # 50m

    # train
    model = Moving_Average(horizons=pred_horizon-1)

    start = time.time()
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()
    loss_fn = torch.nn.MSELoss()

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=y_test.shape[0])
    test_dataloader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    print('*************')

    test_loss = []
    multi_steps_train_loss = []
    multi_steps_test_loss = []
    # model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature']) # [num_dst, batch_size]
            loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            multi_steps_loss = loss_fn(pred[dst_idx,:, :], g.ndata['label'][dst_idx,:, :])
            multi_steps_train_loss.append(multi_steps_loss.item())
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]

            print('Train Prediction: {}'.format(pred[dst_idx]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            # print('Train Velocity: {}'.format(v))
            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature']) # [num_dst, batch_size]
            loss = loss_fn(pred[dst_idx,:, 0], g.ndata['label'][dst_idx,:, 0])
            multi_steps_loss = loss_fn(pred[dst_idx,:, :], g.ndata['label'][dst_idx,:, :])
            multi_steps_test_loss.append(multi_steps_loss.item())

            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            test_loss.append(loss.item())

            print('Test Prediction: {}'.format(pred[dst_idx]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            # print('Test Velocity: {}'.format(v))
            print('*************')

        print('Total Test Loss: {}'.format(np.mean(test_loss)))
        print('Multi-Step Test Loss: {}'.format(np.mean(multi_steps_test_loss)))