# -*- coding: utf-8 -*-
# @Time    : 18/12/2023 22:03
# @Author  : mmai
# @FileName: MA
# @Software: PyCharm

import numpy as np
import time
import torch
class Moving_Average():
    def __init__(self):
        pass
    def inference(self, data):
        return torch.mean(data, dim=-1) # [num_nodes, batch_size]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
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

    # dataset_name = "crossroad"
    dataset_name = "train_station"
    train_sc = ['../sc_sensor/train6']
    test_sc = ['../sc_sensor/train7']
    # train_sc = ['../sc_sensor/crossroad2']
    # test_sc = ['../sc_sensor/crossroad3']
    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)

    # data_dict['./sc_sensor/crossroad4'] = data_dict['./sc_sensor/crossroad4'][..., :4]
    # for k in data_dict.keys():  # debug
    #     data_dict[k] = data_dict[k][:,[0,3]]

    # x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=5)
    x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
                                                                                 lags=5,
                                                                                 portion=0.5,
                                                                                 shuffle=False)

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

    if dataset_name == "crossroad":
        src = np.array([0, 0, 0, 3, 3, 3, 5, 5, 5, 6, 6, 6])
        dst = np.array([4, 2, 7, 1, 4, 7, 2, 7, 1, 2, 4, 1])
        g = dgl.graph((src, dst))

    if dataset_name == "train_station":
        src = np.array([3,3,3,
                        4,4,4,
                        7,7,7,
                        22,22,22,
                        23,23,23,23,23,
                        8,8,8,8,8,
                        11, 11, 11, 11, 11,
                        14, 14, 14, 14, 14,
                        18, 18, 18,
                        17, 17, 17, 17, 17,
                        13, 13, 13,
                        21, 21, 21,
                        0, 0, 0,
                        12, 12, 12, 12, 12])
        dst = np.array([5,6,23,
                        2,6,23,
                        2,5,23,
                        2,5,6,
                        9,10,15,16,13,
                        22,10,13,15,16,
                        22,9,15,16,13,
                        22,9,10,16,13,
                        12,1,20,
                        13,15,9,10,22,
                        20,1,19,
                        19,1,12,
                        12,19,20,
                        15,16,9,10,22])
        g = dgl.graph((src, dst))

    # g = dgl.graph((src, dst))
    # g.edata['distance'] = torch.FloatTensor([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]) # 50m
    # g.edata['distance'] = torch.FloatTensor([50, 50]) # 50m

    # train
    model = Moving_Average()

    start = time.time()
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()
    loss_fn = torch.nn.MSELoss()

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    print('*************')

    test_loss = []
    # model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature']) # [num_dst, batch_size]
            loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
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
            loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            test_loss.append(loss.item())

            print('Test Prediction: {}'.format(pred[dst_idx]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            # print('Test Velocity: {}'.format(v))
            print('*************')

        print('Total Test Loss: {}'.format(np.mean(test_loss)))