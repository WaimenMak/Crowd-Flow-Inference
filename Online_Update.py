# -*- coding: utf-8 -*-
# @Time    : 10/01/2024 15:43
# @Author  : mmai
# @FileName: Online_Update
# @Software: PyCharm
import numpy as np
import torch
from lib.utils import sliding_win
from lib.metric import masked_rmse_np

class test_then_train_env():
    def __init__(self, data_dict, test_sc, chunk_size, pred_horizon, lags, g):
        self.data_dict = data_dict
        self.pred_horizon = pred_horizon
        self.lags = lags
        self.test_sc = test_sc
        self.g = g
        self.src, self.dst = g.edges()
        self.src_idx = self.src.unique()
        self.dst_idx = self.dst.unique()

        self.chunk_size = chunk_size

    def test_then_train(self, model):
        error_per_chunk = []
        # error_per_chunk = np.zeros((len(self.test_sc), self.pred_horizon, self.chunk_size))
        pred_list = []
        v_list = []
        alpha_list = []
        e_list = []
        for sc in self.test_sc:
            data = self.data_dict[sc]

            observation, label = sliding_win(data, lags=self.lags, horizons=self.pred_horizon)
            observation, label = torch.FloatTensor(observation).permute(2, 0, 1), torch.FloatTensor(label).permute(2, 0, 1)
            for i in range(0, observation.shape[1] - self.chunk_size, self.chunk_size):
                '''test'''
                pred = model.predict(observation[:, i:i+self.chunk_size, :])
                # velocity by time
                v_list.append(model.model.g.edata['v'])
                alpha_list.append(model.model.g.edata['alpha'].repeat(1, self.chunk_size))
                e_list.append(model.model.g.edata['e'])
                if type(pred) == torch.Tensor:
                    pred = pred.detach().cpu().numpy()

                pred_list.append(pred)
                '''every step error'''
                for step in range(self.pred_horizon - 1):
                    last_step_error = masked_rmse_np(pred[self.dst_idx, :, step], label[self.dst_idx, i:i+self.chunk_size, step].detach().cpu().numpy())
                    error_per_chunk.append(last_step_error)
                # error_per_chunk.append(masked_rmse_np(pred[self.dst_idx, :, :], label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy()))

                # error_per_chunk.append(pred[self.dst_idx, :, :] - label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy())
                print(f"error per {sc} chunk: {i}: {last_step_error}")
                '''train'''
                model.update(observation=observation[:, i:i+self.chunk_size, :],
                             pred=pred,
                             label=label[:, i:i+self.chunk_size, :])

        # return model, np.stack(error_per_chunk, axis=0).reshape([-1, self.pred_horizon - 1]), np.stack(pred_list, axis=0)
        return model, np.stack(error_per_chunk, axis=0).reshape([-1, self.pred_horizon - 1]), np.stack(pred_list, axis=0), v_list, alpha_list, e_list

if __name__ == '__main__':
    from Online_Models import Online_Diffusion, Online_Xgboost, Online_MA, ELM, Online_GCN, Online_GAT, Online_LSTM, Online_Diffusion_Density, Single_Model
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import seperate_up_down
    import pickle
    import random

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = 'sc_sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)

    # online testing
    # dataset_name = "crossroad"
    dataset_name = "train_station"
    # test_sc = ['sc_sensor/crossroad2', 'sc_sensor/crossroad4', 'sc_sensor/crossroad5']
    # test_sc = ['sc_sensor/crossroad1', 'sc_sensor/crossroad2', 'sc_sensor/crossroad4', 'sc_sensor/crossroad5',
    #            'sc_sensor/crossroad1', 'sc_sensor/crossroad10', 'sc_sensor/crossroad11', 'sc_sensor/crossroad8', 'sc_sensor/crossroad2_2']
    # train_sc = ['sc_sensor/crossroad3']
    # test_sc = ['sc_sensor/crossroad3']
    # train_sc = ['sc_sensor/train3']
    test_sc = ['sc_sensor/train1','sc_sensor/train3', 'sc_sensor/train5', 'sc_sensor/train2', 'sc_sensor/train6', 'sc_sensor/train4']


    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)
    '''Has to >= 2'''
    pred_horizon = 5 # 3, 5

    if dataset_name == "crossroad":
        file_path = 'graphs/graph_data_crossroad.pkl'
        with open(file_path, 'rb') as file:
            g = pickle.load(file)

    elif dataset_name == "train_station":
        file_path = 'graphs/graph_data_trainstation.pkl'
        with open(file_path, 'rb') as file:
            g = pickle.load(file)

    chunk_size = 30
    lags = 5
    test_env = test_then_train_env(data_dict, test_sc, chunk_size, pred_horizon, lags=5, g=g)
    # model = Single_Model(model_type=Online_LSTM, g=g, pred_horizon=pred_horizon, lags=5, device=device, hidden_units=64,
    #                      chunk_size=chunk_size, num_layers=2, train_steps=200, buffer=True)
    # model = Single_Model(model_type=Online_GCN, g=g, pred_horizon=pred_horizon, lags=5, device=device, hidden_units=128)
    # model = Single_Model(model_type=Online_GAT, g=g, hidden_units=32, pred_horizon=pred_horizon,
    #                          lags=lags, device=device, num_heads=3, train_steps=100, chunk_size=chunk_size)
    # model = Single_Model(model_type=Online_Diffusion, g=g, pred_horizon=pred_horizon, lags=lags, device=device, chunk_size=None, train_steps=100)
    model = Single_Model(model_type=Online_Diffusion_Density, g=g, pred_horizon=pred_horizon, lags=lags, device=device,
                         chunk_size=chunk_size, train_steps=120, buffer=True)
    # model = Single_Model(model_type=Online_Xgboost, g=g, data_name=dataset_name, pred_horizon=pred_horizon, lags=lags, device=device, chunk_size=chunk_size, train_steps=None)
    # model = Single_Model(model_type=Online_MA, g=g, pred_horizon=pred_horizon, lags=lags, device=device, train_steps=None)
    # model = Single_Model(model_type=ELM, g=g, pred_horizon=pred_horizon, lags=5, device=None, hidden_units=100)
    # model, curve_data, _ = test_env.test_then_train(model)
    model, curve_data, _, v, alpha = test_env.test_then_train(model)
    print(curve_data)

    # plot curve data

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(curve_data)), curve_data)
    plt.ylim(0, 10)
    plt.show()

    torch.save(model.model.model.state_dict(), "./checkpoint/diffusion/online_diffusion_density.pth")
    v = torch.cat(v, dim=1).detach().numpy()
    alpha = torch.cat(alpha, dim=1).detach().numpy()
    plt.plot(np.arange(v.shape[1]), v[0, :])
    plt.plot(np.arange(v.shape[1]), alpha[0, :])
    plt.show()

    # save curve data, v, alpha as np array
    np.save(f'checkpoint/{dataset_name}_curve_error.npy', curve_data)
    np.save(f'checkpoint/{dataset_name}_v.npy', v)
    np.save(f'checkpoint/{dataset_name}_alpha.npy', alpha)

