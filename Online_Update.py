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
    def __init__(self, data_dict, test_sc, chunk_size, pred_horizon, lags, g, logger):
        self.data_dict = data_dict
        self.pred_horizon = pred_horizon
        self.lags = lags
        self.test_sc = test_sc
        self.logger = logger
        self.device = None
        self.src, self.dst = g.edges()
        self.src_idx = self.src.unique().cpu().numpy()
        self.dst_idx = self.dst.unique().cpu().numpy()

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
            if self.device.type == 'cuda':
                observation, label = observation.to(model.model.device), label.to(model.model.device) # observation [num_nodes, num_samples, lags]

            for i in range(0, observation.shape[1] - self.chunk_size, self.chunk_size):
                '''test'''
                pred = model.predict(observation[:, i:i+self.chunk_size, :])
                # velocity by time
                try:
                    v_list.append(model.model.g.edata['v'])
                    alpha_list.append(model.model.g.edata['alpha'].repeat(1, self.chunk_size))
                    e_list.append(model.model.g.edata['e'])
                except:
                    pass
                if type(pred) == torch.Tensor:
                    pred = pred.detach().cpu().numpy()

                pred_list.append(pred)
                '''every step error'''
                for step in range(self.pred_horizon - 1):
                    last_step_error = masked_rmse_np(pred[self.dst_idx, :, step], label[self.dst_idx, i:i+self.chunk_size, step].detach().cpu().numpy())
                    error_per_chunk.append(last_step_error)
                # error_per_chunk.append(masked_rmse_np(pred[self.dst_idx, :, :], label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy()))

                # error_per_chunk.append(pred[self.dst_idx, :, :] - label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy())
                self.logger.info(f"error per {sc} chunk: {i}: {last_step_error}")
                '''train'''
                model.update(observation=observation[:, i:(i+self.chunk_size-label.shape[2]), :],   # -pred_horizon to prevent time travel
                             label=label[:, i:(i+self.chunk_size-label.shape[2]), :])

        # return model, np.stack(error_per_chunk, axis=0).reshape([-1, self.pred_horizon - 1]), np.stack(pred_list, axis=0)
        return model, np.stack(error_per_chunk, axis=0).reshape([-1, self.pred_horizon - 1]), np.stack(pred_list, axis=0), v_list, alpha_list, e_list

if __name__ == '__main__':
    from Online_Models import (Online_Diffusion, Online_MA, ELM, Online_GCN, Online_GAT, Online_LSTM,
                               Online_Diffusion_UQ, Online_Xgboost, Single_Model, Online_LSTM_Single)
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import seperate_up_down
    from dgl.data.utils import load_graphs
    import time
    import random
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    # random.seed(1)

    import argparse
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='train_station', help='Dataset name')
    parser.add_argument('--lags', type=int, default=5, help='Number of lags')
    parser.add_argument('--pred_horizons', type=int, default=5, help='Prediction horizons')
    parser.add_argument('--chunk_size', type=int, default=10, help='Chunk size')
    parser.add_argument('--train_steps', type=int, default=130, help='Training iterations for each chunk')
    parser.add_argument('--model_type', type=str, default='Online_Diffusion', help='Type of model')
    parser.add_argument('--no-buffer', dest='buffer', action='store_false',
                        help='Whether not to use buffer')
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Whether not to save the results')
    parser.set_defaults(save=True)
    parser.set_defaults(buffer=True)
    # parser.add_argument('--cl', type=bool, default=True,help='whether to do curriculum learning')
    args = parser.parse_args()

    # Log the arguments
    logger.info('Arguments:')
    logger.info(f'- Dataset: {args.dataset}')
    logger.info(f'- lags: {args.lags}')
    logger.info(f'- pred_horizons: {args.pred_horizons}')
    logger.info(f'- chunk_size: {args.chunk_size}')
    logger.info(f'- train_steps: {args.train_steps}')
    logger.info(f'- model_type: {args.model_type}')
    logger.info(f'- buffer: {args.buffer}')
    # logger.info(f'- save: {args.save}')

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
    test_sc = ['sc_sensor/train1', 'sc_sensor/train3', 'sc_sensor/train5',
               'sc_sensor/train2', 'sc_sensor/train6', 'sc_sensor/train4',
               'sc_sensor/train1_2', 'sc_sensor/train7']

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)
    '''Has to >= 2'''
    pred_horizon = args.pred_horizons # 3, 5

    g_data = load_graphs('./graphs/graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]

    g = g.to(device)
    chunk_size = args.chunk_size
    lags = args.lags
    logger.info(f'Graph device: {g.device}')
    test_env = test_then_train_env(data_dict, test_sc, chunk_size, pred_horizon, lags=lags, g=g, logger=logger)
    test_env.device = device

    if args.model_type == 'Online_LSTM_Single':
        model = Single_Model(model_type=Online_LSTM_Single, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=32,
                             chunk_size=chunk_size, num_layers=2, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_LSTM':
        model = Single_Model(model_type=Online_LSTM, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=64,
                             chunk_size=chunk_size, num_layers=2, train_steps=args.train_steps, buffer=args.buffer)
    # model = Single_Model(model_type=Online_GCN, g=g, pred_horizon=pred_horizon, lags=5, device=device, hidden_units=128)
    # model = Single_Model(model_type=Online_GAT, g=g, hidden_units=32, pred_horizon=pred_horizon,
    #                          lags=lags, device=device, num_heads=3, train_steps=100, chunk_size=chunk_size)
    elif args.model_type == 'Online_Diffusion':
        model = Single_Model(model_type=Online_Diffusion, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device,
                                 chunk_size=chunk_size, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_Diffusion_UQ':
        model = Single_Model(model_type=Online_Diffusion_UQ, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device,
                     train_steps=args.train_steps, chunk_size=chunk_size, buffer=args.buffer)
    elif args.model_type == 'Online_Xgboost':
        model = Single_Model(model_type=Online_Xgboost, g=g, dataset=dataset_name, pred_horizon=pred_horizon,
                             lags=lags, device=device, chunk_size=chunk_size, train_steps=None, buffer=False)
    # model = Single_Model(model_type=Online_MA, g=g, pred_horizon=pred_horizon, lags=lags, device=device, train_steps=None)
    # model = Single_Model(model_type=ELM, g=g, pred_horizon=pred_horizon, lags=5, device=None, hidden_units=100)
    # model, curve_data, _ = test_env.test_then_train(model)
    logger.info(f'Model graph device: {model.model.g.device}')
    logger.info("#####################")
    start_time = time.time()
    model, curve_data, prediction, v, alpha, _ = test_env.test_then_train(model)
    logger.info(f"--- {(time.time() - start_time)/60} mins ---")

    # plot curve data
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(len(curve_data)), curve_data)
    # plt.ylim(0, 10)
    # plt.show()
    # plt.plot(np.arange(v.shape[1]), v[0, :])
    # plt.plot(np.arange(v.shape[1]), alpha[0, :])
    # plt.show()

    # save curve data, v, alpha as np array
    logger.info(f"Total error: {np.sum(curve_data)}") # total error
    if args.save:
        print(12)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_curve_error_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', curve_data)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_prediction_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', prediction)
        if args.model_type == 'Online_Diffusion' or args.model_type == 'Online_Diffusion_UQ':
            torch.save(model.model.model.state_dict(),
                       f"./checkpoint/diffusion/{args.model_type}_chunk{chunk_size}_lags{args.lags}_hor{pred_horizon}.pth")
            v = torch.cat(v, dim=1).detach().cpu().numpy()
            alpha = torch.cat(alpha, dim=1).detach().cpu().numpy()
            np.save(f'checkpoint/{args.model_type}_{dataset_name}_v_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', v)
            np.save(f'checkpoint/{args.model_type}_{dataset_name}_alpha_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', alpha)

