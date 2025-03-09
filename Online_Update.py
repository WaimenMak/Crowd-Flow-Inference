# -*- coding: utf-8 -*-
# @Time    : 10/01/2024 15:43
# @Author  : mmai
# @FileName: Online_Update
# @Software: PyCharm
import numpy as np
import torch
from lib.utils import sliding_win
from lib.metric import masked_rmse_np, masked_mae_np


class TestThenTrainEnv:
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
        pred_list2 = []
        label_list2 = []
        v_list = []
        alpha_list = []
        e_list = []
        F_list = []
        compute_time = []
        route_flow = {}
        route_speed = {}
        for sc in self.test_sc:
            data = self.data_dict[sc]
            observation, label = sliding_win(data, lags=self.lags, horizons=self.pred_horizon) # observation is the training data of the entire scenario
            observation, label = torch.FloatTensor(observation).permute(2, 0, 1), torch.FloatTensor(label).permute(2, 0, 1)
            if self.device.type == 'cuda':
                observation, label = observation.to(model.model.device), label.to(model.model.device) # observation [num_nodes, num_samples, lags]

            route_flow_list = []
            route_speed_list = []

            for i in range(0, observation.shape[1] - self.chunk_size, self.chunk_size):
                '''test'''
                pred = model.predict(observation[:, i:i+self.chunk_size, :])  # pred [num_nodes, num_samples, pred_horizon], multi-step prediction
                # velocity by time
                if model.model.name in ['Online_Diffusion', 'Online_Diffusion_UQ', 'Online_Diffusion_plus']:
                    v_list.append(model.model.g.edata['v'])
                    alpha_list.append(model.model.g.edata['alpha'].repeat(1, self.chunk_size))
                    F_list.append(model.model.g.edata['F'])
                    e_list.append(model.model.g.edata['e'])
                    route_flow_list.append(model.model.g.edata['message'])
                    route_speed_list.append(model.model.g.edata['v'])
                # except:
                #     pass
                if type(pred) == torch.Tensor:
                    pred = pred.detach().cpu().numpy()

                pred_list.append(pred)
                pred_list2.append(pred)
                '''label'''
                label_list2.append(label[:, i:i+self.chunk_size, :].detach().cpu().numpy())
                '''every step error'''
                error_per_step = []
                for step in range(self.pred_horizon - 1):
                    last_step_error = masked_rmse_np(pred[self.dst_idx, :, step], label[self.dst_idx, i:i+self.chunk_size, step].detach().cpu().numpy())
                    error_per_step.append(last_step_error)
                error_per_chunk.append(np.array(error_per_step))

                # error_per_chunk.append(masked_rmse_np(pred[self.dst_idx, :, :], label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy()))

                # error_per_chunk.append(pred[self.dst_idx, :, :] - label[self.dst_idx, i:i+self.chunk_size, :].detach().cpu().numpy())
                self.logger.info(f"error per {sc} chunk: {i}: {last_step_error}")
                '''train on each chunk'''
                start_t = time.time()
                model.update(observation=observation[:, i:(i+self.chunk_size-label.shape[2]), :],   # -pred_horizon to prevent time travel
                             label=label[:, i:(i+self.chunk_size-label.shape[2]), :])

                end_t = time.time()
                compute_time.append(end_t - start_t)

            # predict the rest of the data
            pred = model.predict(observation[:, i+self.chunk_size:, :])
            if type(pred) == torch.Tensor:
                pred = pred.detach().cpu().numpy()
            pred_list2.append(pred)
            label_list2.append(label[:, i+self.chunk_size:, :].detach().cpu().numpy())
            self.logger.info(f"Average compute time for {sc}: {np.mean(compute_time)}")

            if len(route_flow_list) != 0:
                route_flow[sc] = torch.cat(route_flow_list, dim=1).detach().cpu().numpy()
                route_speed[sc] = torch.cat(route_speed_list, dim=1).detach().cpu().numpy()

        return (model, np.stack(error_per_chunk, axis=0), np.stack(pred_list, axis=0), np.concatenate(pred_list2, axis=1), np.concatenate(label_list2, axis=1),
                v_list, alpha_list, F_list, e_list, route_flow, route_speed, compute_time)

if __name__ == '__main__':
    from test_then_train.Online_Models import (OnlineDiffusion, OnlineMa, ELM, OnlineGcn, OnlineGat, OnlineLstm, OnlineGcnlstm,
                               OnlineDiffusionUq, OnlineXgboost, SingleModel, OnlineLstmSingle, OnlineDiffusionPlus)
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import seperate_up_down
    from dgl.data.utils import load_graphs
    import time
    import pickle
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
    parser.add_argument('--dataset', type=str, default='maze', help='Dataset name')
    parser.add_argument('--lags', type=int, default=5, help='Number of lags')
    parser.add_argument('--pred_horizons', type=int, default=7, help='Prediction horizons')
    parser.add_argument('--chunk_size', type=int, default=30, help='Chunk size')
    parser.add_argument('--train_steps', type=int, default=200, help='Training iterations for each chunk')
    parser.add_argument('--model_type', type=str, default='Online_Diffusion_plus', help='Type of model')
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
    # df_dict = process_sensor_data(parent_dir, df_dict)
    #
    # data_dict = gen_data_dict(df_dict)

    # online testing
    dataset_name = args.dataset
    if dataset_name == "train_station":
        # test_sc = ['sc_sensor/train1',
        #    'sc_sensor/train2', 'sc_sensor/train6',
        #    'sc_sensor/train1_2', 'sc_sensor/train7', 'sc_sensor/train8',
        #    'sc_sensor/train9']
        test_sc = ['sc_sensor/train1', 'sc_sensor/train3', 'sc_sensor/train5', 'sc_sensor/train11',
                   'sc_sensor/train2', 'sc_sensor/train6', 'sc_sensor/train4', 'sc_sensor/train12',
                   'sc_sensor/train1_2', 'sc_sensor/train7', 'sc_sensor/train8', 'sc_sensor/train13',
                   'sc_sensor/train9', 'sc_sensor/train10']

        inbound_node = [3, 7, 4, 8, 11, 14, 17, 18, 21, 0]
        outbound_node = [2, 6, 5, 9, 10, 15, 16, 19, 20, 1]
        node_of_interest = outbound_node + [22, 23, 12, 13]

    if dataset_name == "crossroad":
    # test_sc = ['sc_sensor/crossroad2', 'sc_sensor/crossroad4', 'sc_sensor/crossroad5']
        test_sc = ['sc_sensor/crossroad1', 'sc_sensor/crossroad2', 'sc_sensor/crossroad4', 'sc_sensor/crossroad5',
                   'sc_sensor/crossroad1', 'sc_sensor/crossroad10', 'sc_sensor/crossroad11', 'sc_sensor/crossroad8',
                   'sc_sensor/crossroad2_2', 'sc_sensor/crossroad6', 'sc_sensor/crossroad7', 'sc_sensor/crossroad9',]

        inbound_node = [0, 3, 5, 6]
        outbound_node = [1, 2, 4, 7]
        node_of_interest = outbound_node

    if dataset_name == "maze":
        # test_sc = ['sc_sensor/maze14', 'sc_sensor/maze2', 'sc_sensor/maze3', 'sc_sensor/maze1', 'sc_sensor/maze8',
        #            'sc_sensor/maze17', 'sc_sensor/maze16', 'sc_sensor/maze15', 'sc_sensor/maze19', 'sc_sensor/maze12',
        #            'sc_sensor/maze8_2', 'sc_sensor/maze10_2', 'sc_sensor/maze18', 'sc_sensor/maze13', 'sc_sensor/maze20']
        test_sc = ['sc_sensor/maze22', 'sc_sensor/maze2', 'sc_sensor/maze3', 'sc_sensor/maze1', 'sc_sensor/maze8',
                   'sc_sensor/maze21', 'sc_sensor/maze16', 'sc_sensor/maze15', 'sc_sensor/maze19', 'sc_sensor/maze12',
                   'sc_sensor/maze8_2', 'sc_sensor/maze10_2', 'sc_sensor/maze18', 'sc_sensor/maze13', 'sc_sensor/maze20']
        inbound_node = [3, 0, 18, 20, 14, 13, 9, 7]
        outbound_node = [2, 1, 19, 21, 15, 12, 8, 6]
        node_of_interest = outbound_node + [4, 5, 22, 23, 16, 17, 11, 10]


    if dataset_name == "maze":
        with open("sc_sensor/maze/flow_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
    else:
        df_dict = process_sensor_data(parent_dir, df_dict)
        data_dict = gen_data_dict(df_dict)
        data_dict = seperate_up_down(data_dict)
    #seperate upstream and downstream
    # data_dict = seperate_up_down(data_dict)
    '''Has to >= 2'''
    pred_horizon = args.pred_horizons # 3, 5

    g_data = load_graphs('graphs/graphs.bin')
    if dataset_name == "crossroad":
        g = g_data[0][0]
    elif dataset_name == "train_station":
        g = g_data[0][1]
    elif dataset_name == "maze":
        g = g_data[0][2]

    g = g.to(device)
    chunk_size = args.chunk_size
    lags = args.lags
    logger.info(f'Graph device: {g.device}')
    test_env = TestThenTrainEnv(data_dict, test_sc, chunk_size, pred_horizon, lags=lags, g=g, logger=logger)
    test_env.device = device

    if args.model_type == 'Online_LSTM_Single':
        model = SingleModel(model_type=OnlineLstmSingle, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=32,
                            chunk_size=chunk_size, num_layers=2, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_LSTM':
        model = SingleModel(model_type=OnlineLstm, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=64,
                            chunk_size=chunk_size, num_layers=2, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_GCN':
        model = SingleModel(model_type=OnlineGcn, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=128,
                            chunk_size=chunk_size, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_GCNLSTM':
        model = SingleModel(model_type=OnlineGcnlstm, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device, hidden_units=32, num_layers=2,
                            chunk_size=chunk_size, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_GAT':
        model = SingleModel(model_type=OnlineGat, dataset=dataset_name, g=g, hidden_units=32, pred_horizon=pred_horizon,
                            lags=lags, device=device, num_heads=3, train_steps=args.train_steps, chunk_size=chunk_size, buffer=args.buffer)
    elif args.model_type == 'Online_Diffusion':
        model = SingleModel(model_type=OnlineDiffusion, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device,
                            chunk_size=chunk_size, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_Diffusion_plus':
        model = SingleModel(model_type=OnlineDiffusionPlus, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device,
                            chunk_size=chunk_size, train_steps=args.train_steps, buffer=args.buffer)
    elif args.model_type == 'Online_Diffusion_UQ':
        model = SingleModel(model_type=OnlineDiffusionUq, dataset=dataset_name, g=g, pred_horizon=pred_horizon, lags=args.lags, device=device,
                            train_steps=args.train_steps, chunk_size=chunk_size, buffer=args.buffer)
    elif args.model_type == 'Online_Xgboost':
        model = SingleModel(model_type=OnlineXgboost, g=g, dataset=dataset_name, pred_horizon=pred_horizon,
                            lags=lags, device=device, chunk_size=chunk_size, train_steps=None, buffer=False)
    elif args.model_type == 'Online_MA':
        model = SingleModel(model_type=OnlineMa, g=g, pred_horizon=pred_horizon, lags=lags, device=device,
                            chunk_size=chunk_size, train_steps=None)
    # model = Single_Model(model_type=Online_MA, g=g, pred_horizon=pred_horizon, lags=lags, device=device, train_steps=None)
    # model = Single_Model(model_type=ELM, g=g, pred_horizon=pred_horizon, lags=5, device=None, hidden_units=100)
    # model, curve_data, _ = test_env.test_then_train(model)
    # if args.model_type != 'Online_MA'
    logger.info(f'Model graph device: {model.model.g.device}')
    logger.info("#####################")
    start_time = time.time()
    # model, curve_data, prediction, v, alpha, F, e, route_flow, route_speed = test_env.test_then_train(model)
    return_val = test_env.test_then_train(model)
    # unpack the return values
    (model, curve_data, prediction, pred_cat, label_cat,
     v, alpha, F, e, route_flow, route_speed, compute_time) = return_val

    #calculate rmse, mae for pred_cat and label_cat

    rmse = masked_rmse_np(pred_cat[node_of_interest,:, :], label_cat[node_of_interest, :, :])
    logger.info(f"RMSE: {rmse}, chunk size: {chunk_size}")
    mae = masked_mae_np(pred_cat[node_of_interest, :, :], label_cat[node_of_interest, :, :])
    logger.info(f"MAE: {mae}, chunk size: {chunk_size}")
    logger.info(f"--- {(time.time() - start_time)/60} mins ---")


    # save curve data, v, alpha as np array
    logger.info(f"Total error: {np.sum(curve_data)}") # total error
    logger.info(f"Mean error: {np.mean(curve_data)}") # mean error
    if args.save:
        print(12)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_curve_error_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', curve_data)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_prediction_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', prediction)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_compute_time_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', compute_time)
        # save prediction and label
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_pred_cat_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', pred_cat)
        np.save(f'checkpoint/{args.model_type}_{dataset_name}_label_cat_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', label_cat)

        if args.model_type in ['Online_Diffusion', 'Online_Diffusion_UQ', 'Online_Diffusion_plus']:
            v = torch.cat(v, dim=1).detach().cpu().numpy()
            alpha = torch.cat(alpha, dim=1).detach().cpu().numpy()
            F = torch.cat(F, dim=1).detach().cpu().numpy()
            e = torch.cat(e, dim=1).detach().cpu().numpy()
            np.save(f'checkpoint/{args.model_type}_{dataset_name}_v_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', v)
            np.save(f'checkpoint/{args.model_type}_{dataset_name}_alpha_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', alpha)
            np.save(f'checkpoint/{args.model_type}_{dataset_name}_F_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', F)

            np.save(f'checkpoint/{args.model_type}_{dataset_name}_e_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.npy', e)
            pickle.dump(route_flow, open(f'checkpoint/{args.model_type}_{dataset_name}_route_flow_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.pkl', 'wb'))
            pickle.dump(route_speed, open(f'checkpoint/{args.model_type}_{dataset_name}_route_speed_chunk{chunk_size}_lags{lags}_hor{pred_horizon}.pkl', 'wb'))

            # save the model
            torch.save(model.model.model.state_dict(),
                       f"./checkpoint/diffusion/{args.model_type}_{dataset_name}_chunk{chunk_size}_lags{args.lags}_hor{pred_horizon}.pth")
        # elif args.model_type == 'Online_LSTM':
        #     torch.save(model.model.model.state_dict(),
        #                f"./checkpoint/lstm/{args.model_type}_{dataset_name}_chunk{chunk_size}_lags{args.lags}_hor{pred_horizon}.pth")
        # elif args.model_type == 'Online_GCN':
        #     torch.save(model.model.model.state_dict(),
        #                f"./checkpoint/gcn/{args.model_type}_{dataset_name}_chunk{chunk_size}_lags{args.lags}_hor{pred_horizon}.pth")
        # elif args.model_type == 'Online_Xgboost':
        #      model.model.model.save_model('../checkpoint/xgboost/offline_xgboost_cross.model')

