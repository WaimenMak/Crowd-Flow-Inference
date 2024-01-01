# -*- coding: utf-8 -*-
# @Time    : 18/12/2023 22:27
# @Author  : mmai
# @FileName: XGBOOST
# @Software: PyCharm

import xgboost as xgb

if __name__ == '__main__':
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    import dgl
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import numpy as np
    import random
    import time

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
    # train_sc = ['../sc_sensor/crossroad2', '../sc_sensor/crossroad9', '../sc_sensor/crossroad10', '../sc_sensor/crossroad11']
    # test_sc = ['../sc_sensor/crossroad3']

    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)

    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=5)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=5,
    #                                                                              portion=0.5,
    #                                                                              shuffle=False)

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


    src_idx = np.unique(src)
    dst_idx = np.unique(dst)

    # g = dgl.graph((src, dst))
    # x_train shape [num_samples, num_input_timesteps, num_nodes]
    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node
    num_dst = len(dst_idx) # number of destination nodes
    pred_horizon = y_train.shape[1] # number of prediction horizon

    x_train = x_train.reshape([-1, num_input_timesteps * num_nodes])
    x_val = x_val.reshape([-1, num_input_timesteps * num_nodes])
    x_test = x_test.reshape([-1, num_input_timesteps * num_nodes])
    y_train = y_train.reshape([-1, pred_horizon * num_nodes])
    y_val = y_val.reshape([-1, pred_horizon * num_nodes])
    y_test = y_test.reshape([-1, pred_horizon * num_nodes])

    # set seed

    #normalization

    # Define the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')
    # params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse", "tree_method":"gpu_hist"}
    params_xgb = {"early_stopping_rounds":5, "eval_metric":"rmse"}
    # Fit the model to the training data
    model.set_params(**params_xgb)
    start_time = time.time()
    eval_set = [(x_val, y_val)]
    model.fit(x_train, y_train, eval_set=eval_set, verbose=True)   # annotate for testing
    end_time = time.time()
    total_train_time = end_time - start_time

    # predict
    y_pred = model.predict(x_test)
    y_train = model.predict(x_train)
    #get the prediction MSE
    test_loss = np.mean((y_pred[..., dst_idx] - y_test[..., dst_idx])**2)
    train_loss = np.mean((y_train[..., dst_idx] - y_train[..., dst_idx])**2)
    print('*************')
    print('Total Train Time: {}'.format(total_train_time))
    print('Total Test Loss: {}'.format(test_loss))

    # save the model
    if dataset_name == "crossroad":
        model.save_model('../checkpoint/xgboost/xgboost_cross.model')
    if dataset_name == "train_station":
        model.save_model('../checkpoint/xgboost/xgboost_train_station.model')


