# -*- coding: utf-8 -*-
# @Time    : 11/01/2024 14:06
# @Author  : mmai
# @FileName: Online Models
# @Software: PyCharm

# from Diffusion_Network3 import Diffusion_Model
from Diffusion_Network3_asc import Diffusion_Model
from Diffusion_Network_UQ import NegativeBinomialDistributionLoss, Diffusion_Model_UQ
from Diffusion_Network_density import Diffusion_Model_Density
from baselines.MA import Moving_Average
from baselines.GCN import GCN
from baselines.GAT import GAT
from baselines.LSTM import SimpleLSTM

import torch
from collections import deque
import numpy as np
import xgboost as xgb
from torch import nn
import dgl
from lib.dataloader import replay_buffer
class Single_Model(): # Online Model
    def __init__(self, model_type, **kwargs):
        self.model = model_type(**kwargs) # the first model

    def predict(self, observation):
        pred = self.model.inference(observation)
        return pred  # pred: [num_nodes, bc, numsteps]

    def update(self, **kwargs):
        """
        single model update with regularization
        """
        # self.model.fine_tune(kwargs['observation'], kwargs['label'])
        self.model.train(kwargs['observation'], kwargs['label'])

class Ensemble_DTEL(): # Online Model
    def __init__(self, g, M, model_type, **kwargs):
        self.models_buffer = deque(maxlen=M)
        self.g = g
        self.src, self.dst = g.edges()
        self.src_idx = self.src.unique()
        self.dst_idx = self.dst.unique()
        self.capacity = M
        self.edge_ids_dict = dict()

        self.model_type = model_type
        self.model = model_type(**kwargs) # the first model
        self.models_buffer.append(self.model)
        '''for calculating divergence'''
        for src_node_id in self.src_idx:
            self.edge_ids_dict[src_node_id] = np.where(np.array(self.src) == src_node_id)[0]


    def predict(self, observation):
        pred = 0
        # attention_data_list = []
        for i, model in enumerate(self.models_buffer):
            pred_i = model.inference(observation)
            # attention_data_list.append(model.get_prob())
            pred += model.weight * pred_i
        # torch.stack(attention_data_list, dim=0)
        return pred  # pred: [num_nodes, bc, numsteps], attention_data_list: [num_models, num_edges, batch_size]

    def update_weights(self, observation, label):
        pass

    def update(self, **kwargs):
        try:
            model = self.model_type(**kwargs)
            model.train(kwargs['observation'], kwargs['label'])
        except KeyError as e:
            raise ValueError(f"Missing required argument: {e}")

        for model in self.models_buffer:
            model.fine_tune(kwargs['observation'], kwargs['label'])

        '''remove'''
        if len(self.models_buffer) < self.capacity:
            self.models_buffer.append(model)
        else:
            self.remove(self.models_buffer)
            self.models_buffer.append(model)

    def remove(self, models_buffer):
        self.cal_div()
        for model in models_buffer:
            model.get_prob()
        pass


    def cal_div(self):
        pass

class Base_Learner(): # Base Learner of Online Model
    def __init__(self, pred_horizon, lags, g, device, train_steps, buffer):
        # self.data_dict = data_dict
        self.pred_horizon = pred_horizon - 1
        self.lags = lags
        # self.test_sc = test_sc
        self.device = device
        self.g = g
        self.src, self.dst = g.edges()
        self.src_idx = self.src.unique()
        self.dst_idx = self.dst.unique()
        self.num_edges = g.number_of_edges()
        self.num_nodes = g.number_of_nodes()
        self.train_steps = train_steps
        self.weight = torch.tensor(1)  # weight for each model
        self.iters = train_steps
        self.buffer = buffer

    def train(self, observation, label):
        '''
        :param observation: [batch_size, num_timesteps_input, num_nodes]
        :param label: [batch_size, pred_horizons, num_nodes]
        training for different base learners, can be gcn, xgboost, MLP, etc.
        '''
        pass

    def fine_tune(self, observation, label):
        pass


class Online_Diffusion_Density(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, train_steps, chunk_size, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = Diffusion_Model_Density(num_edges=len(self.src), num_timesteps_input=lags+1, graph=g, horizons=pred_horizon, scalar=None)
        # self.model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_model_density.pth"))
        self.optimizer = torch.optim.Adam([
            {'params': self.model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
            {'params': [self.model.alpha], 'lr': 0.001, 'weight_decay': 1e-5},
            {'params': self.model.density_layer.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}
        ])
        self.prop_opt = torch.optim.Adam(self.model.transition_probability.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = torch.nn.MSELoss()
        self.chunk_size = chunk_size
        if self.buffer:
            # self.batch_size = self.chunk_size
            self.batch_size = 10
            # self.data_capacity = 1000 // self.chunk_size
            self.data_capacity = 1000
        else:
            self.data_capacity = self.chunk_size
        self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        # self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        # self.g.ndata['label'] = label # [node, batch_size, pred_horizon]
        for s in range(observation.shape[1]):
            self.data_buffer.add(observation[:, s, :], label[:, s, :])
        # self.data_buffer.add(observation, label)
        for iteration in range(self.train_steps):
            if iteration == 0:
                x, y = [observation], [label]
                self.g.ndata['feature'] = torch.cat(x, dim=1)
                self.g.ndata['label'] = torch.cat(y, dim=1)
            else:
                try:
                    # x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size)) # 1
                    x, y = self.data_buffer.sample(self.batch_size)
                    # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))

                self.g.ndata['feature'] = torch.stack(x, dim=1)
                self.g.ndata['label'] = torch.stack(y, dim=1)
                # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
            # self.g.ndata['feature'] = torch.cat(x, dim=1)
            # self.g.ndata['label'] = torch.cat(y, dim=1)

            if iteration < self.train_steps / 2:
                pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst]) # [num_dst, batch_size]
                loss = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0]) # [num_dst, batch_size], one-step prediction
            else:
                _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                loss = self.loss_fn(multi_steps_pred[self.dst_idx, :, :], self.g.ndata['label'][self.dst_idx, :, :])

            self.optimizer.zero_grad()
            # self.model.prop_opt.zero_grad()
            loss.backward()
            # self.model.prop_opt.step()
            self.optimizer.step()

            # update transition probability
            if iteration % 2 == 0:

                if self.pred_horizon > 1 and iteration > self.train_steps / 2:
                    '''add multi steps loss'''
                    _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                    loss2 = self.loss_fn(multi_steps_pred[self.dst_idx, :, :], self.g.ndata['label'][self.dst_idx, :, :])
                else:
                    '''single step'''
                    pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                    loss2 = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0])

                self.model.prop_opt.zero_grad()
                loss2.backward()
                self.model.prop_opt.step()

    def inference(self, observation):
        self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        pred, multistep_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
        return multistep_pred
class Online_Diffusion(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, train_steps, chunk_size, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = Diffusion_Model(num_edges=len(self.src), num_timesteps_input=lags+1, graph=g, horizons=pred_horizon, scalar=None)
        # self.model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_model_network3_cross.pth"))
        self.optimizer = torch.optim.Adam([{'params': self.model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
                                                       {'params': [self.model.alpha], 'lr': 0.001, 'weight_decay': 1e-5}])
        self.prop_opt = torch.optim.Adam(self.model.transition_probability.parameters(), lr=0.001, weight_decay=1e-5)
        # self.optimizer = torch.optim.Adam([self.model.alpha], lr=0.001, weight_decay=1e-5)
        # self.prop_opt = torch.optim.Adam([
        #     {'params': self.model.transition_probability.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        #     {'params': [self.model.alpha], 'lr': 0.001, 'weight_decay': 1e-5}
        # ])

        self.loss_fn = torch.nn.MSELoss()
        self.chunk_size = chunk_size
        if self.buffer:
            # self.batch_size = self.chunk_size
            self.batch_size = 10
            # self.data_capacity = 1000 // self.chunk_size
            self.data_capacity = 1000
        else:
            self.data_capacity = self.chunk_size
        self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        # self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        # self.g.ndata['label'] = label # [node, batch_size, pred_horizon]
        for s in range(observation.shape[1]):
            self.data_buffer.add(observation[:, s, :], label[:, s, :])
        # self.data_buffer.add(observation, label)
        for iteration in range(self.train_steps):
            if iteration == 0:
                x, y = [observation], [label]
                self.g.ndata['feature'] = torch.cat(x, dim=1)
                self.g.ndata['label'] = torch.cat(y, dim=1)
            else:
                try:
                    # x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size)) # 1
                    x, y = self.data_buffer.sample(self.batch_size)
                    # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))

                self.g.ndata['feature'] = torch.stack(x, dim=1)
                self.g.ndata['label'] = torch.stack(y, dim=1)
                # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
            # self.g.ndata['feature'] = torch.cat(x, dim=1)
            # self.g.ndata['label'] = torch.cat(y, dim=1)

            if iteration < self.train_steps / 2:
                pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst]) # [num_dst, batch_size]
                loss = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0]) # [num_dst, batch_size], one-step prediction
            else:
                _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                loss = self.loss_fn(multi_steps_pred[self.dst_idx, :, :], self.g.ndata['label'][self.dst_idx, :, :])

            self.optimizer.zero_grad()
            # self.model.prop_opt.zero_grad()
            loss.backward()
            # self.model.prop_opt.step()
            self.optimizer.step()

            # update transition probability
            if iteration % 2 == 0:

                if self.pred_horizon > 1 and iteration > self.train_steps / 2:
                    '''add multi steps loss'''
                    _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                    loss2 = self.loss_fn(multi_steps_pred[self.dst_idx, :, :], self.g.ndata['label'][self.dst_idx, :, :])
                else:
                    '''single step'''
                    pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                    loss2 = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0])

                self.model.prop_opt.zero_grad()
                # self.optimizer.zero_grad()
                loss2.backward()
                self.model.prop_opt.step()
                # self.optimizer.step()

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        pred, multistep_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
        return multistep_pred

    def get_prob(self):
        return self.g.edata['e'] # [num_edges, batch_size]

class Online_Diffusion_UQ(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, train_steps, buffer, chunk_size):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = Diffusion_Model_UQ(num_edges=len(self.src), num_timesteps_input=lags+1, graph=g, horizons=pred_horizon, scalar=None)
        self.optimizer = torch.optim.Adam([{'params': self.model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
                                                       {'params': [self.model.alpha], 'lr': 0.001, 'weight_decay': 1e-5}])
        self.prop_opt = torch.optim.Adam(self.model.transition_probability.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = NegativeBinomialDistributionLoss()
        self.loss_fn_mse = torch.nn.MSELoss()
        self.chunk_size = chunk_size
        if self.buffer:
            # self.batch_size = self.chunk_size
            self.batch_size = 10
            # self.data_capacity = 1000 // self.chunk_size
            self.data_capacity = 1000
        else:
            self.data_capacity = self.chunk_size
        self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        # self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        # self.g.ndata['label'] = label # [node, batch_size, pred_horizon]
        for s in range(observation.shape[1]):
            self.data_buffer.add(observation[:, s, :], label[:, s, :])
        for iter in range(self.train_steps):
            if iter == 0:
                x, y = [observation], [label]
                self.g.ndata['feature'] = torch.cat(x, dim=1)
                self.g.ndata['label'] = torch.cat(y, dim=1)
            else:
                try:
                    # x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size)) # 1
                    x, y = self.data_buffer.sample(self.batch_size)
                    # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))
                self.g.ndata['feature'] = torch.stack(x, dim=1)
                self.g.ndata['label'] = torch.stack(y, dim=1)


            pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst]) # [num_dst, batch_size]
            # assert torch.isnan(pred).sum() == 0, print(f"epoch: {epoch}, pred: {pred}")
            # var = g.ndata['sigma']**2
            # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0], var[dst_idx]) # gaussian loss
            # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0]) # poissson loss
            alpha = self.g.ndata['sigma']
            loss = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0], alpha[self.dst_idx]) # negative binomial loss
            if iter > self.train_steps / 2:
                _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                mse_loss = self.loss_fn_mse(multi_steps_pred[self.dst_idx, :, 1:], self.g.ndata['label'][self.dst_idx,:, 1:])
                loss = loss + mse_loss

            self.optimizer.zero_grad()
            loss.backward()
            # parameters_to_clip = list(model.velocity_model.parameters()) + [model.alpha]
            # clip_grad_norm_(parameters_to_clip, max_norm=1, norm_type=2)
            self.optimizer.step()

            # update transition probability
            if iter % 2 == 0:
                pred = self.model(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                alpha = self.g.ndata['sigma']
                '''single step'''
                nll_loss = self.loss_fn(pred[self.dst_idx], self.g.ndata['label'][self.dst_idx,:, 0], alpha[self.dst_idx]) # negative binomial loss
                '''multi steps'''
                # if self.pred_horizon - 1 > 1:
                if self.pred_horizon > 1 and iter > self.train_steps / 2:
                    _, multi_steps_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
                    mse_loss = self.loss_fn_mse(multi_steps_pred[self.dst_idx, :, 1:], self.g.ndata['label'][self.dst_idx,:, 1:])
                    nll_loss = nll_loss + mse_loss

                self.model.prop_opt.zero_grad()
                nll_loss.backward()
                # clip_grad_norm_(model.transition_probability.parameters(), max_norm=1, norm_type=2)
                self.model.prop_opt.step()

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        self.g.ndata['feature'] = observation # [node, batch_size, num_timesteps_input]
        pred, multistep_pred = self.model.inference(self.g.ndata['feature'][self.src], self.g.ndata['feature'][self.dst])
        return multistep_pred


class Online_Xgboost(Base_Learner):
    def __init__(self, chunk_size, data_name, pred_horizon, lags, g, device, train_steps, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = xgb.XGBRegressor(max_depth=3)
        if data_name == "train_station":
            self.model.load_model('./checkpoint/xgboost/online_xgboost_train_station.model')
        elif data_name == "crossroad":
            self.model.load_model('./checkpoint/xgboost/online_xgboost_cross.model')
        # self.model = xgb.XGBRegressor(objective='reg:squarederror')
        # params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse", "tree_method":"gpu_hist"}
        params_xgb = {"early_stopping_rounds":10, "eval_metric":"rmse"}
        '''Fit the model to the training data'''
        # self.model.set_params(**params_xgb)
        self.chunk_size = chunk_size
        if self.buffer:
            self.data_capacity = 1000 // self.chunk_size  #total 1000 samples
            self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        x_train = observation.reshape([-1, (self.lags+1) * self.num_nodes]).numpy()
        y_train = label.reshape([-1, self.pred_horizon * self.num_nodes]).numpy()
        if self.buffer:
            self.data_buffer.add(x_train, y_train)
            # shuffle data and sample form the buffer
            x_train, y_train = self.data_buffer.sample(len(self.data_buffer))
            self.model.fit(np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0))
        else:
            self.model.fit(x_train, y_train)
    # def train(self, observation, label):
    #     x_train = observation.reshape([-1, (self.lags+1) * self.num_nodes]).numpy()
    #     y_train = label.reshape([-1, self.pred_horizon * self.num_nodes]).numpy()
    #     self.model.fit(x_train, y_train)

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        pred = self.model.predict(observation.permute([1, 0, 2]).reshape([-1, (self.lags+1) * self.num_nodes]).numpy())
        return pred.reshape([-1, self.num_nodes, self.pred_horizon]).transpose([1, 0, 2])


class Online_MA(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, train_steps, buffer=False):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = Moving_Average(horizons=self.pred_horizon)

    def train(self, observation, label):
        pass

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        pred = self.model.inference(observation)
        return pred

class ELM(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, hidden_units, train_steps, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.n_hidden_units = hidden_units
        self.random_weights = np.random.randn(self.num_nodes * (self.lags+1) + 1, self.n_hidden_units)
        self.w_elm = np.random.randn(self.n_hidden_units, self.pred_horizon * self.num_nodes)
    def train(self, observation, label):
        x_train = observation.reshape([-1, (self.lags+1) * self.num_nodes]).numpy()
        y_train = label.reshape([-1, self.pred_horizon * self.num_nodes]).numpy()
        X = np.column_stack([x_train, np.ones([x_train.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        a = X.dot(self.random_weights)
        G = np.maximum(a, 0, a)
        # G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(y_train)

    def inference(self, observation):
        x_train = observation.reshape([-1, (self.lags+1) * self.num_nodes]).numpy()
        X = np.column_stack([x_train, np.ones([x_train.shape[0], 1])])
        a = X.dot(self.random_weights)
        G = np.maximum(a, 0, a)
        # G = np.tanh(X.dot(self.random_weights))
        pred = G.dot(self.w_elm)

        return pred.reshape([-1, self.pred_horizon, self.num_nodes]).transpose([2, 0, 1])


class Online_GCN(Base_Learner):
    def __init__(self, chunk_size, pred_horizon, lags, g, device, hidden_units, train_steps, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = GCN(in_size=self.lags+1, hid_size=hidden_units, out_size=pred_horizon-1, scalar=None)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = torch.nn.MSELoss()
        self.g = dgl.add_self_loop(g)
        self.chunk_size = chunk_size
        self.batch_size = 60
        if self.chunk_size:
            self.data_capacity = 1000 // self.chunk_size
        else:
            self.data_capacity = 1
        self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        self.g.ndata['feature'] = observation
        self.g.ndata['label'] = label # [node, batch_size, pred_horizon]
        self.data_buffer.add(observation, label)

        for iter in range(self.iters):
            if iter == 0:
                # x, y = self.g.ndata['feature'], self.g.ndata['label']
                x, y = [observation], [label]
            else:
                try:
                    x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size))
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))
            pred = self.model(self.g, torch.cat(x, 1)) # [num_dst, batch_size]
            loss = self.loss_fn(pred[self.dst_idx, :, :], torch.cat(y, 1)[self.dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            # loss = self.loss_fn(pred, self.g.ndata['label'])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        pred = self.model.inference(self.g, observation)
        return pred

class Online_GAT(Base_Learner):
    def __init__(self, pred_horizon, lags, g, device, hidden_units, num_heads, train_steps, chunk_size, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        adj_mat = self.g.adjacency_matrix(transpose=False, scipy_fmt="coo")
        self.model = GAT(g=adj_mat, seq_len=self.lags+1, feature_size=1, hidden_dim=hidden_units, out_dim=pred_horizon-1, nodes=self.num_nodes, num_heads=num_heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = torch.nn.MSELoss()
        self.chunk_size = chunk_size
        self.batch_size = 30
        # self.g = dgl.add_self_loop(g)
        if self.buffer:
            self.data_capacity = 1000 // self.chunk_size
        else:
            self.data_capacity = 1
        self.data_buffer = replay_buffer(self.data_capacity)


    def train(self, observation, label):
        self.data_buffer.add(observation, label)
        for iter in range(self.iters):
            if iter == 0:
                x, y = [observation], [label]
            else:
                try:
                    x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size))
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))
            observation = torch.cat(x, dim=1)
            label = torch.cat(y, dim=1)
            pred = self.model(observation.permute(1, 0, 2)) # inpt : [batch_size, node, num_timesteps_input]
            # loss = self.loss_fn(pred[self.dst_idx, :, :], label[self.dst_idx, :, :]) # [num_dst, batch_size], one-step prediction
            loss = self.loss_fn(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fine_tune(self, observation, label):
        pass

    def inference(self, observation):
        pred = self.model(observation.permute(1 ,0, 2))
        return pred

class Online_LSTM(Base_Learner):
    def __init__(self, chunk_size, pred_horizon, lags, g, device, hidden_units, num_layers, train_steps, buffer):
        super().__init__(pred_horizon, lags, g, device, train_steps, buffer)
        self.model = SimpleLSTM(input_size=self.num_nodes, hidden_size=hidden_units,
                                output_size=self.pred_horizon, num_layers=num_layers, num_nodes=self.num_nodes)
        self.chunk_size = chunk_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = torch.nn.MSELoss()
        if self.buffer:
            # self.batch_size = self.chunk_size
            self.batch_size = 30
            # self.data_capacity = 1000 // self.chunk_size
            self.data_capacity = 1000
        else:
            self.data_capacity = self.chunk_size
        self.data_buffer = replay_buffer(self.data_capacity)

    def train(self, observation, label):
        observation = observation.permute(1, 2, 0) # [batch_size, num_time_steps, num_nodes]
        label = label.permute(1, 2, 0).reshape(label.shape[1], -1) # [batch_size, 1, num_nodes * pred_horizon]
        # self.data_buffer.add(observation, label)
        for s in range(observation.shape[0]):
            self.data_buffer.add(observation[s, :, :], label[s, :])
        # shuffle data and sample form the buffer

        for iter in range(self.iters):
            if iter == 0:
                x, y = [observation], [label]
                pred = self.model(torch.cat(x, dim=0))
                label = torch.cat(y, dim=0)
            else:
                try:
                    # x, y = self.data_buffer.sample(np.round(self.batch_size//self.chunk_size)) # 1
                    x, y = self.data_buffer.sample(self.batch_size)
                    # print(f"sampled data: {torch.cat(x, dim=1).shape[1]}")
                except:
                    x, y = self.data_buffer.sample(len(self.data_buffer))
                pred = self.model(torch.stack(x, dim=0))
                label = torch.stack(y, dim=0)

            # pred = self.model(torch.cat(x, dim=0))
            # label = torch.cat(y, dim=0)
            loss = self.loss_fn(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def inference(self, observation):
        self.model.eval()
        observation = observation.permute(1, 2, 0)
        pred = self.model(observation)
        return pred.reshape([-1, self.pred_horizon, self.num_nodes]).permute(2, 0, 1)

