# -*- coding: utf-8 -*-
# @Time    : 01/06/2024 21:18
# @Author  : mmai
# @FileName: Diffusion_Network5
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as func
from dgl.nn.functional import edge_softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Velocity_Model_NAM(nn.Module):
    def __init__(self, num_timesteps_input, hidden_units, scalar):
        super(Velocity_Model_NAM, self).__init__()
        # MLP
        self._hidden_size = hidden_units
        # self.g = g
        self.linear11 = torch.nn.Linear(num_timesteps_input, self._hidden_size)
        self.ln11 = nn.LayerNorm(self._hidden_size)
        self.linear12 = torch.nn.Linear(self._hidden_size, 1)

        self.linear21 = torch.nn.Linear(num_timesteps_input, self._hidden_size)
        self.ln21 = nn.LayerNorm(self._hidden_size)
        self.linear22 = torch.nn.Linear(self._hidden_size, 1)

        self.linear3 = torch.nn.Linear(2 + 1, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.x_scalar = scalar

    def forward(self, upstream, downstream, T):
        # upstream : [node, batch_size, num_timesteps_input]
        # downstream : [node, batch_size, num_timesteps_input]

        # with torch.no_grad():
        #     upstream = self.x_scalar.transform(upstream)
        #     downstream = self.x_scalar.transform(downstream)
        x_up = self.linear11(upstream)
        x_up = self.dropout(x_up)
        x_up = self.ln11(x_up)
        x_up = self.relu(x_up)
        x_up = self.linear12(x_up)
        # x_up = self.dropout(x_up)
        x_up = torch.sigmoid(x_up)

        x_down = self.linear21(downstream)
        x_down = self.dropout(x_down)
        x_down = self.ln21(x_down)
        x_down = self.relu(x_down)
        x_down = self.linear22(x_down)
        # x_down = self.dropout(x_down)
        x_down = torch.sigmoid(x_down)
        T = T.unsqueeze(1).repeat(1, upstream.shape[1], 1)
        x = torch.cat((x_up, x_down, T), dim=2)
        out = self.linear3(x) # [num of edges, batch_size, 1]
        # out = self.relu(out)
        out = torch.log(1 + torch.exp(out))
        out = out.squeeze(2).transpose(1, 0) # [batch_size, num of edges]

        return out.clamp(max=3)

    # def inference(self, upstream, downstream):
    #     with torch.no_grad():
    #         up_sum = torch.sum(upstream, dim=2)  # [node, batch_size]
    #         up_idx = (up_sum == 0).nonzero(as_tuple=True)
    #         out = self.forward(upstream, downstream)
    #         out[up_idx[1], up_idx[0]] = 0
    #
    #     return out

class Probabilistic_Model(torch.nn.Module):
    def __init__(self, graph, num_edges, num_timesteps_input, hidden_units, scalar):
        """

        :param num_nodes: number of ancestor nodes
        :param num_timesteps_input: time steps of input
        """
        super(Probabilistic_Model, self).__init__()
        self.num_timesteps_input = num_timesteps_input
        self.num_edges = num_edges
        self.g = graph
        self._hidden_size = hidden_units
        self.scalar = scalar
        #params
        self.fc = nn.Linear(self.num_timesteps_input, self._hidden_size, bias=False)
        # self.fc_up  = nn.Linear(self._hidden_size, 1, bias=False)
        # self.fc_down = nn.Linear(self._hidden_size, 1, bias=False)
        self.attn_fc1 = nn.Linear(2 * self._hidden_size + 1, 1, bias=False)
        # self.attn_fc1 = nn.Linear(2 * self._hidden_size, 1, bias=False)
        # self.attn_fc = nn.Linear(3, 1, bias=False)
        # self.ln2 = nn.LayerNorm(2 * self._hidden_size)
        self.ln2 = nn.LayerNorm(2 * self._hidden_size + 1)
        # self.dropout = nn.Dropout(0.5)
        self.device = device
        self.to(self.device)

    def edge_attention(self, edges):
        # up2down = torch.sum(edges.src['feature'] * edges.data['diffusion'], dim=2).unsqueeze(-1) # [num_edges, batch_size]
        # up2down = edge_softmax(self.g, up2down, norm_by='src') # [num_edges, batch_size]
        # F = torch.sum(edges.data["diffusion"], dim=2, keepdim=True) # [num_edges, batch_size, num_timesteps_input]
        # v = edges.data['v'].unsqueeze(-1) # [num_edges, batch_size, 1]
        T = edges.data['T'].unsqueeze(-1) # [num_edges, batch_size, 1]
        # up = self.fc_up(torch.sigmoid(self.dropout(edges.src['embedding']))) # [num_edges, batch_size, 1]
        # down = self.fc_down(torch.sigmoid(self.dropout(edges.dst['embedding'])))
        # down = self.dropout(down)
        # z = torch.cat([edges.src['embedding'], edges.dst['embedding'], F.detach()], dim=2)  # [num_edges, bc, 2 * 16]
        z = torch.cat([edges.src['embedding'], edges.dst['embedding'], T.detach()], dim=2)  # [num_edges, bc, 2 * 16]
        # z = torch.cat([edges.src['embedding'], edges.dst['embedding'], v.detach()], dim=2)  # [num_edges, bc, 2 * 16]
        # z = torch.cat([edges.src['embedding'], edges.dst['embedding']], dim=2)  # [num_edges, bc, 2 * hid + 1]
        # z = edges.src['embedding'] + edges.dst['embedding']
        # z = torch.cat([up, down, up2down], dim=2)  # [num_edges, bc, 3]
        # a = self.attn_fc1(z)
        # a = self.dropout(a)
        # a = self.ln(a)
        # a = torch.sigmoid(a)
        z = self.ln2(z)
        a = self.attn_fc1(z).squeeze(dim=2)  # [num_edges, bc, 2 * hidden size + 1] --> [num_edges, bc]
        score = edge_softmax(self.g, func.leaky_relu(a), norm_by='src') # [num_edges, bc]
        return {'e': score}

    def forward(self, features):
        # with torch.no_grad():
        #     features = self.scalar.transform(features)
        z = self.fc(features)
        return z


class Diffusion_Model(torch.nn.Module):
    def __init__(self, num_edges, num_timesteps_input, graph, horizons, scalar, device, time_units=10):
        """

        :param num_nodes: number of ancestor nodes
        :param num_timesteps_input: time steps of input
        """
        super(Diffusion_Model, self).__init__()
        v_model_hid_dim = 32
        p_model_hid_dim = 64
        self.velocity_model = Velocity_Model_NAM(num_timesteps_input, hidden_units=v_model_hid_dim, scalar=scalar)
        # self.velocity_model = Velocity_Model(num_timesteps_input, hidden_units=16, out_units=8, scalar=scalar)
        self.transition_probability = Probabilistic_Model(graph, num_edges, num_timesteps_input, hidden_units=p_model_hid_dim, scalar=scalar)
        self.scalar = scalar
        # self.residual_block = ResidualBlock(num_timesteps_input, scalar=scalar)
        self.num_timesteps_input = num_timesteps_input
        self.num_edges = num_edges
        self.horizons = horizons
        self.time_units = time_units
        self.alpha = nn.Parameter(torch.ones(self.num_edges, 1).to(device), requires_grad=True)
        # self.alpha = nn.Parameter(0.5 * torch.ones(self.num_edges, 1), requires_grad=True)

        self.g = graph
        self.src, self.dst = graph.edges()
        self.src_idx = self.src.unique()
        self.dst_idx = self.dst.unique()
        self.num_nodes = self.g.number_of_nodes()
        self.g.edata['alpha'] = self.alpha

        self.prop_opt = torch.optim.Adam(self.transition_probability.parameters(), lr=0.001, weight_decay=1e-5)
        # self.prop_opt = torch.optim.Adam([
        #     {'params': self.transition_probability.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        #     {'params': [self.alpha], 'lr': 0.001, 'weight_decay': 1e-5}
        # ])
        # self.prop_opt = torch.optim.Adam([
        #     {'params': self.transition_probability.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        #     {'params': self.alpha_layer.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}
        # ])
        self.device = device
        self.to(self.device)


    @staticmethod
    def diffusion_sequence(F, n):
        indices = torch.arange(n.item()-1, -1, -1)
        result = F * torch.pow(1 - F, indices)
        # bias correction
        # result = result/(1 - torch.pow(1 - F, n.item())).detach()
        return result

    def edge_diffusion(self, edges):
        total_time_steps = self.num_timesteps_input
        F = 1/(1 + edges.data['alpha'].clamp(min=0, max=1) * edges.data['T']) # [num edges, batch_size]

        n = total_time_steps - edges.data['T_idx']
        sequences = list(map(self.diffusion_sequence, F.reshape([-1, ]).to('cpu'), n.reshape([-1, ]).to('cpu')))

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0).to(self.device)
        if padded_sequences.shape[1] < total_time_steps:
            pad_zero = torch.zeros(padded_sequences.shape[0], total_time_steps - padded_sequences.shape[1]).to(self.device)
            padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)
        edges_padded_sequences = padded_sequences.reshape(self.num_edges, F.shape[1], total_time_steps)

        return {'diffusion': edges_padded_sequences, "F": F}

    def message_func(self, edges):
        # 'message':[num_edges, batch_size, num_timesteps_input], 'upstream':[num_edges, batch_size, num_timesteps_input]
        pred_up2down = edges.data['e'] * torch.sum(edges.data['diffusion'] * edges.src['feature'], dim=2)
        # return {'message': edges.data['diffusion'], 'upstream': edges.src['feature'], 'atten': edges.data['e']}
        self.g.edata['message'] = pred_up2down
        return {'message': pred_up2down}

    def reduce_func(self, nodes):
        # alpha = func.softmax(nodes.mailbox['atten'], dim=1) # nodes.mailbox['atten']: [num_dst_nodes, num_neighbors, bc]
        # alpha = nodes.mailbox['atten']
        # pred_up2down = torch.sum(nodes.mailbox['message'] * nodes.mailbox['upstream'], dim=3) # [num_dst_nodes, num_neighbors, bc]
        pred_up2down = nodes.mailbox['message']
        # h = torch.sum(alpha * pred_up2down, dim=1) # [batch_size]
        h = torch.sum(pred_up2down, dim=1) # [batch_size]
        # h = torch.sum(pred_up2down, dim=1) # [batch_size]
        return {'pred': h}

    def forward(self, upstream_flows, downstream_flows):
        # upstream flows : [num of src, batch_size, num_timesteps_input]
        # downstream flows : [num of dst, batch_size, num_timesteps_input]
        # with torch.no_grad():
        #     upstream_flows = self.scalar.transform(upstream_flows)
        #     downstream_flows = self.scalar.transform(downstream_flows)
        # inpt = torch.cat((upstream_flows, downstream_flows), dim=2) # [num of edges, batch_size, 2 * num_timesteps_input]
        z = self.transition_probability(self.g.ndata['feature'])
        self.g.ndata['embedding'] = z   # for attention
        v = self.velocity_model(upstream_flows,
                                downstream_flows, self.alpha) # v : [batch_size, num of edges]
        self.g.edata['v'] = v.permute(1, 0) # [num_edges, batch_size]

        T = torch.divide(self.g.edata["distance"], v + 1e-5) # T : [batch_size, num of edges]
        self.g.edata['T'] = T.permute(1, 0) # [num_edges, batch_size]
        #round T to the time interval
        T_idx = torch.round(T/self.time_units)

        '''remark: < 0  = 0, >= num_timesteps_input = num_timesteps_input [0, num_timesteps_input-1]'''
        T_idx = T_idx.masked_fill(T_idx < 0, 0)
        T_idx = T_idx.masked_fill(T_idx >= self.num_timesteps_input, self.num_timesteps_input-1)
        # T_idx = 0 * torch.ones_like(T)

        # Diffusion
        # F = 1/(1 + self.g.edata['alpha'].T * T) # [num_edges, 1] *  [batch_size, num_edges]
        # for e in range(self.num_edges):
            # diffusion_coefficient.append(self.diffusion_batch(F[:, e],
            #                                                    self.num_timesteps_input,
            #                                                    bc,
            #                                                    T_idx[:, e]))

            # diffusion_coefficient.append(self.diffusion_sample(self.g.edata['alpha'][e],
            #                                                    self.num_timesteps_input,
            #                                                    bc, T[:, e],
            #                                                    T_idx[:, e]))
        # diffusion_coefficient = torch.stack(diffusion_coefficient, dim=0) # [num_edges, batch_size, num_timesteps_input]
        # self.g.edata["diffusion"] = diffusion_coefficient # [num_edges, batch_size, num_timesteps_input]

        self.g.edata["T"] = T.permute(1, 0) # [num_edges, batch_size]
        self.g.edata["T_idx"] = T_idx.permute(1, 0) # [num_edges, batch_size]
        self.g.apply_edges(self.edge_diffusion)
        self.g.apply_edges(self.transition_probability.edge_attention)

        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('pred') # [num_edges, batch_size]

    def pad_src_data(self, data):
        pred = []
        # pred.append(torch.mean(data[..., -4:], dim=-1))
        pred.append(torch.mean(data[..., -5:], dim=-1))
        data = torch.cat([data, pred[0].unsqueeze(-1)], dim=-1)
        for i in range(1, 6):  # 5
            # pred.append(torch.mean(data[..., -4+i:], dim=-1))
            pred.append(torch.mean(data[..., -5+i:], dim=-1))
            data = torch.cat([data, pred[i].unsqueeze(-1)], dim=-1)
        return data

    def inference(self, upstream_flows, downstream_flows):
        '''multi-step prediction'''
        # with torch.no_grad():
        #     upstream_flows = self.scalar.transform(upstream_flows)
        #     downstream_flows = self.scalar.transform(downstream_flows)

        multi_steps_pred = []
        pred = self.forward(upstream_flows, downstream_flows)
        multi_steps_pred.append(pred)
        first_step_pred = pred
        self.g.ndata['future_data'] = self.g.ndata['feature']
        self.g.ndata['future_data'] = self.pad_src_data(self.g.ndata['future_data'])
        # self.g.ndata['pred'] = pred.unsqueeze(-1)
        pred_horizons = self.horizons - 2
        T_idx = self.num_timesteps_input - self.g.edata['T_idx'].long() # T_idx: T_ab + 1
        self.T_idx = T_idx.reshape(-1,)
        self.multi_step_idx = 0
        # self.g.apply_edges(self.message_func) # pred up 2 down
        for i in range(pred_horizons):
            self.g.ndata['future_data'][self.dst_idx, :, self.num_timesteps_input + i] = pred[self.dst_idx]
            # self.g.apply_edges(self.multi_steps_prediction)
            # self.g.apply_nodes(self.reduce_multisteps)
            self.g.update_all(self.multi_steps_prediction, self.reduce_multisteps)
            # msg = self.g.mailbox('multi_steps_pred_edge')
            self.multi_step_idx += 1
            pred = self.g.ndata["multi_steps_pred"]
            multi_steps_pred.append(pred)
            # multi_steps_pred = torch.cat((multi_steps_pred, self.g.ndata['multi_steps_pred'].unsqueeze(dim=2)), dim=2)


        # pred = pred.clamp(min=0)
        # multi_steps_pred = torch.concat((pred.unsqueeze(-1), self.g.ndata['multi_steps_pred']), dim=2)
        multi_steps_pred = torch.stack(multi_steps_pred, dim=2)
        return first_step_pred, multi_steps_pred # [num_edges, batch_size]

    def multi_steps_prediction(self, edges):
        # self.g.apply_edges(self.message_func) # pred up 2 down
        # pred_horizons = self.horizons - 2
        # multi_step_pred = []
        F = edges.data['F']
        P = edges.data['e']
        # future_data = edges.src['feature']
        future_data = edges.src['future_data']
        # future_data = future_data.view(-1, future_data.shape[-1])
        '''pad src data with ma'''
        # future_data = self.pad_src_data(future_data)
        # future_data[self.src_dst_id, :, self.num_timesteps_input] = edges.src['pred'][self.src_dst_id, :, 0]
        future_data = future_data.view(-1, future_data.shape[-1])
        bc = edges.src['feature'].shape[1]

        # T_idx = self.num_timesteps_input - edges.data['T_idx'].long() # T_idx: T_ab + 1
        # T_idx = T_idx.reshape(-1,)

        ''' future_data[torch.arange(future_data.shape[0]), T_idx]: q_a(j - T + 1), edges.data["message"]: q_ab(j)
            step for j + 1
        '''
        if self.multi_step_idx != 0:
            multi_step_pred = (1 - F) * edges.data['multi_steps_pred_edge'] + \
                            P * F  * future_data[torch.arange(future_data.shape[0]), self.T_idx+self.multi_step_idx].view(self.num_edges, bc)
        else:
            multi_step_pred = (1 - F) * edges.data["message"] + \
                                    P * F  * future_data[torch.arange(future_data.shape[0]), self.T_idx].view(self.num_edges, bc)
        '''steps for > j + 2'''
        # for i in range(1, pred_horizons):
        #     multi_step_pred.append((1 - F) * multi_step_pred[i - 1] + \
        #     P * F * future_data[torch.arange(future_data.shape[0]), T_idx + i].view(self.num_edges, bc))

        # multi_step_pred = torch.stack(multi_step_pred, dim=2)
        self.g.edata['multi_steps_pred_edge'] = multi_step_pred

        return {'multi_steps_pred_edge': multi_step_pred}

    def reduce_multisteps(self, nodes):
        pred = torch.sum(nodes.mailbox['multi_steps_pred_edge'], dim=1)
        # pred = torch.cat((nodes.data['pred'].unsqueeze(-1), pred), dim=-1)
        return {'multi_steps_pred': pred}


if __name__ == '__main__': #network 3
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import dgl
    import random
    import pickle

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = 'sc_sensor'

    # out of distribution
    # dataset_name = "crossroad"
    dataset_name = "train_station"
    # dataset_name = "maze"
    if dataset_name == "crossroad":
        train_sc = ['sc_sensor/crossroad2']
        test_sc = ['sc_sensor/crossroad1', 'sc_sensor/crossroad11', 'sc_sensor/crossroad13']
    elif dataset_name == "train_station":
        train_sc = ['sc_sensor/train13']
        test_sc = ['sc_sensor/train2']
    elif dataset_name == "maze":
        train_sc = ['sc_sensor/maze17']
        # train_sc = ['sc_sensor/maze2']
        test_sc = ['sc_sensor/maze13', 'sc_sensor/maze4']

    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    if dataset_name == "maze":
        with open("./sc_sensor/maze/flow_data.pkl", "rb") as f:
            data_dict = pickle.load(f)
    else:
        df_dict = process_sensor_data(parent_dir, df_dict)
        data_dict = gen_data_dict(df_dict)
        data_dict = seperate_up_down(data_dict)

    '''Has to >= 2'''
    pred_horizon = 7 # 3, 5
    lags = 5
    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=lags, horizons=pred_horizon, shuffle=True)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=6,
    #                                                                              horizons=pred_horizon,
    #                                                                              portion=0.7,
    #                                                                              shuffle=True)

    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    # train_dataset = FlowDataset(np.concatenate([x_train, x_val], axis=0),
    #                             np.concatenate([y_train, y_val], axis=0), batch_size=16)
    train_dataset = FlowDataset(x_train,
                                y_train, batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # set seed
    torch.manual_seed(1)
    #normalization
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())


    if dataset_name == "crossroad":
        src = np.array([0, 0, 0, 3, 3, 3, 5, 5, 5, 6, 6, 6])
        dst = np.array([4, 2, 7, 1, 4, 7, 2, 7, 1, 2, 4, 1])
        src_dst = np.intersect1d(dst, src)  # the nodes that are both source and destination
        src_dst_id = np.where(np.isin(src, src_dst))[0]
        g = dgl.graph((src, dst))
        g.edata['distance'] = torch.FloatTensor([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]) # 50m

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

        src_dst = np.intersect1d(dst, src)
        src_dst_id = np.where(np.isin(src, src_dst))[0]
        g = dgl.graph((src, dst))
        g.edata['distance'] = torch.FloatTensor([40,40,28, # 3
                                                 40,50,32, # 4
                                                 40,50,32,
                                                 28,32,32,
                                                 24,24,41,41,35,
                                                 24,50,49,54,65,
                                                 24,50,65,54,49,
                                                 41,54,65,50,32,
                                                 25,47,50,
                                                 32,50,65,54,41,
                                                 25,32,25,
                                                 50,47,25,
                                                 32,47,47,
                                                 32,32,49,49,35])

    if dataset_name == "maze":
        src = np.array([0, 0, 0, 3, 3, 3, 5, 5, 5, 22, 22, 22,
                        23, 23, 23, 18, 18, 18, 20, 20, 20,
                        17, 17, 17, 16, 16, 16, 14, 14, 14,
                        11, 11, 11, 13, 13, 13, 10, 10, 10,
                        4, 4, 4, 9, 9, 9, 7, 7, 7])

        dst = np.array([2, 4, 23, 1, 4, 23, 2, 1, 23, 4, 2, 1,
                        19, 16, 21, 22, 16, 21, 22, 19, 16,
                        22, 19, 21, 12, 10, 15, 17, 10, 12,
                        12, 15, 17, 10, 17, 15, 5, 8, 6,
                        6, 8, 11, 6, 5, 11, 8, 11, 5])

        g = dgl.graph((src, dst))
        src_dst = np.intersect1d(dst, src)
        src_dst_id = np.where(np.isin(src, src_dst))[0]
        g.edata['distance'] = torch.FloatTensor([30, 50, 43, 30, 43, 50, 43,
                                                  50, 50, 50, 50, 43, 43, 50,
                                                  60, 43, 50, 43, 60, 43, 50,
                                                  50, 50, 50, 50, 50, 50, 50,
                                                  60, 43, 43, 60, 50, 43, 50,
                                                  43, 50, 43, 50, 43, 50, 50,
                                                  30, 50, 43, 30, 50, 43])



    # train
    model = Diffusion_Model(num_edges=len(src), num_timesteps_input=x_train.shape[1], graph=g, horizons=pred_horizon, device=device, scalar=None)
    model.src_dst_id = src_dst_id
    # if dataset_name == "crossroad":
    #     model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_model_network3_cross.pth"))
    # if dataset_name == "train_station":
    #     model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_model_network3.pth"))

    # optimizer = torch.optim.Adam([
    #     {'params': model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
    #     {'params': model.alpha_layer.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}
    # ])

    optimizer = torch.optim.Adam([
        {'params': model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        {'params': [model.alpha], 'lr': 0.001, 'weight_decay': 1e-5}
    ])
    # optimizer = torch.optim.Adam(model.velocity_model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer2 = torch.optim.Adam([model.alpha], lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    import time
    start = time.time()
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()

    # train
    for epoch in range(300):
        l = []
        for i, (x, y) in enumerate(train_dataloader):

            # if epoch <= 500:
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # loss = loss_fn(pred, y[:, 0, :])
            if epoch <= 100:
                pred = model(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
                loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0]) # [num_dst, batch_size], one-step prediction
            else:
                '''add multi steps loss'''
                _, multi_steps_pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst])
                loss = loss_fn(multi_steps_pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update transition probability
            if epoch % 2 == 0:
                # '''single step'''
                # if pred_horizon - 1 == 1:
                #     pred = model(g.ndata['feature'][src], g.ndata['feature'][dst])
                #     loss2 = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
                # else:
                #     '''add multi steps loss'''
                #     _, multi_steps_pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst])
                #     loss2 = loss_fn(multi_steps_pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
                if pred_horizon - 1 > 1 and epoch > 50:
                    '''add multi steps loss'''
                    _, multi_steps_pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst])
                    loss2 = loss_fn(multi_steps_pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
                else:
                    pred = model(g.ndata['feature'][src], g.ndata['feature'][dst])
                    loss2 = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])

                model.prop_opt.zero_grad()
                # optimizer.zero_grad()
                loss2.backward()
                # optimizer.step()
                model.prop_opt.step()
                # optimizer2.step()


            l.append(loss.item())

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))

    total_time = time.time() - start

    # for base model in online learning
    # if dataset_name == "crossroad":
    #     torch.save(model.state_dict(), f'./checkpoint/diffusion/diffusion_model_network4_cross_lags{lags}_hor{pred_horizon}.pth')
    # if dataset_name == "train_station":
    #     torch.save(model.state_dict(), f'./checkpoint/diffusion/diffusion_model_network4_train_lags{lags}_hor{pred_horizon}.pth')
    # if dataset_name == "maze":
    #     torch.save(model.state_dict(), f'./checkpoint/diffusion/diffusion_model_network4_maze_lags{lags}_hor{pred_horizon}.pth')

    # for offline analysis
    torch.save(model.state_dict(), f'./checkpoint/diffusion/offline_diffusion_model_network5_{dataset_name}_lags{lags}_hor{pred_horizon}.pth')

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=y_test.shape[0])
    test_dataloader = DataLoader(test_dataset, batch_size=y_test.shape[0])
    print('*************')

    test_loss = []
    train_loss = []
    multi_steps_train_loss = []
    multi_steps_test_loss = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred, multi_steps_pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            train_loss.append(loss.item())
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            # v = model.velocity_model(x_up, x_down)

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(multi_steps_pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_train_loss.append(multisteps_loss.item())

            print('Train Prediction: {}'.format(pred[dst_idx]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            print('Train Multi-Steps Loss: {}'.format(multisteps_loss.item()))
            # print('Train Velocity: {}'.format(v))
            # print("Probabilities: {}".format(model.g.ndata['alpha'][[0,3],...]))

            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred, multi_steps_pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            test_loss.append(loss.item())
            # v = model.velocity_model(x_up, x_down)

            # multi_steps_pred = torch.cat((pred.unsqueeze(-1), multi_steps_pred), dim=2)
            multisteps_loss = loss_fn(multi_steps_pred[dst_idx, :, :], g.ndata['label'][dst_idx, :, :])
            multi_steps_test_loss.append(multisteps_loss.item())

            print('Test Prediction: {}'.format(pred[dst_idx]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            print('Test Multi-Steps Loss: {}'.format(multisteps_loss.item()))
            # print('Test Velocity: {}'.format(v))
            # print("Probabilities: {}".format(model.g.ndata['alpha'][[0,3],...]))
            # print("upstream flows: {}".format(x_up[0, ...]))
            print('*************')

        print('Training Time: {}'.format(total_time))
        print('Total Train Loss: {}'.format(np.mean(train_loss)))
        print('Multi-Steps Train Loss: {}'.format(np.mean(multi_steps_train_loss)))
        print('Total Test Loss: {}'.format(np.mean(test_loss)))
        print('Multi-Steps Test Loss: {}'.format(np.mean(multi_steps_test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model))) # 1287
    print(model.alpha)
    # save checkpoint
    # torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_model_network1.pth')

# print model parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
