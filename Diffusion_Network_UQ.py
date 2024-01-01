# -*- coding: utf-8 -*-
# @Time    : 26/11/2023 15:43
# @Author  : mmai
# @FileName: Diffusion
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn.functional as func
from dgl.nn.functional import edge_softmax
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NegativeBinomialDistributionLoss(nn.Module):
    def __init__(self):
        super(NegativeBinomialDistributionLoss, self).__init__()
        pass
    def forward(self, mu, z, alpha): # alpha: predicted, mu: predicted count, z: ground truth
        # Calculate the negative log likelihood of the negative binomial distribution
        nll_loss = -(torch.lgamma(z + 1/(alpha+1e-6)) - torch.lgamma(1/(alpha+1e-6)) - torch.lgamma(z + 1) - torch.lgamma(1/(alpha+1e-6)) + \
                    z * (torch.log(alpha+1e-6) + torch.log(mu+1e-6) - torch.log(1 + alpha * mu)) - 1/(alpha+1e-6) * torch.log(1 + alpha* mu))

        # Take the mean over the batch
        nll_loss = torch.mean(nll_loss)

        return nll_loss

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

        self.linear3 = torch.nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.x_scalar = scalar

    def forward(self, upstream, downstream):
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

        x = torch.cat((x_up, x_down), dim=2)
        out = self.linear3(x) # [num of edges, batch_size, 1]
        # out = self.relu(out)
        out = torch.log(1 + torch.exp(out))
        out = out.squeeze(2).transpose(1, 0) # [batch_size, num of edges]

        return out

    def inference(self, upstream, downstream):
        with torch.no_grad():
            up_sum = torch.sum(upstream, dim=2)  # [node, batch_size]
            up_idx = (up_sum == 0).nonzero(as_tuple=True)
            out = self.forward(upstream, downstream)
            out[up_idx[1], up_idx[0]] = 0

        return out

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
        # self.ln = nn.LayerNorm(self._hidden_size)
        self.ln2 = nn.LayerNorm(2 * self._hidden_size + 1)
        # self.dropout = nn.Dropout(0.5)
        self.device = device
        self.to(self.device)

    def edge_attention(self, edges):
        # up2down = torch.sum(edges.src['feature'] * edges.data['diffusion'], dim=2).unsqueeze(-1) # [num_edges, batch_size]
        # up2down = edge_softmax(self.g, up2down, norm_by='src') # [num_edges, batch_size]
        # F = torch.sum(edges.data["diffusion"], dim=2, keepdim=True) # [num_edges, batch_size, num_timesteps_input]
        v = edges.data['v'].unsqueeze(-1) # [num_edges, batch_size, 1]
        # up = self.fc_up(torch.sigmoid(self.dropout(edges.src['embedding']))) # [num_edges, batch_size, 1]
        # down = self.fc_down(torch.sigmoid(self.dropout(edges.dst['embedding'])))
        # down = self.dropout(down)
        # z = torch.cat([edges.src['embedding'], edges.dst['embedding'], F.detach()], dim=2)  # [num_edges, bc, 2 * 16]
        z = torch.cat([edges.src['embedding'], edges.dst['embedding'], v.detach()], dim=2)  # [num_edges, bc, 2 * hid + 1]
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
        # z = self.ln(z)
        return z




class Diffusion_Model_UQ(torch.nn.Module):
    def __init__(self, num_edges, num_timesteps_input, graph, scalar, time_units=10):
        """

        :param num_nodes: number of ancestor nodes
        :param num_timesteps_input: time steps of input
        """
        super(Diffusion_Model_UQ, self).__init__()
        self.velocity_model = Velocity_Model_NAM(num_timesteps_input, hidden_units=32, scalar=scalar)
        # self.velocity_model = Velocity_Model(num_timesteps_input, hidden_units=16, out_units=8, scalar=scalar)
        self.transition_probability = Probabilistic_Model(graph, num_edges, num_timesteps_input, hidden_units=64, scalar=scalar)
        self.scalar = scalar

        self.num_timesteps_input = num_timesteps_input
        self.num_edges = num_edges
        self.time_units = time_units
        self.alpha = nn.Parameter(torch.ones(self.num_edges, 1), requires_grad=True)
        self.sigma_fc = nn.Linear(64, 1)
        self.ln = nn.LayerNorm(64)

        self.g = graph
        self.g.edata['alpha'] = self.alpha

        self.prop_opt = torch.optim.Adam(self.transition_probability.parameters(), lr=0.001, weight_decay=1e-5)
        # self.prop_opt = torch.optim.Adam([
        #     {'params': self.transition_probability.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        #     {'params': [self.alpha], 'lr': 0.001, 'weight_decay': 1e-5}
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
        F = 1/(1 + edges.data['alpha'] * edges.data['T']) # [num edges, batch_size]
        edges_padded_sequences = []
        for e in range(self.num_edges):
            sequences = []
            for i in range(F.size(1)): # batch_size
                F_sequence = self.diffusion_sequence(F[e, i], torch.max((total_time_steps - edges.data['T_idx'][e, i]), torch.FloatTensor([1])))
                sequences.append(F_sequence)

            padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0) # [batch_size, num_timesteps_input]
            if padded_sequences.shape[1] < total_time_steps:
                pad_zero = torch.zeros(F.size(1), total_time_steps - padded_sequences.shape[1]).to(self.device)
                padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)

            edges_padded_sequences.append(padded_sequences)

        return {'diffusion': torch.stack(edges_padded_sequences, dim=0), "F": F}

    def message_func(self, edges):
        # 'message':[num_edges, batch_size, num_timesteps_input], 'upstream':[num_edges, batch_size, num_timesteps_input]
        return {'message': edges.data['diffusion'], 'upstream': edges.src['feature'], 'atten': edges.data['e'], 'embedding': edges.src['embedding']}
    #
    def reduce_func(self, nodes):
        # alpha = func.softmax(nodes.mailbox['atten'], dim=1) # nodes.mailbox['atten']: [num_dst_nodes, num_neighbors, bc]
        alpha = nodes.mailbox['atten']
        pred_up2down = torch.sum(nodes.mailbox['message'] * nodes.mailbox['upstream'], dim=3) # [num_dst_nodes, num_neighbors, bc]
        h = torch.sum(alpha * pred_up2down, dim=1) # [batch_size]

        embed = self.ln(nodes.mailbox['embedding'])
        sigma = torch.log(1 + torch.exp(torch.mean(self.sigma_fc(embed).squeeze(), dim=1))) # [batch_size]
        # h = torch.log(1 + torch.exp(h)) # [batch_size]
        return {'pred': h, 'sigma': sigma}

    def forward(self, upstream_flows, downstream_flows):
        # upstream flows : [num of src, batch_size, num_timesteps_input]
        # downstream flows : [num of dst, batch_size, num_timesteps_input]
        # with torch.no_grad():
        #     upstream_flows = self.scalar.transform(upstream_flows)
        #     downstream_flows = self.scalar.transform(downstream_flows)
        z = self.transition_probability(self.g.ndata['feature'])
        self.g.ndata['embedding'] = z   # for attention
        v = self.velocity_model(upstream_flows, downstream_flows) # v : [batch_size, num of edges]
        self.g.edata['v'] = v.permute(1, 0) # [num_edges, batch_size]

        T = torch.divide(self.g.edata["distance"], v + 1e-5) # T : [batch_size, num of edges]
        #round T to the time interval
        T_idx = torch.round(T/self.time_units)

        #remark: < 0  = 0, > num_timesteps_input = num_timesteps_input-1
        T_idx = T_idx.masked_fill(T_idx < 0, 0)
        T_idx = T_idx.masked_fill(T_idx > self.num_timesteps_input, self.num_timesteps_input-1)

        # T_idx = 0 * torch.ones_like(T)

        self.g.edata["T"] = T.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.edata["T_idx"] = T_idx.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.apply_edges(self.edge_diffusion)
        self.g.apply_edges(self.transition_probability.edge_attention)

        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('pred') # [num_edges, batch_size]

    def inference(self, upstream_flows, downstream_flows):
        # with torch.no_grad():
        #     upstream_flows = self.scalar.transform(upstream_flows)
        #     downstream_flows = self.scalar.transform(downstream_flows)
        pred = self.forward(upstream_flows, downstream_flows)

        return pred.clamp(min=0) # [num_edges, batch_size]


if __name__ == '__main__': #network 3
    from torch.utils.data import DataLoader
    from lib.dataloader import FlowDataset
    from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
    from lib.utils import generating_ood_dataset, seperate_up_down, generating_insample_dataset, get_trainable_params_size
    import dgl
    import random

    random.seed(1)
    df_dict = {}
    # Define path to parent directory containing subdirectories with CSV files
    parent_dir = 'sc_sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)

    # out of distribution
    # dataset_name = "crossroad"
    dataset_name = "train_station"
    # train_sc = ['sc_sensor/crossroad2', 'sc_sensor/crossroad9', 'sc_sensor/crossroad10', 'sc_sensor/crossroad11']
    # test_sc = ['sc_sensor/crossroad3']
    train_sc = ['sc_sensor/train6']
    test_sc = ['sc_sensor/train7']
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

    # train
    model = Diffusion_Model_UQ(num_edges=len(src), num_timesteps_input=x_train.shape[1], graph=g, scalar=None)
    # if dataset_name == "crossroad":
    #     model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_uq_cross.pth"))
    # if dataset_name == "train_station":
    #     model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_uq.pth"))

    # init_velocity_model(x_train, x_val, g, model.velocity_model)

    optimizer = torch.optim.Adam([
        {'params': model.velocity_model.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
        {'params': [model.alpha], 'lr': 0.001, 'weight_decay': 1e-5}
    ])
    test_loss_fn = torch.nn.MSELoss()
    # NLL loss
    # loss_fn = torch.nn.GaussianNLLLoss()
    loss_fn = NegativeBinomialDistributionLoss()

    import time
    start = time.time()
    src, dst = g.edges()
    src_idx = src.unique()
    dst_idx = dst.unique()

    # train
    for epoch in range(500):
        l = []
        mse = []
        for i, (x, y) in enumerate(train_dataloader):

            # if epoch <= 500:
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]

            pred = model(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            # assert torch.isnan(pred).sum() == 0, print(f"epoch: {epoch}, pred: {pred}")
            # var = g.ndata['sigma']**2
            # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0], var[dst_idx]) # gaussian loss
            # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0]) # poissson loss

            alpha = g.ndata['sigma']
            loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0], alpha[dst_idx]) # negative binomial loss
            mse_loss = test_loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            optimizer.zero_grad()
            loss.backward()
            # parameters_to_clip = list(model.velocity_model.parameters()) + [model.alpha]
            # clip_grad_norm_(parameters_to_clip, max_norm=1, norm_type=2)
            optimizer.step()

            # update transition probability
            if epoch % 2 == 0:
                pred = model(g.ndata['feature'][src], g.ndata['feature'][dst])
                # var = g.ndata['sigma']**2
                # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0], var[dst_idx])
                # loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0]) # poissson loss
                alpha = g.ndata['sigma']
                loss = loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0], alpha[dst_idx]) # negative binomial loss
                model.prop_opt.zero_grad()
                loss.backward()
                # clip_grad_norm_(model.transition_probability.parameters(), max_norm=1, norm_type=2)
                model.prop_opt.step()

            l.append(loss.item())
            mse.append(mse_loss.item())

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}, MSE: {}'.format(epoch, np.mean(l), np.mean(mse)))
            if dataset_name == "crossroad":
                torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_uq_cross.pth')
            if dataset_name == "train_station":
                torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_uq.pth')

    print('Training Time: {}'.format(time.time() - start))

    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    print('*************')

    test_loss = []
    train_loss = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = test_loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            train_loss.append(loss.item())
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            v = model.velocity_model(x_up, x_down)
            print('Train Prediction: {}'.format(pred[dst_idx]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            print('Train Velocity: {}'.format(v))
            # print("Probabilities: {}".format(model.g.ndata['alpha'][[0,3],...]))

            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = test_loss_fn(pred[dst_idx], g.ndata['label'][dst_idx,:, 0])
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            test_loss.append(loss.item())
            v = model.velocity_model(x_up, x_down)
            print('Test Prediction: {}'.format(pred[dst_idx]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst_idx,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            print('Test Velocity: {}'.format(v))
            # print("Probabilities: {}".format(model.g.ndata['alpha'][[0,3],...]))
            # print("upstream flows: {}".format(x_up[0, ...]))
            print('*************')

        print('Total Train Loss: {}'.format(np.mean(train_loss)))
        print('Total Test Loss: {}'.format(np.mean(test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model))) # 1287
    # save checkpoint
    # torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_model_network1.pth')

# print model parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
