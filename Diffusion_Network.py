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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, num_timesteps_input, scalar):
        super(ResidualBlock, self).__init__()
        # MLP
        self.linear11 = torch.nn.Linear(num_timesteps_input, 32)
        self.linear12 = torch.nn.Linear(32, 1)

        self.linear21 = torch.nn.Linear(num_timesteps_input, 32)
        self.linear22 = torch.nn.Linear(32, 1)

        self.linear3 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
        self.x_scalar = scalar

    def forward(self, upstream, downstream):
        # upstream : [batch_size, num_timesteps_input], only from one node
        # downstream : [batch_size, num_timesteps_input]
        with torch.no_grad():
            upstream = self.x_scalar.transform(upstream)
            downstream = self.x_scalar.transform(downstream)
        x_up = self.linear11(upstream)
        x_up = self.relu(x_up)
        x_up = self.linear12(x_up)
        x_up = torch.sigmoid(x_up)

        x_down = self.linear21(downstream)
        x_down = self.relu(x_down)
        x_down = self.linear22(x_down)
        x_down = torch.sigmoid(x_down)

        x = torch.cat((x_up, x_down), dim=1)
        out = self.linear3(x)
        # out = self.relu(out)
        return out

class Velocity_Model_NAM(nn.Module):
    def __init__(self, num_timesteps_input, scalar):
        super(Velocity_Model_NAM, self).__init__()
        # MLP
        self._hidden_size = 32
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
        out = out.squeeze(2).transpose(1, 0) # [batch_size, num of edges]

        return out

def init_velocity_model(x_train, x_val, g, model):
    # pretrain the model on a sythentic dataset with speed label
    # x_train : [batch_size, num_timesteps_input, num_nodes]
    # construct the label for x_train
    # y_train : [batch_size, 1]
    x_train = torch.FloatTensor(x_train)
    x_val = torch.FloatTensor(x_val)
    x = torch.cat((x_train, x_val), dim=0)
    # upstream_flows = x[:, :, :-1].reshape(-1, self.num_timesteps_input, self.num_nodes)
    # downstream_flows = x[:, :, -1].reshape(-1, self.num_timesteps_input, 1).repeat(1, 1, self.num_nodes)
    # velocity label coming from a gaussian distribution with mean 1.5 and std 0.01
    g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
    src, dst = g.edges()
    y = torch.normal(mean=1.5, std=0.1, size=[x.shape[0], len(g.edges())])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for iter in range(50):
        upstream = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
        downstream = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
        pred = model(upstream, downstream) # [batch_size, num of edges]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Pretrain Iteration: {}, Loss: {}'.format(iter, loss.item()))

class Diffusion_Model(torch.nn.Module):
    def __init__(self, num_edges, num_timesteps_input, graph, scalar, time_units=10):
        """

        :param num_nodes: number of ancestor nodes
        :param num_timesteps_input: time steps of input
        """
        super(Diffusion_Model, self).__init__()
        self.velocity_model = Velocity_Model_NAM(num_timesteps_input, scalar=scalar)
        # self.residual_block = ResidualBlock(num_timesteps_input, scalar=scalar)
        self.num_timesteps_input = num_timesteps_input
        self.num_edges = num_edges
        self.time_units = time_units
        self.alpha = nn.Parameter(torch.ones(self.num_edges, 1), requires_grad=True)

        self.g = graph
        self.g.edata['alpha'] = self.alpha
        self.fc = nn.Linear(self.num_timesteps_input, 32, bias=False)

        self.attn_fc = nn.Linear(2 * 32, 1, bias=False)
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

        return {'diffusion': torch.stack(edges_padded_sequences, dim=0)}

    def diffusion_sample(self, alpha, total_time_steps, bc, T,  T_idx):
        # upstream flow : [batch_size, num_timesteps_input], flow from one node
        # T : [batch_size, 1], travel time
        # T_idx : [batch_size, 1], travel time index

        #diffustion coefficient
        F = 1/(1 + alpha*T) # [batch_size, 1]
        # F = 1/(1 + T) # [batch_size, 1]
        # total_time_steps = upstream_flow.shape[1]
        # construct the diffusion matrix according to T_idx
        sequences = []

        for i in range(F.size(0)): # batch_size
            F_sequence = self.diffusion_sequence(F[i], torch.max((total_time_steps - T_idx[i]), torch.FloatTensor([1])))
            sequences.append(F_sequence)

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0) # [batch_size, num_timesteps_input]
        if padded_sequences.shape[1] < total_time_steps:
            pad_zero = torch.zeros(bc, total_time_steps - padded_sequences.shape[1]).to(self.device)
            padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)

        # prediction = padded_sequences * upstream_flow # [batch_size, num_timesteps_input]

        # prediction = torch.sum(prediction, dim=1, keepdim=True) # [batch_size]

        return padded_sequences # [batch_size, num_timesteps_input]

    def diffusion_batch(self, F, total_time_steps, bc, T_idx):
        sequences = []
        for i in range(F.size(0)):
            F_sequence = self.diffusion_sequence(F[i], torch.max((total_time_steps - T_idx[i]), torch.FloatTensor([1])))
            sequences.append(F_sequence)

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0) # [batch_size, num_timesteps_input]
        if padded_sequences.shape[1] < total_time_steps:
            pad_zero = torch.zeros(bc, total_time_steps - padded_sequences.shape[1]).to(self.device)
            padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)

        return padded_sequences # [batch_size, num_timesteps_input]

    def edge_attention(self, edges):
        z = torch.cat([edges.src['embedding'], edges.dst['embedding']], dim=2)  # [num_edges, bc, 2 * 16]
        a = self.attn_fc(z).squeeze(dim=2)  # [num_edges, bc, 1] --> [num_edges, bc]
        return {'e': func.leaky_relu(a)}
    def message_func(self, edges):
        # 'message':[num_edges, batch_size, num_timesteps_input], 'upstream':[num_edges, batch_size, num_timesteps_input]
        return {'message': edges.data['diffusion'], 'upstream': edges.src['feature'], 'atten': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = func.softmax(nodes.mailbox['atten'], dim=1) # nodes.mailbox['atten']: [num_nodes, num_neighbors, bc]
        scale_upstream = alpha.unsqueeze(-1) * nodes.mailbox['upstream']
        pred_up2down = torch.sum(nodes.mailbox['message'] * scale_upstream, dim=3) # [num_nodes, num_neighbors, bc]
        # h = torch.sum(alpha * pred_up2down, dim=1) # [batch_size]
        h = torch.sum(pred_up2down, dim=1) # [batch_size]
        return {'pred': h}

    def forward(self, upstream_flows, downstream_flows):
        # upstream flows : [num of src, batch_size, num_timesteps_input]
        # downstream flows : [num of dst, batch_size, num_timesteps_input]
        bc = upstream_flows.shape[1]
        diffusion_coefficient = []

        z = self.fc(self.g.ndata['feature'])
        self.g.ndata['embedding'] = z   # for attention
        self.g.apply_edges(self.edge_attention)
        v = self.velocity_model(upstream_flows,
                                downstream_flows) # v : [batch_size, num of edges]

        T = torch.divide(self.g.edata["distance"], v + 1e-5) # T : [batch_size, num of edges]
        #round T to the time interval
        T_idx = torch.round(T/self.time_units)

        #remark: < 0  = 0, > num_timesteps_input = num_timesteps_input-1
        T_idx = T_idx.masked_fill(T_idx < 0, 0)
        T_idx = T_idx.masked_fill(T_idx > self.num_timesteps_input, self.num_timesteps_input-1)
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

        self.g.edata["T"] = T.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.edata["T_idx"] = T_idx.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.apply_edges(self.edge_diffusion)


        self.g.update_all(self.message_func, self.reduce_func)

        # residual connection
        # oupt = oupt + self.residual_block(upstream_flows[:, :, n], downstream_flows[:, :, n])

        # TODO: sum the predictions from all the nodes with transition probability
        return self.g.ndata.pop('pred') # [num_edges, batch_size]

    def inference(self, upstream_flows, downstream_flows):
        # upstream flows : [num of src, batch_size, num_timesteps_input]
        # downstream flows : [num of dst, batch_size, num_timesteps_input]
        z = self.fc(self.g.ndata['feature'])
        self.g.ndata['embedding'] = z   # for attention
        self.g.apply_edges(self.edge_attention)
        v = self.velocity_model(upstream_flows,
                                downstream_flows).clamp(min=0) # v : [batch_size, num of edges]

        T = torch.divide(self.g.edata["distance"], v + 1e-5) # T : [batch_size, num of edges]
        #round T to the time interval
        T_idx = torch.round(T/self.time_units)

        #remark: < 0  = 0, > num_timesteps_input = num_timesteps_input-1
        T_idx = T_idx.masked_fill(T_idx < 0, 0)
        T_idx = T_idx.masked_fill(T_idx > self.num_timesteps_input, self.num_timesteps_input-1)

        self.g.edata["T"] = T.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.edata["T_idx"] = T_idx.permute(1, 0) # [num_edges, batch_size, num_timesteps_input]
        self.g.apply_edges(self.edge_diffusion)


        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('pred') # [num_edges, batch_size]


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
    parent_dir = './sc sensor'
    # adding to the df_dict
    # Loop through each subdirectory in the parent directory
    df_dict = process_sensor_data(parent_dir, df_dict)

    data_dict = gen_data_dict(df_dict)

    # in distributino
    # x_train, y_train, x_val, y_val, x_test, y_test = generate_insample_dataset_ver2(data_dict)

    # out of distribution
    train_sc = ['./sc sensor/crossroad4', './sc sensor/crossroad2', './sc sensor/crossroad3', './sc sensor/crossroad1']
    test_sc = ['./sc sensor/crossroad5']
    # for sc in data_dict.keys():
    #     if sc not in train_sc:
    #         test_sc.append(sc)

    #seperate upstream and downstream
    data_dict = seperate_up_down(data_dict)

    # data_dict['./sc sensor/crossroad4'] = data_dict['./sc sensor/crossroad4'][..., :4]
    # for k in data_dict.keys():  # debug
    #     data_dict[k] = data_dict[k][:,[0,3]]

    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(data_dict, train_sc, test_sc, lags=5)
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
                                                                                 # lags=5,
                                                                                 # shuffle=False)

    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(np.concatenate([x_train, x_val], axis=0),
                                np.concatenate([y_train, y_val], axis=0), batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    # set seed
    torch.manual_seed(1)
    #normalization
    x_scalar = StandardScaler(mean=np.concatenate([x_train, x_val]).mean(),
                              std=np.concatenate([x_train, x_val]).std())

    #processing graph data
    # src = np.array([0])
    # dst = np.array([1])

    # src = np.array([0, 2])
    # dst = np.array([3, 1])

    src = np.array([0, 0, 0, 3, 3, 3, 5, 5, 5, 6, 6, 6])
    dst = np.array([4, 2, 7, 1, 4, 7, 2, 7, 1, 2, 4, 1])

    g = dgl.graph((src, dst))
    g.edata['distance'] = torch.FloatTensor([43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]) # 50m
    # g.edata['distance'] = torch.FloatTensor([50, 50]) # 50m

    # train
    model = Diffusion_Model(num_edges=len(src), num_timesteps_input=x_train.shape[1], graph=g, scalar=x_scalar)
    model.load_state_dict(torch.load("./checkpoint/diffusion/diffusion_model_network1.pth"))
    # init_velocity_model(x_train, x_val, g, model.velocity_model)
    model.velocity_model.requires_grad_(True)
    model.alpha.requires_grad_(True)
    model.fc.requires_grad_(False)
    model.attn_fc.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    import time
    start = time.time()
    src, dst = g.edges()
    # for epoch in range(1000):
    #     l = []
    #     for i, (x, y) in enumerate(train_dataloader):
    #         if epoch % 10 == 0:
    #             model.fc.requires_grad_(True)
    #             model.attn_fc.requires_grad_(True)
    #             model.velocity_model.requires_grad_(False)
    #
    #         g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
    #         g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
    #
    #         pred = model(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
    #         # loss = loss_fn(pred, y[:, 0, :])
    #         loss = loss_fn(pred[dst], g.ndata['label'][dst,:, 0]) # [num_dst, batch_size], one-step prediction
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         l.append(loss.item())
    #         # early stopping
    #
    #         model.fc.requires_grad_(False)
    #         model.attn_fc.requires_grad_(False)
    #         model.velocity_model.requires_grad_(True)
    #
    #     if epoch % 100 == 0:
    #         print('Epoch: {}, Loss: {}'.format(epoch, np.mean(l)))
    #     torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_model_network1.pth')
    #
    # print('Training Time: {}'.format(time.time() - start))
    # test
    test_dataset = FlowDataset(x_test, y_test, batch_size=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    print('*************')

    test_loss = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = loss_fn(pred[dst], g.ndata['label'][dst,:, 0])
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            v = model.velocity_model(x_up, x_down)
            print('Train Prediction: {}'.format(pred[dst]))
            print('Train Ground Truth: {}'.format(g.ndata['label'][dst,:, 0]))
            print('Train Loss: {}'.format(loss.item()))
            print('Train Velocity: {}'.format(v))
            print('*************')

        for i, (x, y) in enumerate(test_dataloader):
            g.ndata['feature'] = x.permute(2, 0, 1) # [node, batch_size, num_timesteps_input]
            g.ndata['label'] = y.permute(2, 0, 1) # [node, batch_size, pred_horizon]
            # x_up = x[:, :, 0].reshape(-1, num_input_timesteps, num_nodes)
            # x_down = x[:, :, -1].reshape(-1, num_input_timesteps, 1).repeat(1, 1, num_nodes)
            pred = model.inference(g.ndata['feature'][src], g.ndata['feature'][dst]) # [num_dst, batch_size]
            loss = loss_fn(pred[dst], g.ndata['label'][dst,:, 0])
            x_up = g.ndata['feature'][src] # [num of src, batch_size, num_timesteps_input], num of src = num of dst = num of edges
            x_down = g.ndata['feature'][dst] # [num of dst, batch_size, num_timesteps_input]
            test_loss.append(loss.item())
            v = model.velocity_model(x_up, x_down)
            print('Test Prediction: {}'.format(pred[dst]))
            print('Test Ground Truth: {}'.format(g.ndata['label'][dst,:, 0]))
            print('Test Loss: {}'.format(loss.item()))
            print('Test Velocity: {}'.format(v))
            print('*************')

        print('Total Test Loss: {}'.format(np.mean(test_loss)))

    print('Total Trainable Parameters: {}'.format(get_trainable_params_size(model))) # 1287
    # save checkpoint
    torch.save(model.state_dict(), './checkpoint/diffusion/diffusion_model_network1.pth')

# print model parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
