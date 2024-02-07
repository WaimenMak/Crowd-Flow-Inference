# -*- coding: utf-8 -*-
# @Time    : 07/12/2023 21:41
# @Author  : mmai
# @FileName: debug
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z' : edges.src['z'], 'e' : edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def forward(self, h):
        # 公式 (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

import time
import numpy as np

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

import dgl
import torch
from dgl.data import CitationGraphDataset

def load_cora_data():
    # Load the Cora dataset
    data = CitationGraphDataset('cora')

    # Extract graph structure and features
    g = data[0]

    # Convert features and labels to PyTorch tensors
    features = torch.FloatTensor(g.ndata['feat'])
    labels = torch.LongTensor(g.ndata['label'])

    # Create a mask for training nodes
    mask = torch.BoolTensor(g.ndata['train_mask'])

    # Return the graph, features, labels, and mask
    return g, features, labels, mask

# g, features, labels, mask = load_cora_data()
#
# # 创建模型
# net = GAT(g,
#           in_dim=features.size()[1],
#           hidden_dim=8,
#           out_dim=7,
#           num_heads=8)
# print(net)
#
# # 创建优化器
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#
# # 主流程
# dur = []
# for epoch in range(30):
#     if epoch >=3:
#         t0 = time.time()
#
#     logits = net(features)
#     logp = F.log_softmax(logits, 1)
#     loss = F.nll_loss(logp[mask], labels[mask])
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if epoch >=3:
#         dur.append(time.time() - t0)
#
#     print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
#             epoch, loss.item(), np.mean(dur)))


def diffusion_sequence(F, n):
    indices = torch.arange(n.item()-1, -1, -1)
    result = F * torch.pow(1 - F, indices)
    # bias correction
    # result = result/(1 - torch.pow(1 - F, n.item())).detach()
    return result
if __name__ == '__main__':
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from multiprocessing import Pool
    # def diffusion_sequence(F, n):
    #     indices = torch.arange(n.item()-1, -1, -1)
    #     result = F * torch.pow(1 - F, indices)
    #     # bias correction
    #     # result = result/(1 - torch.pow(1 - F, n.item())).detach()
    #     return result
    total_time_steps = 6
    T = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])  # shape (bc, num edges)
    alpha = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]) # shape (num edges)
    F = 1/(1 + alpha * T)   # shape (bc, num edges)
    # f = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    n = torch.tensor([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]])
    with Pool() as p:
        sequences = p.starmap(diffusion_sequence, zip(F.reshape([-1, ]), n.reshape([-1, ])))

    p.close()
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    if padded_sequences.shape[1] < total_time_steps:
        pad_zero = torch.zeros(padded_sequences.shape[0], total_time_steps - padded_sequences.shape[1])
        padded_sequences = torch.cat((padded_sequences, pad_zero), dim=1)
    print(padded_sequences.reshape([T.shape[0], T.shape[1], total_time_steps]))