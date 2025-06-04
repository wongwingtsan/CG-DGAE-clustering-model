import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math

class GATLayer(nn.Module):
    """
    Custom GAT layer for edge feature processing
    Source: Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)
        
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

class GAT_encod(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT_encod, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        return z

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                    out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        batch_size = X.shape[0]
        num_node = X.shape[1]
        input_size = X.size(2)
        
        supports = [A_q, A_h]
        x0 = X.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)
        x += self.bias

        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class ConvBiLSTM(nn.Module):
    def __init__(self):
        super(ConvBiLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=288, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=144, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)
        return x

class DGCN_AE(nn.Module):
    """
    Combined model with ConvLSTM, GAT-based GAE, and DGCN-based GAE
    """
    def __init__(self, h, z, k, num_features, hidden_size, embedding_size, alpha):
        super(DGCN_AE, self).__init__()
        self.time_dimension = h
        self.hidden_dimension = z
        self.order = k

        # DGCN components
        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.decoder = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='linear')

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=z, num_heads=8)

        # GAT encoder
        self.GAT_encod = GAT_encod(num_features, hidden_size, embedding_size, alpha)

        # ConvLSTM
        self.ConvBiLSTM = ConvBiLSTM()

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def forward(self, X, A_q, A_h, adj, M):
        # Process through ConvBiLSTM
        X = self.ConvBiLSTM(X)
        
        # DGCN processing
        X_s = X.permute(0, 2, 1)
        X_s1 = self.GNN1(X_s, A_q, A_h)
        X_em = self.GNN2(X_s1, A_q, A_h) + X_s1

        # GAT processing
        Edge_em = []
        A_pred = []
        for i in range(X.shape[0]):
            z_ = self.GAT_encod(torch.transpose(X[i], 0, 1), adj, M)
            A_p = self.dot_product_decode(z_)
            Edge_em.append(z_)
            A_pred.append(A_p)

        Edge_em = torch.stack(Edge_em)
        A_pred = A_pred[0]

        # Cross attention between node and edge embeddings
        X_em, _ = self.cross_attn(X_em, X_em, Edge_em)

        # Decode
        X_s2 = self.decoder(X_em, A_q, A_h)
        X_res = X_s2.permute(0, 2, 1)

        return X_res, X_em, Edge_em, A_pred

def calculate_random_walk_matrix(adj_mx):
    """
    Calculate random walk matrix for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray() 