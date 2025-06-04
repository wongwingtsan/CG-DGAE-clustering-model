import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.FloatTensor(features)

def build_adjacency(coords, k=10):
    """Build adjacency matrix based on k-nearest neighbors."""
    from sklearn.neighbors import kneighbors_graph
    # Construct k-nearest neighbors graph
    A = kneighbors_graph(coords, k, mode='distance', include_self=True)
    # Make it symmetric
    A = 0.5 * (A + A.T)
    return A

def euclidean_dist(x, y):
    """Calculate the euclidean distance between two points."""
    return np.sqrt(np.sum((x - y) ** 2))

def calculate_graph_statistics(adj):
    """Calculate basic statistics of the graph."""
    n_nodes = adj.shape[0]
    n_edges = int(adj.sum() / 2)  # Divide by 2 for undirected graph
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    degree = adj.sum(axis=1)
    avg_degree = degree.mean()
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'min_degree': degree.min(),
        'max_degree': degree.max()
    } 