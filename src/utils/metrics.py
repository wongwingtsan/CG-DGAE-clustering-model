import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import torch

def clustering_metrics(embeddings, labels, pred_labels):
    """
    Calculate various clustering metrics
    
    Parameters:
    -----------
    embeddings : array-like of shape (n_samples, n_features)
        The learned node embeddings
    labels : array-like of shape (n_samples,)
        Ground truth labels (if available)
    pred_labels : array-like of shape (n_samples,)
        Predicted cluster labels
        
    Returns:
    --------
    dict : Dictionary containing various clustering metrics
    """
    metrics = {}
    
    # Silhouette score (-1 to 1, higher is better)
    metrics['silhouette'] = silhouette_score(embeddings, pred_labels)
    
    # Calinski-Harabasz Index (higher is better)
    metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, pred_labels)
    
    # Davies-Bouldin Index (lower is better)
    metrics['davies_bouldin'] = davies_bouldin_score(embeddings, pred_labels)
    
    # If ground truth labels are available
    if labels is not None:
        # Normalized Mutual Information (0 to 1, higher is better)
        metrics['nmi'] = normalized_mutual_info_score(labels, pred_labels)
        
        # Adjusted Rand Index (-1 to 1, higher is better)
        metrics['ari'] = adjusted_rand_score(labels, pred_labels)
    
    return metrics

def reconstruction_metrics(original, reconstructed):
    """
    Calculate reconstruction metrics for autoencoder
    
    Parameters:
    -----------
    original : torch.Tensor
        Original input features
    reconstructed : torch.Tensor
        Reconstructed features from autoencoder
        
    Returns:
    --------
    dict : Dictionary containing reconstruction metrics
    """
    # Convert to numpy if tensors
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)
    metrics['mse'] = mse
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mse)
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(original - reconstructed))
    
    # R-squared score
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot)
    
    return metrics

def modularity_score(adj_matrix, cluster_labels):
    """
    Calculate the modularity score for a graph clustering
    
    Parameters:
    -----------
    adj_matrix : scipy.sparse.csr_matrix or numpy.ndarray
        Adjacency matrix of the graph
    cluster_labels : array-like
        Cluster assignments for each node
        
    Returns:
    --------
    float : Modularity score (-1 to 1, higher is better)
    """
    if not isinstance(adj_matrix, np.ndarray):
        adj_matrix = adj_matrix.toarray()
    
    n_nodes = adj_matrix.shape[0]
    n_edges = np.sum(adj_matrix) / 2
    
    if n_edges == 0:
        return 0.0
    
    modularity = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if cluster_labels[i] == cluster_labels[j]:
                k_i = np.sum(adj_matrix[i])
                k_j = np.sum(adj_matrix[j])
                expected = k_i * k_j / (2 * n_edges)
                modularity += adj_matrix[i, j] - expected
                
    return modularity / (2 * n_edges) 