import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx
import torch

def plot_embeddings(embeddings, labels, title="Node Embeddings Visualization"):
    """
    Plot node embeddings using t-SNE
    
    Parameters:
    -----------
    embeddings : array-like of shape (n_samples, n_features)
        Node embeddings
    labels : array-like of shape (n_samples,)
        Cluster assignments
    title : str
        Plot title
    """
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

def plot_graph_clusters(adj_matrix, labels, coords=None, title="Graph Clustering Visualization"):
    """
    Visualize graph clustering results
    
    Parameters:
    -----------
    adj_matrix : array-like or sparse matrix
        Adjacency matrix of the graph
    labels : array-like
        Cluster assignments
    coords : array-like, optional
        Node coordinates for plotting
    title : str
        Plot title
    """
    # Convert adjacency matrix to networkx graph
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    
    G = nx.from_numpy_array(adj_matrix)
    
    plt.figure(figsize=(12, 8))
    
    # Set node positions
    if coords is not None:
        pos = {i: coords[i] for i in range(len(coords))}
    else:
        pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=labels, 
                          cmap='tab20', node_size=100, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_reconstruction_error(original, reconstructed, n_samples=5):
    """
    Plot original vs reconstructed time series
    
    Parameters:
    -----------
    original : array-like
        Original time series data
    reconstructed : array-like
        Reconstructed time series data
    n_samples : int
        Number of random samples to plot
    """
    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Randomly select samples
    indices = np.random.choice(len(original), size=n_samples, replace=False)
    
    plt.figure(figsize=(15, 3*n_samples))
    for i, idx in enumerate(indices):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(original[idx], label='Original', alpha=0.7)
        plt.plot(reconstructed[idx], label='Reconstructed', alpha=0.7)
        plt.title(f'Sample {idx}')
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cluster_statistics(labels, features):
    """
    Plot statistics about the clusters
    
    Parameters:
    -----------
    labels : array-like
        Cluster assignments
    features : array-like
        Node features
    """
    n_clusters = len(np.unique(labels))
    
    # Plot cluster sizes
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    cluster_sizes = np.bincount(labels)
    plt.bar(range(n_clusters), cluster_sizes)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of nodes')
    
    # Plot feature distributions by cluster
    plt.subplot(1, 2, 2)
    cluster_means = []
    for i in range(n_clusters):
        cluster_means.append(features[labels == i].mean(axis=0))
    cluster_means = np.array(cluster_means)
    
    sns.heatmap(cluster_means, cmap='YlOrRd')
    plt.title('Average Feature Values by Cluster')
    plt.xlabel('Feature')
    plt.ylabel('Cluster')
    
    plt.tight_layout()
    plt.show()

def plot_silhouette(embeddings, labels):
    """
    Plot silhouette analysis
    
    Parameters:
    -----------
    embeddings : array-like
        Node embeddings
    labels : array-like
        Cluster assignments
    """
    from sklearn.metrics import silhouette_samples
    
    # Compute silhouette scores
    silhouette_vals = silhouette_samples(embeddings, labels)
    
    # Plot
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = len(cluster_silhouette_vals)
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         alpha=0.7)
        
        y_lower = y_upper + 10
    
    plt.title('Silhouette Analysis')
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster')
    plt.axvline(x=silhouette_vals.mean(), color="red", linestyle="--")
    plt.show() 