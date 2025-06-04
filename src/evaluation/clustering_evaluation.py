import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import itertools

def extract_embeddings(model, dataloader, adj, M, A_q, A_h, device):
    """Extract embeddings from trained model
    
    Parameters:
    -----------
    model : DGCN_AE
        Trained model
    dataloader : DataLoader
        Data loader containing the input data
    adj, M, A_q, A_h : torch.Tensor
        Model input matrices
    device : torch.device
        Device to run the model on
        
    Returns:
    --------
    embeddings : np.ndarray
        Combined node and edge embeddings
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for data in dataloader:
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            X_res, embed, Edge_em, A_pred = model(data, A_q, A_h, adj, M)
            
            # Process embeddings
            # 1. Process Edge embeddings
            Edge_em = torch.transpose(Edge_em, 0, 1)
            Edge_em = torch.flatten(Edge_em, start_dim=1)
            
            # 2. Process node embeddings
            embed = torch.transpose(embed, 0, 1)
            embed = torch.flatten(embed, start_dim=1)
            
            # 3. Combine node and edge embeddings
            combined_embed = torch.cat((embed, Edge_em), 1)
            combined_embed = combined_embed.detach().cpu().numpy()
            
            embeddings.append(combined_embed)
            print(f"Batch shape: {data.shape}")
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=1)
    return embeddings

def evaluate_clustering(embeddings, cluster_range=range(60, 480)):
    """Evaluate clustering algorithms on the embeddings
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Node embeddings to cluster
    cluster_range : range
        Range of cluster numbers to try for KMeans and AgglomerativeClustering
        
    Returns:
    --------
    results : dict
        Dictionary containing results for each algorithm
    """
    results = {}
    
    # 1. K-Means with different cluster numbers
    kmeans_scores = pd.DataFrame(columns=['clusters'])
    for n_clusters in cluster_range:
        clusterer = KMeans(n_clusters=n_clusters, n_init='auto', random_state=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Calculate scores
        scores = {
            'clusters': n_clusters,
            'inertia_score': clusterer.inertia_,
            'silhouette_score': silhouette_score(embeddings, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(embeddings, cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(embeddings, cluster_labels)
        }
        
        kmeans_scores = kmeans_scores._append(scores, ignore_index=True)
        print(
            "KMeans - For n_clusters =", n_clusters,
            "the inertia score is:", scores['inertia_score'],
            "The average silhouette_score is:", scores['silhouette_score'],
            "the cal_hara score is:", scores['calinski_harabasz_score'],
            "the davidb_score is:", scores['davies_bouldin_score']
        )
    
    # Find best scores for KMeans
    best_kmeans = {
        'best_silhouette': {
            'score': kmeans_scores['silhouette_score'].max(),
            'n_clusters': kmeans_scores.loc[kmeans_scores['silhouette_score'].idxmax(), 'clusters']
        },
        'best_calinski': {
            'score': kmeans_scores['calinski_harabasz_score'].min(),
            'n_clusters': kmeans_scores.loc[kmeans_scores['calinski_harabasz_score'].idxmin(), 'clusters']
        },
        'all_scores': kmeans_scores
    }
    results['kmeans'] = best_kmeans
    
    # 2. Agglomerative Clustering with different cluster numbers
    aggl_scores = pd.DataFrame(columns=['clusters'])
    for n_clusters in cluster_range:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Calculate scores
        scores = {
            'clusters': n_clusters,
            'silhouette_score': silhouette_score(embeddings, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(embeddings, cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(embeddings, cluster_labels)
        }
        
        aggl_scores = aggl_scores._append(scores, ignore_index=True)
        print(
            "Agglomerative - For n_clusters =", n_clusters,
            "The average silhouette_score is:", scores['silhouette_score'],
            "the cal_hara score is:", scores['calinski_harabasz_score'],
            "the davidb_score is:", scores['davies_bouldin_score']
        )
    
    # Find best scores for Agglomerative
    best_aggl = {
        'best_silhouette': {
            'score': aggl_scores['silhouette_score'].max(),
            'n_clusters': aggl_scores.loc[aggl_scores['silhouette_score'].idxmax(), 'clusters']
        },
        'best_calinski': {
            'score': aggl_scores['calinski_harabasz_score'].min(),
            'n_clusters': aggl_scores.loc[aggl_scores['calinski_harabasz_score'].idxmin(), 'clusters']
        },
        'all_scores': aggl_scores
    }
    results['agglomerative'] = best_aggl
    
    # 3. OPTICS
    optics_clusterer = OPTICS()
    optics_labels = optics_clusterer.fit_predict(embeddings)
    
    optics_scores = {
        'n_clusters': len(set(optics_labels)) - (1 if -1 in optics_labels else 0),  # Exclude noise points
        'silhouette_score': silhouette_score(embeddings, optics_labels),
        'calinski_harabasz_score': calinski_harabasz_score(embeddings, optics_labels),
        'davies_bouldin_score': davies_bouldin_score(embeddings, optics_labels)
    }
    print(
        "OPTICS - For n_clusters =", optics_scores['n_clusters'],
        "The average silhouette_score is:", optics_scores['silhouette_score'],
        "the cal_hara score is:", optics_scores['calinski_harabasz_score'],
        "the davidb_score is:", optics_scores['davies_bouldin_score']
    )
    results['optics'] = optics_scores
    
    return results

def print_clustering_results(results):
    """Print clustering results in a formatted way"""
    # Print KMeans results
    print("\nK-MEANS Results:")
    print("-" * 50)
    kmeans_results = results['kmeans']
    print(f"Best Silhouette Score: {kmeans_results['best_silhouette']['score']:.4f} "
          f"(n_clusters = {kmeans_results['best_silhouette']['n_clusters']})")
    print(f"Best Calinski-Harabasz Score: {kmeans_results['best_calinski']['score']:.4f} "
          f"(n_clusters = {kmeans_results['best_calinski']['n_clusters']})")
    
    # Print Agglomerative results
    print("\nAGGLOMERATIVE CLUSTERING Results:")
    print("-" * 50)
    aggl_results = results['agglomerative']
    print(f"Best Silhouette Score: {aggl_results['best_silhouette']['score']:.4f} "
          f"(n_clusters = {aggl_results['best_silhouette']['n_clusters']})")
    print(f"Best Calinski-Harabasz Score: {aggl_results['best_calinski']['score']:.4f} "
          f"(n_clusters = {aggl_results['best_calinski']['n_clusters']})")
    
    # Print OPTICS results
    print("\nOPTICS Results:")
    print("-" * 50)
    optics_results = results['optics']
    print(f"Number of clusters found: {optics_results['n_clusters']}")
    print(f"Silhouette Score: {optics_results['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Score: {optics_results['calinski_harabasz_score']:.4f}")
    print(f"Davies-Bouldin Score: {optics_results['davies_bouldin_score']:.4f}")

def plot_clustering_scores(results):
    """Plot scores across different numbers of clusters for KMeans and Agglomerative"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # KMeans plots
    kmeans_scores = results['kmeans']['all_scores']
    ax1.plot(kmeans_scores['clusters'], kmeans_scores['silhouette_score'], label='KMeans')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('KMeans: Silhouette Score vs Number of Clusters')
    
    ax2.plot(kmeans_scores['clusters'], kmeans_scores['calinski_harabasz_score'], label='KMeans')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.set_title('KMeans: Calinski-Harabasz Score vs Number of Clusters')
    
    # Agglomerative plots
    aggl_scores = results['agglomerative']['all_scores']
    ax3.plot(aggl_scores['clusters'], aggl_scores['silhouette_score'], label='Agglomerative')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('Agglomerative: Silhouette Score vs Number of Clusters')
    
    ax4.plot(aggl_scores['clusters'], aggl_scores['calinski_harabasz_score'], label='Agglomerative')
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Calinski-Harabasz Score')
    ax4.set_title('Agglomerative: Calinski-Harabasz Score vs Number of Clusters')
    
    plt.tight_layout()
    return fig 