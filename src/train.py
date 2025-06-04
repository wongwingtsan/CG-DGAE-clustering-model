import argparse
import torch
import torch.nn.functional as F
import numpy as np
from models.custom_gae import DGCN_AE, calculate_random_walk_matrix
from utils.data_utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, preprocess_features
from evaluation.clustering_evaluation import extract_embeddings, evaluate_clustering, print_clustering_results

def train_model(model, dataloader, adj, M, optimizer, n_epochs, device):
    """Train the model"""
    model.train()
    
    # Calculate random walk matrices
    A_q = calculate_random_walk_matrix(adj.cpu().numpy())
    A_h = calculate_random_walk_matrix(adj.cpu().numpy().T)
    A_q = torch.FloatTensor(A_q).to(device)
    A_h = torch.FloatTensor(A_h).to(device)
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            # Forward pass
            X_res, X_em, Edge_em, A_pred = model(batch, A_q, A_h, adj, M)
            
            # Calculate losses
            reconstruction_loss = F.mse_loss(X_res, batch)
            edge_reconstruction_loss = F.binary_cross_entropy(A_pred, adj.to_dense())
            
            # Total loss
            loss = reconstruction_loss + 0.1 * edge_reconstruction_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}')

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Load data
    print("Loading data...")
    dataloader = torch.load(args.dataloader_path)
    adj = np.load(args.adj_path)
    
    # Preprocess adjacency matrix
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    # Create edge weight matrix M
    M = torch.ones_like(adj.to_dense())
    
    # Initialize model
    model = DGCN_AE(
        h=args.time_dimension,
        z=args.hidden_dimension,
        k=args.order,
        num_features=args.num_features,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha
    )
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    print("Training model...")
    train_model(model, dataloader, adj, M, optimizer, args.epochs, device)
    
    # Calculate random walk matrices for embedding extraction
    A_q = calculate_random_walk_matrix(adj.cpu().numpy())
    A_h = calculate_random_walk_matrix(adj.cpu().numpy().T)
    A_q = torch.FloatTensor(A_q).to(device)
    A_h = torch.FloatTensor(A_h).to(device)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataloader, adj, M, A_q, A_h, device)
    
    # Save embeddings
    print("Saving embeddings...")
    np.save(args.output_embeddings, embeddings)
    
    # Evaluate clustering if requested
    if args.evaluate_clustering:
        print("\nEvaluating clustering algorithms...")
        results = evaluate_clustering(embeddings, adj.to_dense().cpu().numpy())
        print_clustering_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader_path', type=str, required=True,
                      help='Path to saved DataLoader')
    parser.add_argument('--adj_path', type=str, required=True,
                      help='Path to adjacency matrix')
    parser.add_argument('--output_embeddings', type=str, required=True,
                      help='Path to save extracted embeddings')
    parser.add_argument('--time_dimension', type=int, default=288,
                      help='Input time dimension')
    parser.add_argument('--hidden_dimension', type=int, default=144,
                      help='Hidden dimension size')
    parser.add_argument('--order', type=int, default=2,
                      help='Order of graph convolution')
    parser.add_argument('--num_features', type=int, default=288,
                      help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='Hidden size for GAT')
    parser.add_argument('--embedding_size', type=int, default=128,
                      help='Embedding size for GAT')
    parser.add_argument('--alpha', type=float, default=0.2,
                      help='LeakyReLU angle for GAT')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs')
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA')
    parser.add_argument('--evaluate_clustering', action='store_true',
                      help='Whether to evaluate clustering algorithms')
    
    args = parser.parse_args()
    main(args) 