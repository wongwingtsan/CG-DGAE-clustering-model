import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import argparse

def load_station_data(data_dir, distance_file, coordinates_file, n_neighbors):
    """Load and process station data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing station CSV files
    distance_file : str
        Path to distance matrix CSV file
    coordinates_file : str
        Path to station coordinates CSV file
    n_neighbors : int
        Number of neighbors to use for adjacency matrix construction
        
    Returns:
    --------
    X : np.ndarray
        Node features matrix
    adj : np.ndarray
        Adjacency matrix
    station_names : list
        List of station names
    """
    # Get list of stations
    arr = os.listdir(data_dir)
    namelist = [station[:-4] for station in arr]
    
    # Load station IDs from distance file
    stations = pd.read_csv(distance_file)['Y0']
    stations = list(stations)
    
    # Filter station names
    stationname = []
    crdinstation = []
    for item in namelist:
        if item[0:8] in stations:
            stationname.append(item)
            crdinstation.append(item[0:8])
    
    crdinstation = list(set(crdinstation))
    
    # Load features (X)
    X = []
    for station in stationname:
        values = []
        file_path = os.path.join(data_dir, f"{station}.csv")
        station_df = pd.read_csv(file_path)
        for col in station_df.columns[1:8]:  # columns 1-7 contain time series data
            values.extend(station_df[col].tolist())
        X.append(values)
    
    X = np.array(X).astype(np.float32)
    
    # Generate adjacency matrix
    adj = generate_adjacency_matrix(
        distance_file=distance_file,
        coordinates_file=coordinates_file,
        station_names=stationname,
        core_station_ids=crdinstation,
        n_neighbors=n_neighbors
    )
    
    return X, adj, stationname

def generate_adjacency_matrix(distance_file, coordinates_file, station_names, core_station_ids, n_neighbors):
    """Generate adjacency matrix from distance matrix
    
    Parameters:
    -----------
    distance_file : str
        Path to distance matrix file
    coordinates_file : str
        Path to coordinates file
    station_names : list
        List of full station names
    core_station_ids : list
        List of core station IDs (first 8 characters)
    n_neighbors : int
        Number of neighbors to keep for each node
        
    Returns:
    --------
    adj : np.ndarray
        Adjacency matrix
    """
    def generate_distance_df(path_distance, path_coordinates, namelist):
        df = pd.read_csv(path_distance)
        names1 = list(df['Y0'])
        df = df.drop(columns=['Y0'])
        df.columns = names1
        CO = pd.read_csv(path_coordinates)
        df['stationname'] = CO.ID
        df = df.set_index('stationname')
        df = df.loc[namelist]
        df = df[namelist]
        return df
    
    # Generate distance matrix
    Md = generate_distance_df(
        path_distance=distance_file,
        path_coordinates=coordinates_file,
        namelist=core_station_ids
    )
    
    # Create full distance matrix
    matrix = pd.DataFrame(columns=station_names, index=station_names)
    for station in station_names:
        for STATION in station_names:
            matrix.loc[station, STATION] = Md.loc[station[0:8], STATION[0:8]]
    
    # Normalize and convert to similarity matrix
    MAX = matrix.max().max()
    matrix = matrix / MAX
    matrix = np.exp(-matrix.astype(float))
    
    # Convert to adjacency matrix (keep top k neighbors)
    A = matrix.values
    adj = np.zeros_like(A)
    for i in range(len(A)):
        ind = np.argsort(A[i])[::-1][:n_neighbors]
        adj[i][ind] = 1
    
    return adj

def prepare_training_data(features, batch_size=32):
    """Prepare training data in the required format
    
    Parameters:
    -----------
    features : np.ndarray
        Input features matrix
    batch_size : int
        Batch size for DataLoader
        
    Returns:
    --------
    dataloader : torch.utils.data.DataLoader
        DataLoader for training
    """
    # Transpose features
    training_set = features.transpose()
    
    # Create batches of size 288 (assuming this is the sequence length)
    needles = list(range(int(len(training_set)/288)))
    needles = [item*288 for item in needles]
    
    feed_batch = []
    for i in range(len(needles)):
        feed_batch.append(training_set[needles[i]:needles[i]+288,:])
    
    inputs = np.array(feed_batch)
    
    # Convert to torch tensor and normalize
    inputs = torch.from_numpy(inputs.astype('float32'))
    inputs = torch.nn.functional.normalize(inputs, dim=1)
    
    # Create DataLoader
    dataloader = DataLoader(inputs, batch_size=batch_size, shuffle=True)
    
    return dataloader

def main(args):
    # Load and process data
    print("Loading and processing data...")
    X, adj, station_names = load_station_data(
        args.data_dir,
        args.distance_file,
        args.coordinates_file,
        args.n_neighbors
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Using {args.n_neighbors} neighbors for adjacency matrix")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save processed data
    print("Saving processed data...")
    np.save(os.path.join(args.output_dir, 'features.npy'), X)
    np.save(os.path.join(args.output_dir, 'adj_matrix.npy'), adj)
    
    # Save station names
    with open(os.path.join(args.output_dir, 'station_names.txt'), 'w') as f:
        for station in station_names:
            f.write(f"{station}\n")
    
    # Create and save DataLoader if requested
    if args.create_dataloader:
        print("Creating DataLoader...")
        dataloader = prepare_training_data(X, args.batch_size)
        torch.save(dataloader, os.path.join(args.output_dir, 'dataloader.pth'))
    
    print("Data processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process time series data for graph clustering')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing station CSV files')
    parser.add_argument('--distance_file', type=str, required=True,
                      help='Path to distance matrix CSV file')
    parser.add_argument('--coordinates_file', type=str, required=True,
                      help='Path to station coordinates CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--n_neighbors', type=int, default=12,
                      help='Number of neighbors to use for adjacency matrix construction')
    parser.add_argument('--create_dataloader', action='store_true',
                      help='Whether to create and save DataLoader')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for DataLoader')
    
    args = parser.parse_args()
    main(args) 