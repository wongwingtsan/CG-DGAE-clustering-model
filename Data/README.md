# Dataset Description

The complete dataset can be found here: [Dataset](https://drive.google.com/drive/folders/1uboZBZwdevcPBgOViv2EEZ6ibdcMJ2U_?usp=drive_link)

## Overview
This dataset is designed for graph-based clustering of time series data from stations. It contains temporal measurements and spatial relationships between stations. The weekly median traffic sensor data is used for training.
## Required Data Structure

### Directory Structure
```
data/
├── stations/                     # Directory containing all station data files
│   ├── station1.csv
│   ├── station2.csv
│   └── ...
├── distance.csv                  # Distance matrix between stations
└── coordinates.csv              # Station coordinates
```

### Required Files

1. **Station Data Files**
   - **Directory**: `data/stations/` #see week_median_data
   - **Files**: Multiple CSV files, one per station
   - **Naming**: `[station_id].csv`
   - **Content**: Time series data with 7 measurement columns
   - **Dimensions**: Each station file contains:
     - Rows: Time points
     - Columns: Different measurements/features (7 columns)

2. **Distance Matrix File**
   - **Path**: `data/distance.csv` #distancewith0.csv
   - **Format**: CSV file containing pairwise distances between stations
   - **Structure**: 
     - Row/Column headers: Station IDs
     - Values: Distance between corresponding stations
     - Symmetric matrix (distance(A,B) = distance(B,A))

3. **Coordinates File**
   - **Path**: `data/coordinates.csv`#station_coordinates.csv
   - **Format**: CSV file containing station locations
   - **Columns**:
     - `ID`: Station identifier
     - Coordinate information for each station

## Data Processing

### Feature Extraction
The data processing pipeline (`src/data/process_data.py`) performs the following steps:
1. Loads individual station CSV files from `stations/` directory
2. Extracts time series features
3. Generates adjacency matrix based on k-nearest neighbors
4. Creates batched data for model training

### Adjacency Matrix Generation
- Based on station distances from `distance.csv`
- Uses k-nearest neighbors approach
- k is configurable (default in code)
- Normalized using random walk matrix

## Usage

### Data Loading
```python
from src.data.process_data import load_station_data

# Load and process data
X, adj, station_names = load_station_data(
    data_dir='data/stations',          # Directory containing station CSV files
    distance_file='data/distance.csv',  # Distance matrix file
    coordinates_file='data/coordinates.csv',  # Station coordinates file
    n_neighbors=10  # Adjustable parameter
)
```

### Data Format After Processing
- **X**: Node features matrix (stations × features)
- **adj**: Adjacency matrix (stations × stations)
- **station_names**: List of station identifiers

### Creating DataLoader
```python
from src.data.process_data import prepare_training_data

dataloader = prepare_training_data(X, batch_size=32)
```

## Data Statistics
- Number of stations: [Total number of stations]
- Time series length: 288 points per station
- Number of features: 7 per time point
- Spatial coverage: [Geographic area covered]

## Notes
- All time series data is normalized during processing
- Missing values are handled during data loading
- The adjacency matrix is symmetric and normalized
- Station IDs are consistent across all files
- Make sure all file paths match the expected structure exactly 