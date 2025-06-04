# Graph Node Clustering Project

This repository contains implementation of graph node clustering using various deep learning approaches, specifically focusing on Graph Attention Networks (GAT) and Deep Graph Convolutional Networks (DGCN). For the stage two, reconstruction, please see the repository [reconstruction model](https://github.com/wongwingtsan/CG-DGAE-reconstruction-model)

## Project Overview

The project implements and compares different graph-based clustering approaches:
- Graph Attention Networks (GAT)
- Deep Graph Convolutional Networks (DGCN)
- DGCN with Autoencoder (DGCN_AE) - Final Model

## Project Structure

```
.
├── Data/               # Dataset directory
├── src/               # Source code
│   ├── models/        # Model implementations
│   │   ├── gat.py    # Graph Attention Network
│   │   ├── dgcn.py   # Deep Graph Convolutional Network
│   │   └── dgcn_ae.py# DGCN with Autoencoder
│   ├── utils/        # Utility functions
│   └── data/         # Data processing scripts
├── tests/            # Test files
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/graph-clustering.git
cd graph-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Models

### GAT (Graph Attention Networks)
Implementation of the Graph Attention Network model for node clustering.

### DGCN (Deep Graph Convolutional Networks)
Implementation of Deep Graph Convolutional Networks for improved node representation learning.

### DGCN_AE (DGCN with Autoencoder)
The final model combining DGCN with an autoencoder architecture for enhanced clustering performance.

## Usage

Detailed usage instructions and examples will be provided in the individual model documentation.

## Testing

The clustering results are evaluated using various methods to ensure robust performance measurement.

## Requirements

See `requirements.txt` for a full list of dependencies.
