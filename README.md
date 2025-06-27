# Image Pattern Extraction

A Python tool for extracting and analyzing patterns in image datasets using deep learning feature extraction, dimensionality reduction, and clustering algorithms.

## Overview

This project helps you discover patterns in your image collections by:

1. **Feature Extraction**: Using pre-trained deep learning models (ResNet, EfficientNet, MobileNetV3) to extract high-dimensional features from images
2. **Dimensionality Reduction**: Reducing features to 2D or 3D using t-SNE or UMAP for visualization
3. **Clustering**: Grouping similar images using K-Means or DBSCAN algorithms
4. **Pattern Analysis**: Identifying representative images from each cluster and outliers

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install Dependencies

Using uv (recommended):

```bash
# Clone the repository
git clone <repository-url>
cd image_pattern_extraction

# Install dependencies
uv sync

# Install development dependencies (for type checking)
uv sync --group dev
```

## Usage

### Quick Start

1. **Prepare your data**: Place your images in the `data/` directory, organized in subdirectories
2. **Run the tool**: `uv run python main.py`
3. **Follow the interactive prompts** to:
   - Select your image dataset
   - Choose a feature extraction backbone
   - Pick a dimensionality reduction method
   - Select a clustering algorithm
   - Configure output settings

### Command Line Options

```bash
python main.py -c /path/to/config.toml
```

- `-c, --config`: Path to configuration file (default: `./configs/user-config.toml`)

### Configuration

The tool uses TOML configuration files. Key settings include:

```toml
[global_settings]
data_dir = "./data"          # Input data directory
result_dir = "./results"     # Output results directory
exit_keys = ["q", "Q", "exit", "quit"]

[extractor.backbone.ResNet]
input = 256                  # Input image size
output = 2048                # Output feature dimension

[reduction.t-SNE]
n_components = 2             # Target dimensions
max_iter = 1000              # Maximum iterations

[clustering.K-Means]
n_clusters = 6               # Number of clusters
```

## Output

The tool generates:

1. **Cluster directories**: Each cluster gets its own folder with representative images
2. **Outlier directory**: Contains images identified as outliers (DBSCAN only)
3. **Visualization plots**: Interactive plots showing clustering results
4. **Result CSV**: Detailed data with cluster assignments and distances
5. **Configuration copy**: Saved configuration for reproducibility
