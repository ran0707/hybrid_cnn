

# Hybrid CNN package

This package provides a custom Hybrid CNN model with residual blocks, Squeeze-and-Excitation (SE) blocks for attention, and a DBSR (Deep Back-Projection Super-Resolution) model for image enhancement. It also includes a custom dataset loader that applies DBSR and data augmentation.

## Features

- **HybridCNN**: A custom CNN architecture with residual and SE blocks.
- **DBSR**: A Deep Back-Projection Super-Resolution model for image resolution enhancement.
- **CustomDBSRDataset**: A dataset class that applies DBSR to images along with augmentations.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
    
2. Navigate into the project directory:
    ```bash
    cd hybrid_cnn
    ```
    
3. Install the package:
    ```bash
    pip install .
    ```

4. Ensure all dependencies are installed:
    - PyTorch: `torch`
    - TorchVision: `torchvision`
    - Other dependencies will be installed automatically by the package.

## Usage

### 1. Import the Package

After installation, you can import the `HybridCNN` model, the `DBSR` model, and the `CustomDBSRDataset` class as follows:

```python
from hybrid_cnn import HybridCNN, DBSR, CustomDBSRDataset
```

### 2. Example: Training the HybridCNN

```python
import torch
from hybrid_cnn import HybridCNN

# Initialize the HybridCNN model for a classification task with 10 output classes
model = HybridCNN(num_classes=10)

# Example input: A batch of 8 images with 3 channels, each 224x224 pixels
inputs = torch.randn(8, 3, 224, 224)

# Forward pass
outputs = model(inputs)
print(outputs.shape)  # Expected output: [8, 10]
```

### 3. Example: Using the DBSR Model and Custom Dataset

```python
from hybrid_cnn import DBSR, CustomDBSRDataset, get_dataloaders

# Path to the dataset folder
dataset_folder = 'path/to/your/dataset'

# Create the data loaders with DBSR and augmentations
train_loader, test_loader = get_dataloaders(
    folder_path=dataset_folder, 
    batch_size=32, 
    dbsr_blocks=4, 
    augment=True
)

# Example: Accessing a batch from the train loader
for images, labels in train_loader:
    print(images.shape, labels)
```

### 4. Customizing DBSR and Dataset

- You can modify the number of DBSR blocks in the model by passing a different `dbsr_blocks` value when initializing the `CustomDBSRDataset` or `get_dataloaders()` functions.

### 5. Installation for Different Python Versions

If you're using multiple Python versions (e.g., 3.10 and 3.11), ensure that you install the package in each Python environment individually:

- Activate your virtual environment for Python 3.10 or 3.11 and run:
    ```bash
    pip install .
    ```

## Dependencies

The package requires the following dependencies, which will be installed automatically:

- `torch` (PyTorch)
- `torchvision`
- `numpy`

Ensure that these packages are available in your Python environment, especially if you switch between Python versions.


## Contributing

Feel free to submit issues and pull requests if you have suggestions for improvements or new features.

---


