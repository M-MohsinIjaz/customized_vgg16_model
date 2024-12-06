# Customized VGG16 Model Implementation

This repository contains an implementation of a VGG16 based customized CNN model using PyTorch. The project includes custom modules for data loading, training, testing, and visualization of results.

## Features
- **Custom DataLoader**: A custom `MNIST_DATASET` class for loading and preprocessing data.
- **Custom Convolution Layer**: Implementation of a convolutional layer from scratch.
- **Custom Max Pooling**: Custom max pooling operation.
- **VGG16-based Architecture**: Includes modifications with a custom block.
- **Training and Validation**: Dynamic learning rate decay and early stopping.
- **Performance Metrics**: Accuracy, confusion matrix, F1 score, and recall.
- **t-SNE Visualization**: Visualizes feature representations after the first epoch and after full training.

## Dataset

The implementation uses the MNIST dataset. Images and their corresponding labels are provided in CSV files and directories.

### Dataset Structure
- `train.csv` and `test.csv`: CSV files containing image filenames and their labels.
- `train/` and `test/`: Directories containing image files.

## Setup

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Scikit-image

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cnn-implementation.git
   cd cnn-implementation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
1. Prepare your dataset following the expected structure.
2. Call the `Load_DataSet` function to load the data.
3. Train the model using the `train_model` function.
   ```python
   train_loader, val_loader, test_loader = Load_DataSet("path/to/dataset", 50000, 10000, 10000, 32)
   model, train_loss_list, val_loss_list, val_accuracy_list = train_model(train_loader, val_loader)
   ```

### Testing the Model
Use the `test_model` function to evaluate the model on the test set:
```python
test_model(model, test_loader, device)
```

### Visualizing t-SNE
Generate t-SNE plots to visualize the model's feature representations:
```python
plot_tsne_first_last(test_loader, device)
```

### Plotting Loss and Accuracy
Visualize training loss, validation loss, and accuracy over epochs:
```python
plot_loss_acc(executed_epochs, train_loss_list, val_loss_list, val_accuracy_list)
```

### Custom Layers and Blocks
This implementation includes custom layers and blocks:
- **Custom Convolution Layer**: `Conv` class.
- **Custom Block**: `CustomBlock` class used in place of VGG16's second block.

### Model Saving and Loading
Save and load model checkpoints:
```python
save_model("model_name", model)
model = load_model("model_name", model)
```

## Results
The implementation achieves competitive performance on the MNIST dataset with:
- Accuracy: >98% on test data (depending on hyperparameters and epochs).
- F1 Score and Confusion Matrix for detailed evaluation.

## File Structure
```
.
├── main.py               # Main script
├── requirements.txt      # Dependencies
├── mnist_dataset/
│   ├── dla3_train/
│   │   ├── train/        # Directory containing training images
│   │   └── train.csv     # Training labels CSV
│   ├── dla3_test/
│   │   ├── test/        # Directory containing training images
│   │   └── test.csv     # Training labels CSV
└── README.md             # Documentation
```

## Contributing
Feel free to open issues or submit pull requests for improvements and bug fixes.
