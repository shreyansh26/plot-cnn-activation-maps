# Plot CNN Activation Maps

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification and provides a script to visualize the activation maps of convolutional layers.

## Project Structure

```
.
├── README.md
├── train.py          # Script for training the CNN model
├── visualize.py      # Script for inference and activation map visualization
├── data/            # Directory for MNIST dataset (auto-downloaded)
├── mnist_cnn.pth    # Saved model weights (generated after training)
└── activation_maps/ # Directory for saved activation visualizations
```

## Requirements

The project requires the following Python packages:
- torch
- torchvision
- matplotlib
- numpy

You can install them using pip:
```bash
pip install torch torchvision matplotlib numpy
```

## Model Architecture

The CNN architecture consists of:
- Input layer (28x28 grayscale images)
- Conv2D layer (16 filters, 3x3 kernel) + ReLU + MaxPool2D
- Conv2D layer (32 filters, 3x3 kernel) + ReLU + MaxPool2D
- Fully connected layer (128 units) + ReLU
- Output layer (10 units for digit classification)

## Usage

1. First, train the model:
```bash
python train.py
```
This will:
- Download the MNIST dataset
- Train the model for 5 epochs
- Save the model weights to `mnist_cnn.pth`

2. Then, visualize the activation maps:
```bash
python visualize.py
```
This will:
- Load the trained model
- Select 5 random test images
- Generate and save activation maps for both convolutional layers
- Save visualizations in the `activation_maps` directory

## Visualization Output

The script generates visualization files in the following format:
- `sample_X_convY_activations.png`
  - Where X is the sample number (1-5)
  - Y is the convolutional layer number (1-2)