import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from train import SimpleCNN
from torch.utils.data import DataLoader
import os

def visualize_activations(model, sample_images, save_dir='activation_maps'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get activations for each sample
        for idx, image in enumerate(sample_images):
            # Add batch dimension
            input_tensor = image.unsqueeze(0)
            
            # Forward pass to get activations
            _ = model(input_tensor)
            
            # Plot activations for each conv layer
            for layer_name, activations in model.activations.items():
                print(activations.shape)
                activations = activations.squeeze(0)
                num_kernels = activations.size(0)
                
                # Create a grid for plotting
                num_cols = 8
                num_rows = (num_kernels + num_cols - 1) // num_cols
                fig, axes = plt.subplots(num_rows, num_cols, 
                                       figsize=(num_cols*2, num_rows*2))
                axes = axes.flatten()
                
                # Plot each kernel's activation
                for k in range(num_kernels):
                    if k < len(axes):
                        axes[k].imshow(activations[k].cpu(), cmap='viridis')
                        axes[k].axis('off')
                
                # Remove empty subplots
                for k in range(num_kernels, len(axes)):
                    fig.delaxes(axes[k])
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/sample_{idx+1}_{layer_name}_activations.png')
                plt.close()

def main():
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST('./data', train=False, 
                                            download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device, weights_only=True))
    
    # Get a few sample images
    sample_images = []
    for data, _ in test_loader:
        data = data.to(device)
        sample_images.append(data.squeeze(0))
        if len(sample_images) == 5:  # Visualize 5 samples
            break
    
    # Visualize activations
    visualize_activations(model, sample_images)

if __name__ == '__main__':
    main() 