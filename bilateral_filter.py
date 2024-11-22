"""
PyTorch Implementation of Trainable Bilateral Filter for Image Denoising

Based on the paper: https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15718

This module implements a trainable bilateral filter for image denoising using PyTorch.
It supports both 2D and 3D image processing, handling grayscale and RGB images with
GPU acceleration when available.

Key Components:
    - BilateralFilter: Neural network module implementing the bilateral filter
    - DenoisingPipeline: Multi-stage denoising pipeline using bilateral filters

Features:
    - Flexible processing of 2D/3D and grayscale/RGB images
    - GPU acceleration support
    - Cached kernel computations for improved performance
    - Comprehensive error handling and logging
    - Multi-stage denoising pipeline
    - Gradient-based optimization of filter parameters

Usage:
    # Basic usage example
    model = DenoisingPipeline()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    trained_model = train(model, clean_image, noisy_image, device=device)
    
    # Denoise new images
    with torch.no_grad():
        denoised = model(noisy_tensor)

Dependencies:
    - torch
    - numpy
    - matplotlib
    - skimage
    - loguru
    - tqdm

Author: Yipeng Sun
Email: yipeng.sun@fau.de
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from torch.utils.data import DataLoader
from loguru import logger
import math
import sys
from tqdm import tqdm
import torch.optim as optim
import os


class BilateralFilter(nn.Module):
    """Trainable bilateral filter implementation as a PyTorch neural network module.

    This class implements a trainable bilateral filter that can process both 2D and 3D images.
    The filter combines spatial and range filtering to preserve edges while removing noise.

    Args:
        sigma_sx (float, optional): Initial spatial sigma for x dimension. Defaults to 0.5.
        sigma_sy (float, optional): Initial spatial sigma for y dimension. Defaults to 0.5.
        sigma_sz (float, optional): Initial spatial sigma for z dimension (3D only). Defaults to 0.5.
        sigma_r (float, optional): Initial range sigma for intensity differences. Defaults to 0.1.

    Attributes:
        sigma_sx (nn.Parameter): Learnable parameter for x-dimension spatial sigma.
        sigma_sy (nn.Parameter): Learnable parameter for y-dimension spatial sigma.
        sigma_sz (nn.Parameter): Learnable parameter for z-dimension spatial sigma.
        sigma_r (nn.Parameter): Learnable parameter for range sigma.
        std_multiplier (int): Multiplier for standard deviation to determine kernel size.
        kernel_size (int): Size of the filter kernel.
        spatial_kernel_cache (torch.Tensor): Cache for computed spatial kernels.
        last_spatial_params (torch.Tensor): Parameters from last spatial kernel computation.

    Raises:
        ValueError: If any sigma parameter is not positive.
        RuntimeError: If kernel computation fails.
        TypeError: If input tensor is not of correct type.
    """
    def __init__(self, sigma_sx=0.5, sigma_sy=0.5, sigma_sz=0.5, sigma_r=0.1):
        super(BilateralFilter, self).__init__()
        
        # Input validation
        for param_name, value in {'sigma_sx': sigma_sx, 'sigma_sy': sigma_sy, 
                                'sigma_sz': sigma_sz, 'sigma_r': sigma_r}.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{param_name} must be a positive number")
        
        # Parameters
        self.sigma_sx = nn.Parameter(torch.tensor(float(sigma_sx)))
        self.sigma_sy = nn.Parameter(torch.tensor(float(sigma_sy)))
        self.sigma_sz = nn.Parameter(torch.tensor(float(sigma_sz)))
        self.sigma_r = nn.Parameter(torch.tensor(float(sigma_r)))
        self.std_multiplier = 5  # Using 5σ rule for coverage range
        self.kernel_size = None
        
        # Cache for computed kernels
        self.register_buffer('spatial_kernel_cache', None)
        self.register_buffer('last_spatial_params', None)

    def _update_kernel_size(self):
        """Update the kernel size based on the current spatial sigma values.

        The kernel size is calculated to cover 5 standard deviations on each side
        of the center pixel, ensuring adequate spatial support for the filter.
        The final size is adjusted to be odd and at least 3 pixels wide.
        """
        # Calculate kernel size based on largest spatial sigma
        max_sigma = max(float(self.sigma_sx), float(self.sigma_sy))
        if len(self.input_shape) == 5:  # 3D case
            max_sigma = max(max_sigma, float(self.sigma_sz))
        
        # Kernel size should be odd and cover 5 standard deviations on each side
        kernel_size = int(2 * np.ceil(max_sigma * self.std_multiplier) + 1)
        # Ensure minimum size of 3
        kernel_size = max(3, kernel_size)
        # Ensure kernel size is odd
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def compute_spatial_kernel(self, device):
        """Compute the spatial component of the bilateral filter kernel.

        Args:
            device (torch.device): Device to perform computations on.

        Returns:
            torch.Tensor: Computed spatial kernel.

        Raises:
            RuntimeError: If kernel computation fails.
        """
        # Check if we can use cached kernel
        current_params = torch.stack([
            self.sigma_sx,
            self.sigma_sy,
            self.sigma_sz if len(self.input_shape) == 5 else torch.tensor(0.0, device=device)
        ]).to(device)
        
        if (self.spatial_kernel_cache is not None and 
            self.last_spatial_params is not None and 
            torch.allclose(current_params, self.last_spatial_params)):
            return self.spatial_kernel_cache
        
        try:
            # Compute grid coordinates efficiently using torch.meshgrid
            coords = torch.meshgrid(
                *[torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1, device=device) 
                  for _ in range(2)],
                indexing='ij'
            )
            
            # Ensure parameters are on the correct device
            sigma_sx = self.sigma_sx.to(device)
            sigma_sy = self.sigma_sy.to(device)
            
            # Vectorized computation of spatial kernel
            spatial_kernel = torch.exp(
                -(coords[1].float()**2) / (2 * sigma_sx**2) +
                -(coords[0].float()**2) / (2 * sigma_sy**2)
            )
            
            # Cache the computed kernel
            self.spatial_kernel_cache = spatial_kernel
            self.last_spatial_params = current_params
            
            return spatial_kernel
        except Exception as e:
            raise RuntimeError(f"Error computing spatial kernel: {str(e)}")

    def compute_range_kernel(self, center_values, neighbor_values):
        """Compute the range/intensity component of the bilateral filter kernel.

        Args:
            center_values (torch.Tensor): Center pixel values.
            neighbor_values (torch.Tensor): Neighboring pixel values.

        Returns:
            torch.Tensor: Computed range kernel.

        Raises:
            RuntimeError: If kernel computation fails.
        """
        try:
            device = center_values.device
            # Ensure sigma_r is on the correct device
            sigma_r = self.sigma_r.to(device)
            
            # Compute difference with proper broadcasting
            diff = center_values - neighbor_values
            
            if center_values.shape[1] > 1:  # RGB case
                # Efficient computation for RGB using sum instead of mean
                squared_diff = (diff ** 2).sum(dim=1, keepdim=True)
            else:  # Grayscale case
                squared_diff = diff ** 2
            
            # Add small epsilon for numerical stability
            range_kernel = torch.exp(-squared_diff / (2 * sigma_r**2 + 1e-8))
            return range_kernel
        except Exception as e:
            raise RuntimeError(f"Error computing range kernel: {str(e)}")

    def forward(self, x):
        """Forward pass of the bilateral filter.

        Processes input images through the bilateral filter, supporting both 2D and 3D inputs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, [depth,] height, width).

        Returns:
            torch.Tensor: Filtered output tensor of same shape as input.

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If input dimensions are invalid.
        """
        # Input validation and device handling
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")
        
        if x.dim() not in [4, 5]:
            raise ValueError(f"Input must be a 4D or 5D tensor, got shape: {x.shape}")
        
        if not x.is_floating_point():
            x = x.float()
        
        # Get device from input tensor
        device = x.device
        logger.debug(f"BilateralFilter input - Device: {device}, Shape: {x.shape}")
        
        # Ensure model is on the same device
        self = self.to(device)
        
        self.input_shape = x.shape
        self._update_kernel_size()
        
        try:
            if len(x.shape) == 5:  # 3D case
                batch, channels, depth, height, width = x.shape
                logger.debug(f"Processing 3D input with shape: {x.shape}")
                
                # Handle each depth slice separately
                output_slices = []
                for d in range(depth):
                    slice_2d = x[:, :, d:d+1, :, :]
                    logger.debug(f"Processing slice {d+1}/{depth}, shape: {slice_2d.shape}")
                    
                    # Pad only height and width dimensions
                    x_pad = nn.functional.pad(slice_2d.squeeze(2), [self.pad]*4, mode='reflect')
                    x_pad = x_pad.unsqueeze(2)
                    
                    # Extract patches
                    patches = x_pad.unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)
                    patches = patches.contiguous()
                    patches = patches.view(batch, channels, 1, height, width, -1)
                    
                    # Compute kernels
                    spatial_kernel = self.compute_spatial_kernel(device)
                    spatial_kernel = spatial_kernel.view(1, 1, 1, 1, 1, -1)
                    
                    center_values = slice_2d.unsqueeze(-1)
                    range_kernel = self.compute_range_kernel(center_values, patches)
                    
                    # Combine kernels
                    kernel = spatial_kernel * range_kernel
                    kernel_sum = kernel.sum(dim=-1, keepdim=True)
                    kernel = kernel / (kernel_sum + 1e-8)
                    
                    # Apply filter
                    output_slice = (patches * kernel).sum(dim=-1)
                    output_slices.append(output_slice)
                    
                    logger.debug(f"Slice {d+1} processed, output shape: {output_slice.shape}")
                
                # Combine slices
                output = torch.cat(output_slices, dim=2)
                logger.debug(f"Combined 3D output shape: {output.shape}")
                
            else:  # 2D case
                logger.debug(f"Processing 2D input with shape: {x.shape}")
                batch, channels, height, width = x.shape
                
                # Pad input
                x_pad = nn.functional.pad(x, [self.pad]*4, mode='reflect')
                
                # Extract patches
                patches = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
                patches = patches.contiguous()
                patches = patches.view(batch, channels, height, width, -1)
                
                # Compute kernels
                spatial_kernel = self.compute_spatial_kernel(device)
                spatial_kernel = spatial_kernel.view(1, 1, 1, 1, -1)
                
                center_values = x.unsqueeze(-1)
                range_kernel = self.compute_range_kernel(center_values, patches)
                
                # Combine kernels
                kernel = spatial_kernel * range_kernel
                kernel_sum = kernel.sum(dim=-1, keepdim=True)
                kernel = kernel / (kernel_sum + 1e-8)
                
                # Apply filter
                output = (patches * kernel).sum(dim=-1)
                logger.debug(f"2D output shape: {output.shape}")
            
            logger.debug(f"BilateralFilter output - Device: {output.device}, Shape: {output.shape}")
            return output
            
        except Exception as e:
            logger.error(f"Error in bilateral filter forward pass: {str(e)}")
            raise


class DenoisingPipeline(nn.Module):
    """Multi-stage denoising pipeline using bilateral filters.

    This class implements a sequential pipeline of bilateral filters for
    progressive image denoising. Each stage refines the output of the previous stage.

    Args:
        num_stages (int, optional): Number of bilateral filter stages. Defaults to 3.

    Attributes:
        stages (nn.ModuleList): List of bilateral filter stages.
    """
    def __init__(self, num_stages=1):
        super(DenoisingPipeline, self).__init__()
        self.stages = nn.ModuleList([BilateralFilter() for _ in range(num_stages)])
    
    def forward(self, x):
        """Forward pass through all denoising stages.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, [depth,] height, width).

        Returns:
            torch.Tensor: Denoised output tensor of same shape as input.

        Raises:
            TypeError: If input is not a torch.Tensor.
        """
        # Input validation and device handling
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")
        
        device = x.device
        logger.debug(f"DenoisingPipeline input - Device: {device}, Shape: {x.shape}")
        
        # Ensure model is on the same device as input
        self.to(device)
        
        for i, stage in enumerate(self.stages):
            stage.to(device)
            x = stage(x)
            logger.debug(f"Stage {i+1} output - Device: {x.device}, Shape: {x.shape}")
        
        return x


def prepare_image(image, device='cpu'):
    """Prepare image for model input by converting to appropriate tensor format.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image.
        device (str, optional): Device to place tensor on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Prepared tensor of shape (batch, channels, [depth,] height, width).

    Raises:
        TypeError: If image is neither numpy array nor torch tensor.
        ValueError: If image shape is not supported.
    """
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).float()
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Handle different input shapes
    if len(tensor.shape) == 2:  # Single grayscale image
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif len(tensor.shape) == 3:
        if tensor.shape[0] == 1:  # Single slice 3D (depth=1, H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, D=1, H, W)
        else:  # RGB image (H, W, C)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Move channels first and add batch dim
    else:
        raise ValueError(f"Unsupported image shape: {tensor.shape}")
    
    return tensor.to(device)


def restore_image(tensor):
    """Convert model output tensor back to numpy array in original format.

    Args:
        tensor (torch.Tensor): Model output tensor.

    Returns:
        np.ndarray: Restored image array.
    """
    if tensor.shape[1] == 1:  # Grayscale
        return tensor.squeeze().cpu().numpy()
    else:  # RGB
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def train(model, clean_image, noisy_image, device='cuda', epochs=20, lr=0.01):
    """Train the denoising model using clean and noisy image pairs.

    Args:
        model (nn.Module): The denoising model to train.
        clean_image (Union[np.ndarray, torch.Tensor]): Ground truth clean image.
        noisy_image (Union[np.ndarray, torch.Tensor]): Input noisy image.
        device (str, optional): Device to train on. Defaults to 'cuda'.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        nn.Module: Trained model.

    Raises:
        Exception: If training fails.
    """
    try:
        logger.info(f"Starting training with {epochs} epochs")
        
        # Input validation and device handling
        if not torch.cuda.is_available() and device == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        
        # Convert images to tensors and move to device
        clean = prepare_image(clean_image, device)
        noisy = prepare_image(noisy_image, device)
        
        # Debug logging
        logger.info(f"Clean tensor - Type: {type(clean)}, Device: {clean.device}, Shape: {clean.shape}")
        logger.info(f"Noisy tensor - Type: {type(noisy)}, Device: {noisy.device}, Shape: {noisy.shape}")
        
        # Move model to device and set to train mode
        model = model.to(device)
        for param in model.parameters():
            param.data = param.data.to(device)
        model.train()
        
        # Debug model device placement
        logger.info("Model device placement:")
        for name, param in model.named_parameters():
            logger.info(f"{name} - Device: {param.device}")
        
        # Optimizer with gradient clipping
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss().to(device)
        
        # Track metrics
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Debug tensor before forward pass
            logger.debug(f"Input tensor device before forward pass: {noisy.device}")
            
            # Forward pass
            output = model(noisy)
            
            # Debug tensor after forward pass
            logger.debug(f"Output tensor device after forward pass: {output.device}")
            
            # Calculate loss
            loss = criterion(output, clean)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
            
            # Update best loss
            if loss.item() < best_loss:
                best_loss = loss.item()
                logger.info(f"New best loss: {best_loss:.6f}")
        
        logger.info("Training completed")
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def main():
    # Set up logging configuration
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # Load and prepare data
    logger.info("Loading data...")
    use_rgb = False  # Set to False for grayscale
    use_3d = False   # Set to True for 3D processing
    
    if use_rgb:
        image = img_as_float(data.astronaut())  # RGB image
    else:
        base_image = img_as_float(data.camera())  # Grayscale image
        if use_3d:
            # Add a dimension to create a 3D volume (simulating a slice)
            image = np.expand_dims(base_image, axis=0)
            logger.info(f"Created 3D volume with shape: {image.shape}")
        else:
            image = base_image
    
    noise_level = 0.1
    noisy_image = np.clip(image + np.random.normal(0, noise_level, image.shape), 0, 1)
    
    # Create and train model
    model = DenoisingPipeline()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model = train(model, image, noisy_image, device=device, epochs=300)
    
    # Log model parameters
    logger.info("Model parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.data}")
    
    # Denoise full image
    with torch.no_grad():
        noisy_tensor = prepare_image(noisy_image).to(device)
        denoised = model(noisy_tensor)
        denoised = restore_image(denoised.cpu())
    
    def plot_results(original, noisy, denoised, save_path=None):
        """Plot and optionally save the results of image denoising.
        
        Args:
            original: Original clean image
            noisy: Noisy image
            denoised: Denoised image
            save_path: Optional path to save the plot. If None, only displays the plot.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Handle different image shapes
        def prepare_for_display(img):
            if len(img.shape) == 3 and img.shape[0] == 1:  # 3D with single channel/depth
                return img.squeeze(0)  # Remove the first dimension
            return img
        
        # Prepare images for display
        original = prepare_for_display(original)
        noisy = prepare_for_display(noisy)
        denoised = prepare_for_display(denoised)
        
        # Determine if image is grayscale or RGB
        is_grayscale = len(original.shape) == 2
        cmap = 'gray' if is_grayscale else None
        
        # Plot images
        axes[0].imshow(original, cmap=cmap)
        axes[0].set_title('Original')
        axes[1].imshow(noisy, cmap=cmap)
        axes[1].set_title('Noisy')
        axes[2].imshow(denoised, cmap=cmap)
        axes[2].set_title('Denoised')
        
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            # Create filename based on image type
            filename = 'GRAYSCALE.png' if is_grayscale else 'RGB.png'
            save_path = os.path.join(save_path, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved result image to: {save_path}")
        
        plt.show()
        
    
    # Call plot_results with the current directory as save_path
    plot_results(image, noisy_image, denoised, save_path='.')

if __name__ == '__main__':
    main()