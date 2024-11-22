# Trainable Bilateral Filter for Image Denoising

A pure PyTorch implementation of trainable bilateral filter for image denoising, based on the paper published in Medical Physics: [Ultralow-parameter denoising: Trainable bilateral filter layers in computed tomography](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15718)

## Example Results

![GRAYSCALE](GRAYSCALE.png)
![RGB](RGB.png)

## Additional Features compare to the official implementation

- Flexible processing of 2D and 3D images
- Support for both grayscale and RGB images
- Cached kernel computations for improved performance
- Multi-stage denoising pipeline

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sypsyp97/trainable_bilateral_filter_torch.git
cd trainable_bilateral_filter_torch
```

2. Install the required dependencies:

```bash
pip install torch numpy matplotlib scikit-image loguru tqdm
```

## Usage

### Basic Usage

```python
import torch
from bilateral_filter import DenoisingPipeline

# Initialize the model
model = DenoisingPipeline(num_stages=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
trained_model = train(model, clean_image, noisy_image, device=device)

# Denoise new images
with torch.no_grad():
    denoised = model(noisy_tensor)
```

## Acknowledgments

This implementation is based on the paper [Ultralow-parameter denoising: Trainable bilateral filter layers in computed tomography](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15718) and acknowledges the official implementation available at: [https://github.com/faebstn96/trainable-bilateral-filter-source](https://github.com/faebstn96/trainable-bilateral-filter-source)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wagner2022ultra,
  title={Ultralow-parameter denoising: Trainable bilateral filter layers in computed tomography},
  author={Wagner, Fabian and Thies, Mareike and Gu, Mingxuan and Huang, Yixing and Pechmann, Sabrina and Patwari, Mayank and Ploner, Stefan and Aust, Oliver and Uderhardt, Stefan and Schett, Georg and Christiansen, Silke and Maier, Andreas},
  journal={Medical Physics},
  volume={49},
  number={8},
  pages={5107-5120},
  year={2022},
  doi={https://doi.org/10.1002/mp.15718}
}
