# evaluate_reconstruction.py

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange

# Make sure the EEGDataset class is accessible
# Option 1: If it's in the same directory or Python path
# from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
# Option 2: If you need to add the directory containing it to the path
import sys
# Add the directory containing your dataset script if needed:
# Or adjust the import below if the file name is different
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset


from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image # Keep this import if needed, though PIL.Image.open is often sufficient

# --- Model Definitions (Copied from your training script) ---

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class encoder_low_level(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(encoder_low_level, self).__init__()        
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_subjects)])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #self.loss_func = ClipLoss()
        self.dropout = nn.Dropout(0.5)

        # CNN upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),    # Keep size (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),    # Output shape (4, 64, 64)
        )


    def forward(self, x):
        # Apply subject-wise linear layer
        x = self.subject_wise_linear[0](x)  # Output shape: (batchsize, 63, 128)
        # Reshape to match the input size for the upsampler
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
        out = self.upsampler(x)  # Pass through the upsampler
        return out


# --- Evaluation Function ---

def evaluate_single_sample(args):
    """Loads models, processes a single sample, and shows/saves comparison."""

    # --- Device Setup ---
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Load VAE ---
    print("Loading VAE...")
    image_processor = VaeImageProcessor()
    # Use DiffusionPipeline to easily get the VAE
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, # Use float16 for memory efficiency if GPU supports it
        variant="fp16",
        use_safetensors=True
    )
    vae = pipe.vae.to(device).eval()
    # Ensure VAE doesn't calculate gradients
    vae.requires_grad_(False)
    del pipe # Free up memory
    torch.cuda.empty_cache()
    print("VAE loaded.")

    # --- Load Trained EEG Encoder ---
    print(f"Loading trained EEG encoder from: {args.eeg_encoder_weights}")
    # Instantiate the correct encoder model class
    # Assuming encoder_low_level was used, adjust if needed
    eeg_model = encoder_low_level() # Add necessary arguments if constructor requires them
    
    # Load the saved state dictionary
    checkpoint = torch.load(args.eeg_encoder_weights, map_location=device)
    
    # Handle potential keys mismatch (e.g., if saved with DataParallel)
    # Or if the state dict is nested (common if saving optimizer state too)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if saved using DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        eeg_model.load_state_dict(state_dict, strict=True) # Use strict=False if minor mismatches are expected
    except RuntimeError as e:
        print(f"Warning: Potential mismatch loading state dict: {e}")
        print("Attempting load with strict=False")
        eeg_model.load_state_dict(state_dict, strict=False)

    eeg_model.to(device)
    eeg_model.eval() # Set model to evaluation mode (disables dropout etc.)
    print("EEG Encoder loaded and set to evaluation mode.")

    # --- Load Dataset and Specific Sample ---
    print(f"Loading dataset for subject {args.subject_id}, accessing test sample index {args.sample_index}")
    # Load the *test* dataset for the specified subject
    try:
        test_dataset = EEGDataset(args.data_path, subjects=[args.subject_id], train=False)
    except ImportError:
         print("Error: Could not import EEGDataset. Check path and script name.")
         sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Data path not found: {args.data_path}")
        sys.exit(1)

    if args.sample_index >= len(test_dataset):
        print(f"Error: Sample index {args.sample_index} is out of bounds for test dataset size {len(test_dataset)}")
        sys.exit(1)

    # Get the specific sample (EEG data and the path to the original image)
    # Adjust indices based on what EEGDataset returns:
    # Example: eeg_data, label, text, text_features, img_path, img_features
    eeg_data, label, _, _, img_path, _ = test_dataset[args.sample_index]
    print(f"Loaded sample: Label {label}, Image Path {img_path}")

    # --- Prepare EEG Data ---
    # Add batch dimension (B, C, T), move to device, ensure correct type and sequence length
    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)
    # Select the sequence length the model was trained on (e.g., first 250 points)
    eeg_input = eeg_data_tensor[:, :, :250] # Adjust 250 if your model used a different length
    print(f"EEG Input Tensor shape: {eeg_input.shape}")


    # --- Generate Latent Representation ---
    print("Generating latent representation from EEG...")
    with torch.no_grad(): # Ensure no gradients are calculated
        latent_z = eeg_model(eeg_input).float() # Ensure output is float for VAE
    print(f"Generated Latent shape: {latent_z.shape}") # Should be like (1, 4, 64, 64) for SDXL VAE

    # --- Reconstruct Image from Latent ---
    print("Reconstructing image using VAE decoder...")
    with torch.no_grad():
        # VAE expects latent in float32, scale if needed (SDXL VAE usually doesn't need scaling factor)
        # latent_z = latent_z / vae.config.scaling_factor # Example if scaling needed
        reconstructed_output = vae.decode(latent_z.to(vae.dtype)).sample # Use vae.dtype for potential float16
        
    # Post-process the output tensor to a PIL image
    reconstructed_image = image_processor.postprocess(reconstructed_output.float(), output_type='pil')[0] # Use float() before postprocessing
    print("Image reconstructed.")

    # --- Load Original Image ---
    print(f"Loading original image from: {img_path}")
    try:
        original_image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Original image file not found at {img_path}")
        return # Exit if original image can't be loaded

    # --- Compare and Display/Save ---
    print("Displaying comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image\n(Label: {label})")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f"Reconstructed from EEG\n(Sample Index: {args.sample_index})")
    axes[1].axis('off')

    plt.tight_layout()

    # Save the comparison figure
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        plt.savefig(args.output_path)
        print(f"Comparison saved to: {args.output_path}")

    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EEG to Image Reconstruction')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the root EEG dataset directory.')
    parser.add_argument('--eeg_encoder_weights', type=str, required=True, help='Path to the trained EEG encoder .pth weights file.')
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID (e.g., sub-01) to load data for.')
    parser.add_argument('--sample_index', type=int, default=0, help='Index of the sample within the *test* dataset to evaluate.')
    # parser.add_argument('--encoder_type', type=str, default='encoder_low_level', help='Specify the encoder class name (must match the loaded weights).') # Currently hardcoded to encoder_low_level
    parser.add_argument('--output_path', type=str, default=None, help='Optional path to save the comparison image (e.g., output/comparison.png).')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if device is gpu.')

    args = parser.parse_args()

    # --- Basic Checks ---
    if not os.path.exists(args.eeg_encoder_weights):
        print(f"Error: Weights file not found at {args.eeg_encoder_weights}")
        sys.exit(1)

    evaluate_single_sample(args)