# evaluate_reconstruction.py

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
import sys
import warnings

# Make sure the EEGDataset class is accessible
try:
    # Adjust the import path if your dataset script is elsewhere
    from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
except ImportError:
    print("ERROR: Could not import EEGDataset.")
    print("Please ensure 'eegdatasets_leaveone_latent_vae_no_average.py' is in the Python path or the current directory.")
    sys.exit(1)


from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor

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
    # Define the scaling factor for SDXL VAE
    vae_scaling_factor = 0.18215 # Standard for SDXL

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scaling_factor) # Pass scaling factor here too for consistency

    # Use DiffusionPipeline to easily get the VAE
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.bfloat16, # Use float16 for memory efficiency if GPU supports it
            variant="fp16",
            use_safetensors=True
        )
        vae = pipe.vae.to(device)
        vae.eval()
        # Ensure VAE doesn't calculate gradients
        vae.requires_grad_(False)
        # Verify the loaded VAE has the expected scaling factor config if available
        if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor is not None:
             print(f"Loaded VAE scaling factor: {vae.config.scaling_factor}")
             # Use the VAE's specific scaling factor if available and different
             # vae_scaling_factor = vae.config.scaling_factor # Uncomment if you want to use the VAE's value strictly
        else:
            print(f"Using default SDXL VAE scaling factor: {vae_scaling_factor}")

        del pipe # Free up memory
        torch.cuda.empty_cache()
        print("VAE loaded.")
    except Exception as e:
        print(f"Error loading diffusion pipeline or VAE: {e}")
        sys.exit(1)


    # --- Load Trained EEG Encoder ---
    print(f"Loading trained EEG encoder from: {args.eeg_encoder_weights}")
    eeg_model = encoder_low_level() # Add necessary arguments if constructor requires them

    try:
        checkpoint = torch.load(args.eeg_encoder_weights, map_location=device)
        # Handle potential nested state dicts or 'module.' prefix
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = eeg_model.load_state_dict(state_dict, strict=False) # Load leniently first to check
        if missing_keys:
            print(f"Warning: Missing keys when loading EEG model: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading EEG model: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("EEG model weights loaded successfully (strict match).")
        else:
             print("EEG model weights loaded (with some mismatches handled by strict=False).")


    except Exception as e:
        print(f"Error loading EEG encoder weights from {args.eeg_encoder_weights}: {e}")
        sys.exit(1)

    eeg_model.to(device)
    eeg_model.eval()
    print("EEG Encoder set to evaluation mode.")

    # --- Load Dataset and Specific Sample ---
    print(f"Loading dataset for subject {args.subject_id}, accessing test sample index {args.sample_index}")
    try:
        # Ensure you are loading the TEST dataset
        test_dataset = EEGDataset(args.data_path, subjects=[args.subject_id], train=False) # Humne true tha
        print(f"Test dataset loaded. Size: {len(test_dataset)}")
    except FileNotFoundError:
        print(f"Error: Data path not found: {args.data_path}")
        sys.exit(1)
    except Exception as e:
         print(f"Error initializing EEGDataset: {e}")
         sys.exit(1)

    if args.sample_index >= len(test_dataset):
        print(f"Error: Sample index {args.sample_index} is out of bounds for test dataset size {len(test_dataset)}")
        sys.exit(1)

    # Get the specific sample
    try:
        # Adjust indices based on what EEGDataset returns
        eeg_data, label, _, _, img_path, _ = test_dataset[args.sample_index]
        print(f"Loaded sample: Label {label}, Image Path {img_path}")
    except Exception as e:
        print(f"Error accessing sample index {args.sample_index} from dataset: {e}")
        sys.exit(1)

    # --- Prepare EEG Data ---
    with warnings.catch_warnings(): # Suppress the specific UserWarning about torch.tensor
        warnings.simplefilter("ignore", UserWarning)
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)
    # Select the sequence length the model was trained on (e.g., first 250 points)
    seq_len = 250 # Make sure this matches your training config
    eeg_input = eeg_data_tensor[:, :, :seq_len]
    print(f"EEG Input Tensor shape: {eeg_input.shape}") # Should be [1, 63, 250]

    # --- Generate Latent Representation ---
    print("Generating latent representation from EEG...")
    with torch.no_grad(): # Ensure no gradients are calculated
        latent_z_scaled = eeg_model(eeg_input).float() # This is the SCALED latent
 
    print(f"Generated SCALED Latent shape: {latent_z_scaled.shape}") # Should be [1, 4, 64, 64]

    # --- !!!!! THE FIX: Unscale the latent !!!!! ---
    vae_scaling_factor = 1.0 #humne 1.0 set kiya
    latent_z_unscaled = latent_z_scaled / vae_scaling_factor
    print(f"UNSCALED Latent shape: {latent_z_unscaled.shape}")

    # --- Debug: Check Latent Values ---
    print(f"  Scaled Latent Stats: Min={latent_z_scaled.min():.4f}, Max={latent_z_scaled.max():.4f}, Mean={latent_z_scaled.mean():.4f}, Std={latent_z_scaled.std():.4f}")
    print(f"Unscaled Latent Stats: Min={latent_z_unscaled.min():.4f}, Max={latent_z_unscaled.max():.4f}, Mean={latent_z_unscaled.mean():.4f}, Std={latent_z_unscaled.std():.4f}")
    if torch.isnan(latent_z_unscaled).any():
        print("ERROR: NaNs detected in unscaled latent vector!")
        # return # Optionally stop here if NaNs are found
    elif torch.isinf(latent_z_unscaled).any():
        print("ERROR: Infs detected in unscaled latent vector!")
        # return

    # --- Reconstruct Image from UNSCALED Latent ---
    print("Reconstructing image using VAE decoder...")
    try:
        with torch.no_grad():
            # Feed the UNSCALED latent to the decoder
            # Ensure dtype matches VAE (often float16 on GPU, float32 on CPU)
            reconstructed_output = vae.decode(latent_z_unscaled.to(vae.dtype)).sample

        print(reconstructed_output)
        # Post-process the output tensor to a PIL image
        # Process with float32 for stability before converting to PIL
        reconstructed_image = image_processor.postprocess(reconstructed_output.float(), output_type='pil')[0]

        print("Image reconstructed.")
    except Exception as e:
        print(f"Error during VAE decoding or postprocessing: {e}")
        # Often helps to see the state of the tensor causing the error
        print(f"  Output tensor before postprocessing stats: Min={reconstructed_output.min():.4f}, Max={reconstructed_output.max():.4f}, Mean={reconstructed_output.mean():.4f}")
        print(f"  Contains NaNs: {torch.isnan(reconstructed_output).any()}, Contains Infs: {torch.isinf(reconstructed_output).any()}")
        # If you get the 'invalid value encountered in cast' here, it means NaNs/Infs likely came from decode
        sys.exit(1)


    # --- Load Original Image ---
    print(f"Loading original image from: {img_path}")
    try:
        original_image = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Original image file not found at {img_path}")
        return
    except Exception as e:
        print(f"Error opening original image {img_path}: {e}")
        return

    # --- Compare and Display/Save ---
    print("Displaying comparison...")
    try:
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
            try:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                plt.savefig(args.output_path)
                print(f"Comparison saved to: {args.output_path}")
            except Exception as e:
                print(f"Error saving comparison image to {args.output_path}: {e}")

        # Try showing the plot
        plt.show()

    except Exception as e:
        print(f"Error during plotting or saving figure: {e}")


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EEG to Image Reconstruction')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the root EEG dataset directory.')
    parser.add_argument('--eeg_encoder_weights', type=str, required=True, help='Path to the trained EEG encoder .pth weights file.')
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID (e.g., sub-08) to load data for.')
    parser.add_argument('--sample_index', type=int, default=0, help='Index of the sample within the *test* dataset to evaluate.')
    parser.add_argument('--output_path', type=str, default=None, help='Optional path to save the comparison image (e.g., output/comparison.png).')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if device is gpu.')

    args = parser.parse_args()

    # --- Basic Checks ---
    if not os.path.exists(args.eeg_encoder_weights):
        print(f"Error: Weights file not found at {args.eeg_encoder_weights}")
        sys.exit(1)
    if not os.path.isdir(args.data_path):
        print(f"Error: Data path directory not found at {args.data_path}")
        sys.exit(1)


    evaluate_single_sample(args)