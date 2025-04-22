# from datasets import load_dataset


# ds = load_dataset("instruction-tuning-sd/low-level-image-proc")

# print(ds)

from datasets import load_dataset
from PIL import Image
import torch
from torchvision import transforms
import math
from safetensors.torch import load_file


import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch.nn.functional as F
# 1. Load the split you need
ds = load_dataset("instruction-tuning-sd/low-level-image-proc")["train"]


to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024)),            # model expects 1024×1024
    transforms.ToTensor(),                      # [0,1], C×H×W
])



# # 2. Define a transform to convert PIL→tensor
# to_tensor = transforms.Compose([
#     transforms.ToTensor(),                # [0,1], C×H×W
# ])

# # 3. Wrap into a DataLoader
# from torch.utils.data import DataLoader

# def preprocess(example):
#     example["input_tensor"] = to_tensor(example["input_image"])
#     example["gt_tensor"]    = to_tensor(example["ground_truth_image"])
#     return example

# ds = ds.map(preprocess, remove_columns=["instruction","input_image","ground_truth_image"])
# loader = DataLoader(ds, batch_size=4, shuffle=False)

from diffusers import StableDiffusion3Pipeline

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


# Load the ControlNet model
# controlnet = ControlNetModel.from_pretrained(
#     "stabilityai/stable-diffusion-3.5-large-controlnet-blur", torch_dtype=torch.float16
# )

# # Load other necessary components without specifying subfolders
# vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
# text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
# tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3.5-large")
# unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
# scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-3.5-large")
# feature_extractor = CLIPFeatureExtractor.from_pretrained("stabilityai/stable-diffusion-3.5-large")
# safety_checker = StableDiffusionSafetyChecker.from_pretrained("stabilityai/stable-diffusion-3.5-large")

# # Initialize the pipeline
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3.5-large",
#     subfolder="transformer",
#     torch_dtype=torch.float16
# ).to("cuda")

from diffusers import StableDiffusion3Pipeline
import torch

# Load the pre-trained Stable Diffusion 3.5 Large model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16
).to("cuda")



from tqdm.auto import tqdm

recons = []   # store reconstructed tensors
gts    = []   # store ground‑truth tensors

for batch in tqdm(loader):
    # input_blur: B×C×H×W
    input_blur = batch["input_tensor"].to("cuda", dtype=torch.float16)
    gt         = batch["gt_tensor"].to("cuda", dtype=torch.float16)

    # ControlNet expects images in PIL format or tensors 0–1
    # Here we pass tensors directly as `image` argument:
    outputs = pipe(
        prompt=[""] * input_blur.shape[0],    # empty prompt
        image=input_blur,                     # guiding blur map
        num_inference_steps=30,
        guidance_scale=7.5,
    )
    # outputs.images is a list of PIL.Images
    # convert back to tensor:
    recon_batch = torch.stack([
        to_tensor(img).to("cuda", dtype=torch.float16)
        for img in outputs.images
    ], dim=0)

    recons.append(recon_batch)
    gts.append(gt)

# Concatenate all
recons = torch.cat(recons, dim=0)
gts    = torch.cat(gts,    dim=0)

psnrs = []
ssims = []

# Move to CPU + numpy for skimage
recons_np = (recons.cpu().numpy() * 255).astype(np.uint8)  # N×C×H×W
gts_np    = (gts.cpu().numpy()    * 255).astype(np.uint8)

for recon, gt in zip(recons_np, gts_np):
    # reshape to H×W×C
    recon = np.transpose(recon, (1,2,0))
    gt    = np.transpose(gt,    (1,2,0))

    # PSNR
    psnr = peak_signal_noise_ratio(gt, recon, data_range=255)
    psnrs.append(psnr)

    # SSIM (multichannel)
    ssim = structural_similarity(gt, recon, multichannel=True, data_range=255)
    ssims.append(ssim)

print(f"Average PSNR: {np.mean(psnrs):.2f} dB")
print(f"Average SSIM: {np.mean(ssims):.4f}")



idxs = [0, 10, 50]
for i in idxs:
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(np.transpose(gts_np[i], (1,2,0))); axs[0].set_title("Ground Truth"); axs[0].axis("off")
    axs[1].imshow(np.transpose((recons_np[i]), (1,2,0))); axs[1].set_title("Reconstruction"); axs[1].axis("off")
    axs[2].imshow(np.transpose((recons_np[i]-gts_np[i]), (1,2,0))); axs[2].set_title("Error Map"); axs[2].axis("off")
    plt.show()
