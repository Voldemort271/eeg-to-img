import os
import torch
from transformers import pipeline
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler
)
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from pytorch_msssim import ms_ssim
import lpips
import torchvision.transforms.functional as TF

# -----------------------------
# 1. Setup pipelines/models
# -----------------------------
# Depth estimation
depth_estimator = pipeline(
    "depth-estimation",
    model="Intel/dpt-large",
    torch_dtype=torch.float16,
    device=0
)

# Captioning (BLIP) using image-to-text
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16,
    device=0
)

from transformers import CLIPProcessor, CLIPModel




# ControlNet-guided Img2Img
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)
pipe_with_depth = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe_with_depth.scheduler = UniPCMultistepScheduler.from_config(pipe_with_depth.scheduler.config)
pipe_with_depth.enable_xformers_memory_efficient_attention()
pipe_with_depth.enable_model_cpu_offload()
pipe_with_depth.to("cuda")

# Standard Img2Img (no depth)
pipe_without_depth = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe_without_depth.scheduler = UniPCMultistepScheduler.from_config(pipe_without_depth.scheduler.config)
pipe_without_depth.enable_xformers_memory_efficient_attention()
pipe_without_depth.enable_model_cpu_offload()
pipe_without_depth.to("cuda")

# LPIPS model
lpips_fn = lpips.LPIPS(net="alex").to("cuda")

# -----------------------------
# 2. Metrics calculation
# -----------------------------

def calculate_metrics(orig: Image.Image, enhanced: Image.Image):
    orig_np = np.array(orig).astype(np.float32)
    enh_np  = np.array(enhanced).astype(np.float32)
    _psnr   = psnr(orig_np, enh_np, data_range=255)
    _ssim   = ssim(orig_np, enh_np, data_range=255, channel_axis=-1, win_size=7)
    orig_t  = TF.to_tensor(orig).unsqueeze(0).to("cuda")
    enh_t   = TF.to_tensor(enhanced).unsqueeze(0).to("cuda")
    _mssim  = ms_ssim(orig_t, enh_t, data_range=1.0, size_average=True).item()
    orig_lp = orig_t * 2 - 1
    enh_lp  = enh_t  * 2 - 1
    _lpips  = lpips_fn(orig_lp, enh_lp).item()
    return {
        "PSNR":    _psnr,
        "SSIM":    _ssim,
        "MS-SSIM": _mssim,
        "LPIPS":   _lpips
    }

# -----------------------------
# 3. Batch processing
# -----------------------------

input_dir = "output/images/04-20_03-30_vae_train_imgs/epoch_99"
output_dir = "output/images/results_epoch99_embeds_test"
with_depth_dir = os.path.join(output_dir, "with_depth")
without_depth_dir = os.path.join(output_dir, "without_depth")
depth_map_dir = os.path.join(output_dir, "depth_maps")
os.makedirs(with_depth_dir, exist_ok=True)
os.makedirs(without_depth_dir, exist_ok=True)
os.makedirs(depth_map_dir, exist_ok=True)

metrics_accum = {
    "with_depth": {"PSNR":0, "SSIM":0, "MS-SSIM":0, "LPIPS":0},
    "without":   {"PSNR":0, "SSIM":0, "MS-SSIM":0, "LPIPS":0},
    "with_depth_and_embedds":   {"PSNR":0, "SSIM":0, "MS-SSIM":0, "LPIPS":0}
}
count = 0

for fname in os.listdir(input_dir):
    if not fname.startswith("train_image_") or not fname.endswith(".png"):
        continue
    idx = fname.split("train_image_")[-1].split(".png")[0]
    orig_path = os.path.join(input_dir, f"train_image_{idx}.png")
    blur_path = os.path.join(input_dir, f"reconstructed_image_{idx}.png")
    if not os.path.exists(blur_path):
        continue

    orig = Image.open(orig_path).convert("RGB").resize((512,512))
    blur = Image.open(blur_path).convert("RGB").resize((512,512))

    # Generate prompt and print it
    caption_out = captioner(orig)[0]
    prompt = caption_out.get("generated_text", caption_out.get("text", ""))
    print(f"Image {idx} prompt: {prompt}")

    # Build and save depth map
    depth_map = depth_estimator(orig)["depth"]
    d_np = np.array(depth_map)
    d_np = ((d_np - d_np.min())/(d_np.max()-d_np.min())*255).astype(np.uint8)
    depth_img = Image.fromarray(np.stack([d_np]*3, axis=-1)).resize((512,512))
    depth_img.save(os.path.join(depth_map_dir, f"{idx}_depth.png"))

    # Enhance with depth
    out_d = pipe_with_depth(
        prompt=prompt,
        image=blur,
        control_image=depth_img,
        num_inference_steps=20
    ).images[0]
    out_d.save(os.path.join(with_depth_dir, f"{idx}_with_depth.png"))
    m_d = calculate_metrics(orig, out_d)
    for k,v in m_d.items(): metrics_accum["with_depth"][k] += v

    # Initialize the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Preprocess the original image
    inputs = clip_processor(images=blur, return_tensors="pt", do_resize=True, size=512)

    # Generate the image embedding
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)


    # Enhance with depth and image embeddings
    out_ed = pipe_with_depth(
        prompt=prompt,
        image=blur,
        control_image=depth_img,
        prompt_embeds=image_embedding,
        num_inference_steps=20
    ).images[0]
    out_d.save(os.path.join(with_depth_dir, f"{idx}_with_embeddings.png"))
    m_d = calculate_metrics(orig, out_d)
    for k,v in m_d.items(): metrics_accum["with_depth_embedds"][k] += v


    # Enhance without depth
    out_nd = pipe_without_depth(
        prompt=prompt,
        image=blur,
        num_inference_steps=20
    ).images[0]
    out_nd.save(os.path.join(without_depth_dir, f"{idx}_without_depth.png"))
    m_nd = calculate_metrics(orig, out_nd)
    for k,v in m_nd.items(): metrics_accum["without"][k] += v

    count += 1
    print(f"Processed {idx}")

# -----------------------------
# 4. Final aggregate metrics
# -----------------------------

avg_with = {k: v / count for k, v in metrics_accum["with_depth"].items()}
avg_without = {k: v / count for k, v in metrics_accum["without"].items()}
avg_embeds = {k: v / count for k, v in metrics_accum["with_depth_embedds"].items()}

print(f"\nProcessed {count} image pairs.")
print("-- With Depth --")
for k,v in avg_with.items(): print(f"{k:7s}: {v:.4f}")
print("-- Without Depth --")
for k,v in avg_without.items(): print(f"{k:7s}: {v:.4f}")
print("-- Withdepthandembeds --")
for k,v in avg_embeds.items(): print(f"{k:7s}: {v:.4f}")
