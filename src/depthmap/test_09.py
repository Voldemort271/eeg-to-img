# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torch.nn as nn
# from datasets import load_dataset
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from transformers import pipeline
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from PIL import Image
# import numpy as np
# from diffusers.utils import load_image
# import io

# # 1. Load dataset
# from datasets import load_dataset

# # 1. Load the dataset
# ds = load_dataset("instruction-tuning-sd/low-level-image-proc")["train"]

# # 2. Grab the first sample
# sample = ds[0]

# # 3. It may already be a PIL Image:
# image_input = sample["input_image"]

# # 4. (Optional) display
# # print(type(image_input))


# # load the depth estimator model
# depth_estimator = pipeline('depth-estimation')
# # load the controlnet model for depth estimation
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
# # load the stable diffusion pipeline with controlnet
# pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # enable efficient implementations using xformers for faster inference
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()


# # get depth estimates
# image_depth = depth_estimator(image_input)['depth']
# # convert to PIL image format
# image_depth = np.array(image_depth)
# image_depth = image_depth[:, :, None]
# image_depth = np.concatenate([image_depth, image_depth, image_depth], axis=2)
# image_depth = Image.fromarray(image_depth)
# # image_depth


# image_output = pipe("", image_depth, num_inference_steps=20).images[0]
# # image_output

# image_input.save("original_input.png")
# image_output.save("generated_output.png")
# image_depth.save("depth.png")
# print("Saved original input as original_input.png")
# print("Saved generated output as generated_output.png")
# print("dept.png")

# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoImageProcessor,
#     DepthAnythingForDepthEstimation
# )
# from diffusers import (
#     ControlNetModel,
#     StableDiffusionControlNetPipeline,
#     UniPCMultistepScheduler
# )
# from PIL import Image
# import numpy as np

# # 1. Load data
# ds = load_dataset("instruction-tuning-sd/low-level-image-proc")["train"]
# sample = ds[0]
# image_input    = sample["input_image"].convert("RGB")
# image_to_depth = sample["ground_truth_image"].convert("RGB")

# # 2. Load processor + model for Depth Anything (float16)
# processor = AutoImageProcessor.from_pretrained(
#     "depth-anything/Depth-Anything-V2-Large-hf",
#     torch_dtype=torch.float16
# )
# model = DepthAnythingForDepthEstimation.from_pretrained(
#     "depth-anything/Depth-Anything-V2-Large-hf",
#     torch_dtype=torch.float16
# ).to("cuda").eval()

# # 3. Preprocess: PIL → float16 tensor on GPU
# inputs = processor(
#     images=image_to_depth,
#     return_tensors="pt"
# ).pixel_values  # this will be float32 by default

# # cast to float16 and move to GPU
# inputs = inputs.to(dtype=torch.float16, device="cuda")



# # 4. Inference
# with torch.no_grad():
#     outputs = model(inputs)            # model returns a dict
#     depth_logits = outputs.predicted_depth  # shape (1, 1, H, W), float16

# # 5. Convert to numpy H×W
# depth_arr = depth_logits.squeeze().cpu().numpy().astype(np.float32)

# # 6. Normalize & convert to 3‑channel PIL
# depth_norm = (255 * (depth_arr - depth_arr.min()) / (depth_arr.ptp() + 1e-8)).astype(np.uint8)
# depth_rgb  = np.stack([depth_norm]*3, axis=-1)
# image_depth = Image.fromarray(depth_rgb)






# # 7. Build ControlNet+SD (same as before)
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-depth",
#     torch_dtype=torch.float16
# )
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     controlnet=controlnet,
#     safety_checker=None,
#     torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

# # 8. Generate
# image_output = pipe(
#     prompt="",
#     image=image_depth,
#     num_inference_steps=20
# ).images[0]


# # 9. Saveimport torch
# from datasets import load_dataset
# from transformers import (
#     AutoImageProcessor,
#     DepthAnythingForDepthEstimation
# )
# from diffusers import (
#     ControlNetModel,
#     StableDiffusionControlNetPipeline,
#     UniPCMultistepScheduler
# )
# from PIL import Image
# import numpy as np

# # 1. Load data
# ds = load_dataset("instruction-tuning-sd/low-level-image-proc")["train"]
# sample = ds[0]
# image_input    = sample["input_image"].convert("RGB")
# image_to_depth = sample["ground_truth_image"].convert("RGB")

# # 2. Load processor + model for Depth Anything (float16)
# processor = AutoImageProcessor.from_pretrained(
#     "depth-anything/Depth-Anything-V2-Large-hf",
#     torch_dtype=torch.float16
# )
# model = DepthAnythingForDepthEstimation.from_pretrained(
#     "depth-anything/Depth-Anything-V2-Large-hf",
#     torch_dtype=torch.float16
# ).to("cuda").eval()

# # 3. Preprocess: PIL → float16 tensor on GPU
# inputs = processor(
#     images=image_to_depth,
#     return_tensors="pt"
# ).pixel_values  # this will be float32 by default

# # cast to float16 and move to GPU
# inputs = inputs.to(dtype=torch.float16, device="cuda")



# # 4. Inference
# with torch.no_grad():
#     outputs = model(inputs)            # model returns a dict
#     depth_logits = outputs.predicted_depth  # shape (1, 1, H, W), float16

# # 5. Convert to numpy H×W
# depth_arr = depth_logits.squeeze().cpu().numpy().astype(np.float32)

# # 6. Normalize & convert to 3‑channel PIL
# depth_norm = (255 * (depth_arr - depth_arr.min()) / (depth_arr.ptp() + 1e-8)).astype(np.uint8)
# depth_rgb  = np.stack([depth_norm]*3, axis=-1)
# image_depth = Image.fromarray(depth_rgb)






# # 7. Build ControlNet+SD (same as before)
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-depth",
#     torch_dtype=torch.float16
# )
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     controlnet=controlnet,
#     safety_checker=None,
#     torch_dtype=torch.float16
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

# # 8. Generate
# image_output = pipe(
#     prompt="",
#     image=image_depth,
#     num_inference_steps=20
# ).images[0]


# # 9. Save
# image_input.save("original_input.png")
# image_to_depth.save("org.png")
# image_depth.save("depth.png")
# image_output.save("generated_output.png")

# print("Saved original_input.png, depth.png, generated_output.png")


# image_output.save("generated_output.png")

# print("Saved original_input.png, depth.png, generated_output.png")

import torch
from datasets import load_dataset
from transformers import pipeline
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler
)
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Load dataset
ds = load_dataset("icewiny/blurred_image_coyo_1M")["train"]

# Load depth estimation pipeline
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

# Load ControlNet and Stable Diffusion pipeline
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Evaluation metrics
total_psnr = 0.0
total_ssim = 0.0
num_samples = 50  # You can increase this

for i in tqdm(range(num_samples), desc="Evaluating"):
    try:
        sample = ds[i]
        original_image = sample["image"].convert("RGB")
        blurred_input = sample["blurred_img"].convert("RGB")

        # Resize for consistency
        original_image = original_image.resize((512, 512))
        blurred_input = blurred_input.resize((512, 512))

        # Depth estimation from original image
        depth = depth_estimator(original_image)['depth']
        depth = np.array(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype(np.uint8)
        depth_image = Image.fromarray(np.stack([depth]*3, axis=-1))

        # ✅ Apply blur to the depth map
        # blurred_depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=6))

        # Generate enhanced image using blurred depth as control image
        result = pipe(
            prompt="Enhance the image quality",
            image=blurred_input,
            control_image=depth_image,
            num_inference_steps=20
        ).images[0]

        # Resize generated to match original
        result = result.resize(original_image.size)

        # Convert to numpy
        orig_np = np.array(original_image).astype(np.float32)
        result_np = np.array(result).astype(np.float32)

        # Metrics
        psnr_val = psnr(orig_np, result_np, data_range=255)
        ssim_val = ssim(orig_np, result_np, data_range=255, channel_axis=-1, win_size=7)

        total_psnr += psnr_val
        total_ssim += ssim_val

    except Exception as e:
        print(f"Error on sample {i}: {e}")
        continue

# Final results
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples

print(f"\nEvaluated {num_samples} samples")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")
