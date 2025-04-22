import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from datasets import load_dataset
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 1. Load dataset
ds = load_dataset("instruction-tuning-sd/low-level-image-proc")["train"]

# 2. Resize and normalization
resize = transforms.Resize((1024, 1024))

to_input = transforms.Compose([
    resize,
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1]
])

to_gt = transforms.Compose([
    resize,
    transforms.ToTensor(),  # Keep in [0, 1]
])

# 3. Collate function
def collate(batch):
    input_tensors = torch.stack([to_input(x["input_image"]) for x in batch])
    gt_tensors = torch.stack([to_gt(x["ground_truth_image"]) for x in batch])
    input_pils = [x["input_image"] for x in batch]  # ControlNet needs PIL
    return {
        "input_tensor": input_tensors,
        "gt_tensor": gt_tensors,
        "input_pil": input_pils
    }

loader = DataLoader(ds, batch_size=2, collate_fn=collate)

# 4. Load ControlNet + SD 3.5 pipeline
controlnet = ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-blur",
    torch_dtype=torch.float16,
    device_map="auto"  # 🚀 fix for meta tensor issue
)


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    device_map="auto"  # 👈 important fix
)


pipe.enable_attention_slicing()
pipe.safety_checker = lambda images, **kwargs: (images, False)  # disable NSFW checker

# 5. MSE Loss
criterion = nn.MSELoss()
all_losses = []

# 6. Loop through dataset
for batch in loader:
    input_pil = batch["input_pil"]
    gt_tensor = batch["gt_tensor"].to("cuda", dtype=torch.float16)

    with torch.no_grad():
        result = pipe(
            prompt=[""] * len(input_pil),
            control_image=input_pil,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

    # Convert output images to tensor and normalize to [0,1]
    recon = torch.stack([
        transforms.ToTensor()(img).to("cuda", dtype=torch.float16)
        for img in result.images
    ])

    # If needed, normalize recon to [0, 1]
    if recon.max() > 1:
        recon = recon / 255.0

    # Compute loss
    loss = criterion(recon, gt_tensor)
    all_losses.append(loss.item())
    print(f"🧮 Batch Loss: {loss.item():.4f}")

# 7. Report average loss
avg_loss = sum(all_losses) / len(all_losses)
print(f"\n✅ Average MSE Reconstruction Loss: {avg_loss:.4f}")

