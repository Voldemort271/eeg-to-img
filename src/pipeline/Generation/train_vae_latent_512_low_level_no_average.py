import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pdb
os.environ["WANDB_API_KEY"] = "ca7c8c2f0c62fceea4c8b6a1395a53048508eaa0"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger

import csv
from torch import Tensor
import itertools
import math

from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding_inverted
import numpy as np

from diffusers.utils import load_image
from IPython.display import display
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import datetime
import itertools
import csv
from torch.nn import MSELoss
from ATMS_reconstruction_output_h_a import EEGWithEmbedding, load_atms_model



torch.cuda.empty_cache()
torch.cuda.ipc_collect()

image_processor = VaeImageProcessor()
# path = "stabilityai/stable-diffusion-xl-base-1.0"

# vae = AutoencoderKL.from_pretrained(path, subfolder='vae').to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")


if hasattr(pipe, 'vae'):
    for param in pipe.vae.parameters():
        param.requires_grad = False

vae = pipe.vae.to(device)
vae.requires_grad_(False)
vae.eval()

class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                      # Sequence length
        self.pred_len = 250                     # Prediction length
        self.output_attention = False          # Whether to output attention weights
        self.d_model = 250                     # Model dimension
        self.embed = 'timeF'                   # Time encoding method
        self.freq = 'h'                        # Time frequency
        self.dropout = 0.25                    # Dropout rate
        self.factor = 1                        # Attention scaling factor
        self.n_heads = 4                       # Number of attention heads
        self.e_layers = 3                     # Number of encoder layers
        self.d_ff = 256                       # Feed-forward network dimension
        self.activation = 'gelu'               # Activation function
        self.enc_in = 63                        # Encoder input dimension (example value)
    

class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
    def forward(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            # PatchEmbedding(emb_size),
            # FlattenHead()
        )

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(            
            nn.Linear(250, proj_dim),
            Rearrange('B C L->B L C'),
            nn.Linear(63, 16),
            Rearrange('B L C->B C L'),
            nn.Dropout(drop_proj),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

# Change the loss function to MAE
from loss import ClipLoss
clip_loss = ClipLoss()

import torch
import torch.nn as nn
import numpy as np
# class encoder_low_level(nn.Module):
#     def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
#         super(encoder_low_level, self).__init__()        
#         self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_subjects)])
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.loss_func = ClipLoss()
#         self.dropout = nn.Dropout(0.5)

#         # CNN upsampler
#         self.upsampler = nn.Sequential(
#             nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),    # Keep size (64, 64)
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),    # Output shape (4, 64, 64)
#         )


#     def forward(self, x):
#         # Apply subject-wise linear layer
#         x = self.subject_wise_linear[0](x)  # Output shape: (batchsize, 63, 128)
#         # Reshape to match the input size for the upsampler
#         x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
#         out = self.upsampler(x)  # Pass through the upsampler
#         return out


class encoder_low_level(nn.Module):
    def __init__(self, num_latents=1024, num_channels=63, latent_dim=128):
        super().__init__()
        # project 1024D EEG embed -> flattened CNN input 63*128=8064
        self.embed_to_latent = nn.Linear(num_latents, num_channels * latent_dim)
        # CNN upsampler  (8064,1,1) -> (4,64,64)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,   32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,   16,  1, 1, 0), nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,    4,  1, 1, 0)
        )
    def forward(self, x_embed):
        x = self.embed_to_latent(x_embed)         # (B,8064)
        x = x.view(x.size(0), 8064, 1, 1)          # (B,8064,1,1)
        return self.upsampler(x)                  # (B,4,64,64)

from loss import ClipLoss
clip_loss = ClipLoss()

# def train_model(eegmodel, imgmodel, dataloader, optimizer, device, text_features_all, img_features_all, save_dir, epoch):
#     eegmodel.train()
#     img_features_all = (img_features_all[::10]).to(device).float()
#     total_loss = 0
#     correct = 0
#     total = 0
#     alpha=0.9
#     features_list = []  # List to store features
#     save_features= True
#     ridge_lambda = 0.1
#     mse_loss_fn = nn.MSELoss()
#     mae_loss_fn = nn.L1Loss()
#     image_reconstructed = False  # Flag to track if the image has been reconstructed
#     epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
#     if not os.path.exists(epoch_save_dir):
#         os.makedirs(epoch_save_dir)
#     for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
#         eeg_data = eeg_data.to(device)
#         # eeg_data = eeg_data.permute(0, 2, 1)
#         img_features = img_features.to(device).float()
#         labels = labels.to(device)
        
#         optimizer.zero_grad()
#         eeg_features = eegmodel(eeg_data[:, :, :250]).float()
#         pdb.set_trace()
#         # img_features_outputs = regression(eeg_features).float()
#         # features_list.append(eeg_features)
#         logit_scale = eegmodel.logit_scale
#         # print("eeg_features", eeg_features.shape)
#         # print("img_features", img_features.shape)
#         # contras_loss = clip_loss(eeg_features.view(eeg_features.size(0), -1), img_features.view(img_features.size(0), -1), logit_scale)
#         # img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
#         # text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
#         # contrastive_loss = img_loss
#         # print("text_loss", text_loss)
#         # print("img_loss", img_loss)
        
#         regress_loss =  mae_loss_fn(eeg_features, img_features)
#         # l2_norm = sum(p.pow(2.0).sum() for p in eegmodel.parameters())
#         # loss = (regress_loss + ridge_lambda * l2_norm)       
#         loss = regress_loss
#         loss.backward()
        
#         optimizer.step()
#         total_loss += loss.item()

#         # pdb.set_trace() # humne added this
      
#         with torch.no_grad():
#             if not image_reconstructed:
#                 z = eeg_features
#                 x_rec = vae.decode(z).sample
#                 x_train = vae.decode(img_features).sample
#                 image_rec = image_processor.postprocess(x_rec, output_type='pil')
#                 image_train = image_processor.postprocess(x_train, output_type='pil')
#                 # Use label to create a unique file name
#                 for i, label in enumerate(labels.tolist()):                    
#                     save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label}.png")                    
#                     image_rec[i].save(save_path)
                                 
#                     save_path2 = os.path.join(epoch_save_dir, f"train_image_{label}.png")                    
#                     image_train[i].save(save_path2)
#                     image_reconstructed = True                    
        
#         # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
#         # Compute the corresponding logits
#         # logits_img = logit_scale * eeg_features @ img_features_all.T
#         # logits_text = logit_scale * eeg_features @ text_features_all.T
#         # logits_single = (logits_text + logits_img) / 2.0        
#         # logits_text = logit_scale * eeg_features @ text_features_all.T
#         # logits_single = logits_img
#         # predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

#         # batch_size = predicted.shape[0]
#         # total += batch_size
#         # correct += (predicted == labels).sum().item()
#         del eeg_features, img_features, eeg_data
        
#     torch.cuda.empty_cache()

#     average_loss = total_loss / (batch_idx+1)
#     accuracy = 0
#     top5_acc = 0
#     return average_loss, accuracy, top5_acc


def train_model(eegmodel, imgmodel, dataloader, optimizer, device, text_features_all, img_features_all, save_dir, epoch):
    eegmodel.train()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    mse_loss_fn = nn.MSELoss()
    image_reconstructed = False  # Flag to track if the image has been reconstructed
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)

    for batch_idx, (raw_eeg, label, text, text_feats, img, img_feats, eeg_embed) in enumerate(dataloader):
        eeg_embed = eeg_embed.to(device).float()        # (B, 1024)
        img_feats = img_feats.to(device).float()        # (B, 4, 64, 64)
        label = label.to(device)

        optimizer.zero_grad()
        predicted_latents = eegmodel(eeg_embed)         # (B, 4, 64, 64)

        loss = mse_loss_fn(predicted_latents, img_feats)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            if not image_reconstructed:
                x_rec = vae.decode(predicted_latents).sample
                x_train = vae.decode(img_feats).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')
                image_train = image_processor.postprocess(x_train, output_type='pil')

                for i, lab in enumerate(label.tolist()):
                    recon_path = os.path.join(epoch_save_dir, f"reconstructed_image_{lab}.png")
                    train_path = os.path.join(epoch_save_dir, f"train_image_{lab}.png")
                    image_rec[i].save(recon_path)
                    image_train[i].save(train_path)
                image_reconstructed = True

        del eeg_embed, img_feats, predicted_latents

    torch.cuda.empty_cache()
    average_loss = total_loss / (batch_idx + 1)
    return average_loss, 0, 0



# def evaluate_model(eegmodel, imgmodel, dataloader, device, text_features_all, img_features_all, k, save_dir, epoch):
#     eegmodel.eval()
#     img_features_all = img_features_all.to(device).float()
#     total_loss = 0
#     mse_loss_fn = nn.MSELoss()
#     mae_loss_fn = nn.L1Loss()
#     ridge_lambda = 0.1 
#     accuracy = 0
#     alpha = 0.9
#     top5_acc = 0
    
#     epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
#     if not os.path.exists(epoch_save_dir):
#         os.makedirs(epoch_save_dir)
#     fg = True
#     with torch.no_grad():
#         for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):            
#             if batch_idx >= 15:
#                 break #HUMNE uncommented out this code, which we wrote ourselves
#             eeg_data = eeg_data.to(device)
#             # eeg_data = eeg_data.permute(0, 2, 1)
#             labels = labels.to(device)
#             print("labels length:", len(labels))
#             # pdb.set_trace()
#             print("Batch idx ",batch_idx)
#             # pdb.set_trace()
#             img_features = img_features.to(device).float()
#             eeg_features = eegmodel(eeg_data[:, :, :250]).float()
#             logit_scale = eegmodel.logit_scale
#             regress_loss = mae_loss_fn(eeg_features, img_features)
#             # contras_loss = clip_loss(eeg_features.view(eeg_features.size(0), -1), img_features.view(img_features.size(0), -1), logit_scale)
#             loss = regress_loss 
#             total_loss += loss.item()
            
#             print(labels) # yeh humne add kiya hai
            
#             if epoch % 10                    == 0:  #humne 10 ko 1 kiya hai 
#                 z = eeg_features
#                 x_rec = vae.decode(z).sample
#                 image_rec = image_processor.postprocess(x_rec, output_type='pil')
                
#                 # Use label to create a unique file name
#                 # label_name = str(labels.item())
#                 # save_path = os.path.join(epoch_save_dir, f"reconstructed_image_weichen_{label_name}.png")
#                 # image_rec[0].save(save_path)
                
#                 # Use label to create a unique file name
#                 for i, label in enumerate(labels.tolist()):  
#                     # print("Iteration", it)

#                     recon_epoch_save_dir = os.path.join(epoch_save_dir, "recons")
#                     if not os.path.exists(recon_epoch_save_dir):
#                         os.makedirs(recon_epoch_save_dir)

#                     base_save_path = os.path.join(recon_epoch_save_dir, f"reconstructed_image_{label}_0.png")
#                     save_path = base_save_path

#                     ref_epoch_save_dir = os.path.join(epoch_save_dir, "refs")
#                     if not os.path.exists(ref_epoch_save_dir):
#                         os.makedirs(ref_epoch_save_dir)

#                     ref_save_path = os.path.join(ref_epoch_save_dir, f"reference_image_{label}_0.png") # Humne wrote this

#                     #humne is neeche wale paragraph ko comment kiya hia 
#                     # k = 0
#                     # # Check if the file already exists
#                     # while os.path.exists(save_path):
#                     #     save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{label}_{k}.png")
#                     #     k += 1


#                     # Save the image
#                     image_rec[i].save(save_path) # Humne commented this out 
#                     print(type(img), img)
#                     # pdb.set_trace()
#                     Image.open(img[0]).save(ref_save_path) # Humne wrote this
#                 # pdb.set_trace()
#                 del eeg_features, img_features, eeg_data, image_rec, x_rec    
#                 continue
#             del eeg_features, img_features, eeg_data
            
#     torch.cuda.empty_cache()
#     average_loss = total_loss / (batch_idx + 1)
#     return average_loss, accuracy, top5_acc

def evaluate_model(eegmodel, imgmodel, dataloader, device, text_features_all, img_features_all, k, save_dir, epoch):
    eegmodel.eval()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0

    total_batches = len(dataloader)
    batches_per_epoch = 20
    start_batch = (epoch * batches_per_epoch) % total_batches
    end_batch = start_batch + batches_per_epoch

    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (raw_eeg, labels, text, text_feats, img, img_feats, eeg_embed) in enumerate(dataloader):
            if batch_idx < start_batch or batch_idx >= end_batch:
                continue  # skip batches not in current epoch's chunk

            eeg_embed = eeg_embed.to(device).float()
            img_feats = img_feats.to(device).float()
            labels = labels.to(device)

            pred_latents = eegmodel(eeg_embed)
            loss = mse_loss_fn(pred_latents, img_feats)
            total_loss += loss.item()

            x_rec = vae.decode(pred_latents).sample
            image_rec = image_processor.postprocess(x_rec, output_type='pil')

            for i, lab in enumerate(labels.tolist()):
                recon_dir = os.path.join(epoch_save_dir, "recons")
                ref_dir = os.path.join(epoch_save_dir, "refs")
                os.makedirs(recon_dir, exist_ok=True)
                os.makedirs(ref_dir, exist_ok=True)

                recon_path = os.path.join(recon_dir, f"reconstructed_image_{lab}_b{batch_idx}_i{i}.png")
                ref_path = os.path.join(ref_dir, f"reference_image_{lab}_b{batch_idx}_i{i}.png")

                image_rec[i].save(recon_path)
                Image.open(img[i]).save(ref_path)
    torch.cuda.empty_cache()
    average_loss = total_loss / batches_per_epoch
    return average_loss, 0, 0



# def main_train_loop(sub, current_time, eeg_model, img_model, train_dataloader, test_dataloader, optimizer, device, 
#                     text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
#     # Introduce cosine annealing scheduler
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
#     logger = wandb_logger(config) if logger else None
#     logger.watch(eeg_model,logger) 
#     logger.watch(img_model,logger) 
#     train_losses, train_accuracies = [], []
#     test_losses, test_accuracies = [], []
#     v2_accs = []
#     v4_accs = []
#     v10_accs = []

#     best_accuracy = 0.0
#     best_model_weights = None
#     best_epoch_info = {}
#     results = []  # List to store results for each epoch
    
#     for epoch in range(config.epochs):

#         # Add date-time prefix to save_dir
#         train_save_dir = f'output/images/{current_time}_vae_train_imgs'
#         train_loss, train_accuracy, features_tensor = train_model(eeg_model, img_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all, save_dir=train_save_dir, epoch=epoch)

#         if (epoch +1) % 5 == 0:            
#             # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')                  
#             if config.insubject==True:       
#                 os.makedirs(f"output/models/generation-contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
#                 file_path = f"output/models/generation-contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
#                 torch.save(eeg_model.state_dict(), file_path)            
#             else:                
#                 os.makedirs(f"output/models/generation-contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
#                 file_path = f"output/models/generation-contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
#                 torch.save(eeg_model.state_dict(), file_path)
#             print(f"model saved in {file_path}!")
        
#         train_losses.append(train_loss)
#         train_accuracies.append(train_accuracy)

#         # Update learning rate
#         scheduler.step()
        
#         # Evaluate the model
#         # test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
#                 # Call evaluate_model function
#                         # Get the current date and time, format as "YYYYMMDD_HHMM"

#         # Add date-time prefix to save_dir
#         test_save_dir = f'output/images/{current_time}_vae_imgs'
#         test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=200, save_dir=test_save_dir, epoch=epoch)
#         test_losses.append(test_loss)
#         test_accuracies.append(test_accuracy)

        
#         # Append results for this epoch
#         epoch_results = {
#         "epoch": epoch + 1,
#         # "train_loss": train_loss,
#         # "train_accuracy": train_accuracy,
#         "test_loss": test_loss,
#         "test_accuracy": test_accuracy,
#         }

#         results.append(epoch_results)
#         # If the test accuracy in the current epoch is the best, save the model and related information
#         if test_accuracy > best_accuracy:
#             best_accuracy = test_accuracy
#             # best_model_weights = model.state_dict().copy()
            
#             best_epoch_info = {
#                 "epoch": epoch + 1,
#                 "train_loss": train_loss,
#                 "train_accuracy": train_accuracy,
#                 "test_loss": test_loss,
#                 "test_accuracy": test_accuracy,
#             }
#         logger.log({
#             "Train Loss": train_loss,
#             "Train Accuracy": train_accuracy,
#             "Test Loss": test_loss,
#             "Test Accuracy": test_accuracy,
#             "Epoch": epoch
#         })

#         print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
#         torch.cuda.empty_cache()

#     logger.finish()
#     return results

def main_train_loop(
    sub, current_time,
    eeg_model, img_model,
    train_dataloader, test_dataloader,
    optimizer, device,
    text_features_train_all, text_features_test_all,
    img_features_train_all, img_features_test_all,
    config, logger=None
):
    # 1) load the pre-trained ATM that produces your 1×1024 EEG embeddings
    atm = load_atms_model(config.atm_checkpoint, device)

    # 2) wrap both train & test loaders so each batch yields eeg_embed + img_feats
    #    subject_id: convert "sub-01" → 0, etc.
    subject_id = int(sub.split('-')[-1]) - 1
    train_loader = DataLoader(
        EEGWithEmbedding(train_dataloader.dataset, atm, subject_id, device),
        batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(
        EEGWithEmbedding(test_dataloader.dataset, atm, subject_id, device),
        batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True
    )

    # 3) set up scheduler & logger
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    logger = wandb_logger(config) if logger else None
    if logger:
        logger.watch(eeg_model,logger)
        logger.watch(img_model,logger)

    results = []
    for epoch in range(config.epochs):
        # train step (uses eeg_embed → encoder_low_level → MSE against img_feats)
        train_save_dir = f'output/images/{current_time}_vae_train_imgs'
        train_loss, train_acc, _ = train_model(
            eeg_model, img_model,
            train_loader, optimizer, device,
            text_features_train_all, img_features_train_all,
            save_dir=train_save_dir, epoch=epoch
        )

        # checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_dir = (f"output/models/generation-contrast/{config.encoder_type}/{sub}/{current_time}"
                        if config.insubject
                        else f"output/models/generation-contrast/across/{config.encoder_type}/{current_time}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(eeg_model.state_dict(), f"{ckpt_dir}/{epoch+1}.pth")

        scheduler.step()

        # eval step
        test_save_dir = f'output/images/{current_time}_vae_imgs'
        test_loss, test_acc, _ = evaluate_model(
            eeg_model, img_model,
            test_loader, device,
            text_features_test_all, img_features_test_all,
            k=200,
            save_dir=test_save_dir,
            epoch=epoch
        )

        # log & record
        if logger:
            logger.log({
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Epoch": epoch
            })
        print(f"Epoch {epoch+1}/{config.epochs}  Train Loss: {train_loss:.4f}  Test Loss: {test_loss:.4f}")

        results.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        torch.cuda.empty_cache()

    if logger:
        logger.finish()
    return results

# def main():
#     # Argument parser setup
#     parser = argparse.ArgumentParser(description='EEG Model Training Script')
#     parser.add_argument('--data_path', type=str, default='data/weights/EEG_Image_decode/Preprocessed_data_250Hz', help='Path to data')
#     parser.add_argument('--output_dir', type=str, default='output/statistics/generation-contrast', help='Directory to save output results')
#     parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='Project name for logging')
#     parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
#     parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
#     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')  #humne epochs ko 200 se 6 kiya THA
#     parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
#     parser.add_argument('--insubject', default=True, help='Flag to indicate within-subject training')
#     parser.add_argument('--encoder_type', type=str, default='encoder_low_level', 
#                         choices=['EEGNetv4_Encoder', 'ATCNet_Encoder', 'EEGConformer_Encoder', 'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder', 'encoder_low_level'], 
#                         help='Encoder type')
#     parser.add_argument('--img_encoder', type=str, default='Proj_img', help='Image encoder type')
#     parser.add_argument('--logger', default=True, help='Enable logging')
#     parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
#     parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')
#     parser.add_argument('--subjects', nargs='+', default=['sub-08'], help='List of subject IDs')
    
#     args = parser.parse_args()

#     # Set device based on the argument
#     if args.device == 'gpu' and torch.cuda.is_available():
#         device = torch.device(args.gpu)
#     else:
#         device = torch.device('cpu')
    
#     data_path = args.data_path
#     subjects = args.subjects
#     current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

#     for sub in subjects:
#         # Re-initialize the models for each subject
#         eeg_model = globals()[args.encoder_type]()
#         img_model = globals()[args.img_encoder]()
        
#         eeg_model.to(device)
#         img_model.to(device)
        
#         optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=args.lr)
        
#         if args.insubject:
#             train_dataset = EEGDataset(data_path, subjects=[sub], train=True)
#             # pdb.set_trace()
#             test_dataset = EEGDataset(data_path, subjects=[sub], train=False)
#             # pdb.set_trace()
#         else:
#             train_dataset = EEGDataset(data_path, exclude_subject=sub, train=True)
#             test_dataset = EEGDataset(data_path, exclude_subject=sub, train=False)
            
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True) # Humne false kiya THA
#         test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0, drop_last=True) #Humne false kiya THA

#         text_features_train_all = train_dataset.text_features
#         text_features_test_all = test_dataset.text_features
#         img_features_train_all = train_dataset.img_features
#         img_features_test_all = test_dataset.img_features

#         results = main_train_loop(sub, current_time, eeg_model, img_model, train_loader, test_loader, optimizer, device, 
#                                   text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, 
#                                   config=args, logger=args.logger)

#         # Save results to a CSV file
#         results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
#         os.makedirs(results_dir, exist_ok=True)
        
#         if args.insubject:
#             results_file = os.path.join(results_dir, f"{args.encoder_type}_{sub}.csv")
#         else:
#             results_file = os.path.join(results_dir, f"{args.encoder_type}_cross_exclude_{sub}.csv")

#         with open(results_file, 'w', newline='') as file:
#             writer = csv.DictWriter(file, fieldnames=results[0].keys())
#             writer.writeheader()
#             writer.writerows(results)
#             print(f'Results saved to {results_file}')

# if __name__ == '__main__':
#     main()

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--data_path',    type=str,   default='data/weights/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--atm_checkpoint', type=str, default="/DATA/deep_learning/eeg-to-img/output/models/generation-contrast/ATMS/sub-08/04-25_09-00/40.pth",
                        help='/DATA/deep_learning/eeg-to-img/output/models/generation-contrast/ATMS/sub-08/04-25_09-00/40.pth')
    parser.add_argument('--output_dir',   type=str,   default='output/statistics/generation-contrast')
    parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='Project name for logging')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--epochs',       type=int,   default=201)
    parser.add_argument('--batch_size',   type=int,   default=10)
    parser.add_argument('--insubject', default=True, help='Flag to indicate within-subject training')
    parser.add_argument('--encoder_type', type=str,   default='encoder_low_level',
                        choices=['encoder_low_level'], 
                        help='Which EEG→latent encoder to train')
    parser.add_argument('--img_encoder',  type=str,   default='Proj_img')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--device',       type=str,   choices=['cpu','gpu'], default='gpu')
    parser.add_argument('--gpu',          type=str,   default='cuda:0')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--subjects',     nargs='+',  default=['sub-08'])
    args = parser.parse_args()

    # Make the checkpoint path available in config
    args.atm_checkpoint = args.atm_checkpoint

    # Set device
    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in args.subjects:
        # instantiate models
        eeg_model = globals()[args.encoder_type]()
        img_model = globals()[args.img_encoder]()
        eeg_model.to(device); img_model.to(device)

        optimizer = torch.optim.AdamW(
            itertools.chain(eeg_model.parameters(), img_model.parameters()),
            lr=args.lr
        )

        # raw EEGDataset loaders (will be re-wrapped inside main_train_loop)
        if args.insubject:
            train_ds = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_ds  = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_ds = EEGDataset(args.data_path, exclude_subject=sub, train=True)
            test_ds  = EEGDataset(args.data_path, exclude_subject=sub, train=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=20,             shuffle=False, num_workers=0, drop_last=True)

        # features pre-extracted from CLIP/VAE
        text_tr, text_te = train_ds.text_features, test_ds.text_features
        img_tr,  img_te  = train_ds.img_features,  test_ds.img_features

        # pdb.set_trace()

        # call the unified training loop
        results = main_train_loop(
            sub, current_time,
            eeg_model, img_model,
            train_loader, test_loader,
            optimizer, device,
            text_tr, text_te,
            img_tr, img_te,
            config=args, logger=args.logger
        )

        # save results CSV
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        fname = f"{args.encoder_type}_{sub}.csv" if args.insubject else f"{args.encoder_type}_cross_{sub}.csv"
        with open(os.path.join(results_dir, fname),'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
        print(f"Results saved to {results_dir}/{fname}")

if __name__=='__main__':
    main()



