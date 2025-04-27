import torch
from torch.utils.data import DataLoader, Dataset
from ATMS_reconstruction import ATMS
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
import torch.multiprocessing as mp
import pdb

# helper exactly as before
def load_atms_model(checkpoint_path: str, device: torch.device) -> ATMS:
    model = ATMS()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def get_eeg_embeddings(model: ATMS, raw_eeg: torch.Tensor, subject_id: int, device: torch.device):
    raw_eeg = raw_eeg.to(device)
    B = raw_eeg.size(0)
    subject_ids = torch.full((B,), subject_id, dtype=torch.long, device=device)
    with torch.no_grad():
        return model(raw_eeg, subject_ids)  # → (B, 1024)

class EEGWithEmbedding(Dataset):
    """
    Wraps your original EEGDataset, runs ATMS on the fly,
    and returns: (raw_eeg, label, text, text_feats, img, img_feats, eeg_embed)
    """
    def __init__(self, base_ds, atms_model, subject_id, device):
        self.base = base_ds
        self.model = atms_model
        self.sub_id = subject_id
        self.device = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Unpack all original fields
        raw_eeg, label, text, text_feats, img, img_feats = self.base[idx]
        # Add the new field
        raw = raw_eeg.unsqueeze(0)  # shape (1, C, T)
        embed = get_eeg_embeddings(self.model, raw, self.sub_id, self.device)
        # squeeze back to (1024,)
        print(embed.shape)
        print("embed: ",len(embed))
        eeg_embed = embed.squeeze(0).cpu()
        print(eeg_embed.shape)
        print("EEG embed: ",len(eeg_embed))
        return raw_eeg, label, text, text_feats, img, img_feats, eeg_embed

if __name__ == "__main__":
    # -------- Config --------
    mp.set_start_method('spawn', force=True)
    data_path  = "/DATA/deep_learning/eeg-to-img/data/weights/EEG_Image_decode/Preprocessed_data_250Hz"
    checkpoint = "/DATA/deep_learning/eeg-to-img/output/models/generation-contrast/ATMS/sub-08/04-25_09-00/40.pth"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_id = 7
    BATCH      = 16

    # 1) Load ATMS once
    atms = load_atms_model(checkpoint, device)

    # 2) Original dataset + loader
    base_test = EEGDataset(data_path, subjects=['sub-08'], train=False)
    wrapped_ds = EEGWithEmbedding(base_test, atms, subject_id, device)
    loader     = DataLoader(wrapped_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    # 3) Inference: now each `batch` has 7 elements
    for (raw_eeg, label, text, text_feats, img, img_feats, eeg_embed) in loader:
        # raw_eeg: (B, C, T)
        # label:   (B,)
        # text:    list of length B
        # text_feats: Tensor or None
        # img:     list of length B (paths)
        # img_feats: Tensor of image latents
        # eeg_embed: Tensor (B, 1024)
        # … do whatever you need with these seven fields …
        pass

    # If you still want to collect *just* the embeddings into a matrix:
    all_embeds = []
    for *_, eeg_embed in loader:
        all_embeds.append(eeg_embed)
    embeddings_matrix = torch.cat(all_embeds, dim=0)  # shape (N, 1024)
    print("Got embeddings:", embeddings_matrix.shape)
