# vae_check.py
import os, glob
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from math import log10
from skimage.metrics import structural_similarity as ssim

# --- EDIT THESE ---
CONFIG_PATH = dmr-latent-diffusion/logs/2025-08-27T20-09-27_autoencoder_ACCESS/configs/2025-08-28T17-15-58-project.yaml               # your AE config
CKPT_PATH   = /u/nstjohn/dmr-latent-diffusion/logs/2025-08-27T20-09-27_autoencoder_ACCESS/checkpoints/last.ckpt  # your AE checkpoint
IMAGE_DIR   = /u/nstjohn/vae_check_samples          # 5–20 real dataset images
OUT_DIR     = "vae_recons"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load AE (CompVis style) ---
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_model(cfg_path, ckpt_path):
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in pl_sd: pl_sd = pl_sd["state_dict"]
    missing, unexpected = model.load_state_dict(pl_sd, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0: return 99.0
    return 10 * log10(1.0 / mse)

def to_tensor_01(img_pil):
    return T.ToTensor()(img_pil)  # [0,1]

def to_tensor_m11(img_pil):
    x = to_tensor_01(img_pil)
    return x * 2 - 1  # [-1,1]

def to_pil_from_m11(x):
    x = (x.clamp(-1,1) + 1) / 2
    x = (x * 255.0).byte().permute(1,2,0).cpu().numpy()
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
        return Image.fromarray(x, mode="L")
    return Image.fromarray(x)

def compute_ssim(a, b):
    # inputs in [0,1], CHW
    a = a.clamp(0,1).permute(1,2,0).cpu().numpy()
    b = b.clamp(0,1).permute(1,2,0).cpu().numpy()
    if a.ndim == 3 and a.shape[2] == 3:
        return ssim(a, b, channel_axis=2, data_range=1.0)
    else:
        return ssim(a.squeeze(), b.squeeze(), data_range=1.0)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(CONFIG_PATH, CKPT_PATH)
    # try to read scale_factor if present
    scale_factor = getattr(model, "scale_factor", 1.0)
    print("scale_factor:", scale_factor)

    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))
    paths = [p for p in paths if p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))][:20]
    assert paths, f"No images found in {IMAGE_DIR}"

    psnrs, ssims = [], []
    for p in paths:
        img = Image.open(p).convert("RGB")  # or "L" if grayscale; keep consistent with training
        x_m11 = to_tensor_m11(img).unsqueeze(0).to(DEVICE)  # [-1,1]
        with torch.no_grad():
            # encode -> z
            z = model.encode(x_m11).sample() if hasattr(model.encode(x_m11), "sample") else model.encode(x_m11)
            z = z * scale_factor  # match training convention (if used)
            # decode <- z
            x_rec = model.decode(z / scale_factor)
        # metrics
        x01 = (x_m11 + 1) / 2
        rec01 = (x_rec + 1) / 2
        P = psnr(x01, rec01)
        S = compute_ssim(x01.squeeze(0), rec01.squeeze(0))
        psnrs.append(P); ssims.append(S)

        # save side-by-side
        cat = Image.new("RGB", (img.width*2, img.height))
        cat.paste(img, (0,0))
        cat.paste(to_pil_from_m11(x_rec.squeeze(0)), (img.width,0))
        cat.save(os.path.join(OUT_DIR, os.path.basename(p).rsplit(".",1)[0] + "_recon.jpg"))

    print(f"VAE recon PSNR mean={np.mean(psnrs):.2f} dB  median={np.median(psnrs):.2f} dB")
    print(f"VAE recon SSIM mean={np.mean(ssims):.3f}  median={np.median(ssims):.3f}")
    print(f"Saved examples to {OUT_DIR}")

if __name__ == "__main__":
    main()
