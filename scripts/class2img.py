import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch, argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler  # NEW

def load_model(cfg_path, ckpt, device):
    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def save_imgs(x, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    x = (x.clamp(-1,1) + 1)/2
    x = (x*255).to(torch.uint8).permute(0,2,3,1).cpu()
    for i, img in enumerate(x):
        Image.fromarray(img.numpy()).save(os.path.join(outdir, f"{prefix}_{i:04d}.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class_id", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--outdir", default="samples_cls")
    ap.add_argument("--prefix", default=None)                 # NEW
    ap.add_argument("--scale", type=float, default=3.0)       # NEW (CFG guidance scale)
    ap.add_argument("--sampler", choices=["ddim","plms"], default="ddim")  # NEW
    ap.add_argument("--autocast", action="store_true")        # NEW
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, args.ckpt, device)
    sampler = DDIMSampler(model) if args.sampler=="ddim" else PLMSSampler(model)

    # --- infer latent shape robustly (works for any VAE downsample factor) ---
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.H, args.W, device=device)
        posterior = model.encode_first_stage(dummy)
        z0 = model.get_first_stage_encoding(posterior)    # [1, C, H/f, W/f]
    C, H_z, W_z = z0.shape[1], z0.shape[2], z0.shape[3]
    # UNet usually needs multiples of 8 in latent space:
    assert H_z % 8 == 0 and W_z % 8 == 0, f"Pick H/W so latent is multiple of 8, got {H_z}x{W_z}."

    # --- build conditioning (reads the key you trained with) ---
    key = getattr(model.cond_stage_model, "key", "class_label")
    labels = torch.full((args.n,), args.class_id, device=device, dtype=torch.long)
    cond = {key: labels}
    c = model.get_learned_conditioning(cond)

    # unconditional for CFG
    try:
        uc = model.get_unconditional_conditioning(args.n)
    except Exception:
        uc = torch.zeros_like(c)

    # unique prefix (prevents overwrites across Slurm array runs)
    run_tag = args.prefix or f"s{args.seed}_t{int(time.time())}"
    save_prefix = f"class{args.class_id}_{args.H}_{args.W}_{run_tag}"

    # --- sample with EMA + (optional) autocast + CFG ---
    amp = torch.autocast("cuda") if (args.autocast and device=="cuda") else torch.cuda.amp.autocast(enabled=False)
    with torch.no_grad(), model.ema_scope("Sampling"), amp:
        z, _ = sampler.sample(
            S=args.steps,
            conditioning=c,
            batch_size=args.n,
            shape=(C, H_z, W_z),
            eta=args.eta,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
        )
        x = model.decode_first_stage(z)

    save_imgs(x, args.outdir, save_prefix)
