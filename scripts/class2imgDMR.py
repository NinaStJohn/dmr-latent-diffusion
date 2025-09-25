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
    key = getattr(model, "cond_stage_key", "class_label")   # <- FIX
    labels = torch.full((args.n,), args.class_id, device=device, dtype=torch.long)

    # Many LDM repos accept either a tensor or a dict keyed by cond_stage_key.
    # Prefer passing the raw labels when using ClassEmbedder; it avoids dict-key mismatches.
    try:
        c = model.get_learned_conditioning(labels)          # <- prefer direct labels
    except Exception:
        c = model.get_learned_conditioning({key: labels})   # <- fallback to dict

    # --- unconditional for CFG (use proper "null" embedding if available) ---
    uc = None
    # If your ClassEmbedder supports a null label (common), use -1:
    try:
        null_labels = torch.full((args.n,), -1, device=device, dtype=torch.long)
        uc = model.get_learned_conditioning(null_labels)
    except Exception:
        # Some implementations expose a helper:
        try:
            uc = model.get_unconditional_conditioning(args.n)
        except Exception:
            # Last resort: same shape zeros (still works, just weaker guidance)
            if isinstance(c, dict):
                uc = {k: torch.zeros_like(v) for k, v in c.items()}
            elif isinstance(c, list):
                uc = [torch.zeros_like(v) for v in c]
            else:
                uc = torch.zeros_like(c)

    print("conditioning_key:", getattr(model, "conditioning_key", None))
    print("cond_stage_key:", getattr(model, "cond_stage_key", None))
    def _shape_tree(x):
        if isinstance(x, dict): return {k: (v.shape if torch.is_tensor(v) else type(v)) for k,v in x.items()}
        if isinstance(x, list): return [ (v.shape if torch.is_tensor(v) else type(v)) for v in x ]
        return x.shape if torch.is_tensor(x) else type(x)
    print("c shape:", _shape_tree(c))
    print("uc shape:", _shape_tree(uc))

    # --- one-step sanity check: does class actually change the predicted noise? ---
    with torch.no_grad():
        z_dbg = torch.randn(1, C, H_z, W_z, device=device)
        t_dbg = torch.tensor([args.steps // 2], device=device)  # mid-timestep

        # build cond for class_id and a different class (wrap around to ensure in-range)
        y0 = torch.full((1,), args.class_id, device=device, dtype=torch.long)
        y1 = torch.full((1,), (args.class_id + 1) % max(2, int(getattr(model, "num_classes", 9999))), 
                        device=device, dtype=torch.long)
        try:
            c0 = model.get_learned_conditioning(y0)
            c1 = model.get_learned_conditioning(y1)
        except Exception:
            c0 = model.get_learned_conditioning({key: y0})
            c1 = model.get_learned_conditioning({key: y1})

        eps0 = model.apply_model(z_dbg, t_dbg, c0)
        eps1 = model.apply_model(z_dbg, t_dbg, c1)
        delta = (eps0 - eps1).abs().mean().item()
        print(f"[sanity] mean|Δε| between class {y0.item()} and {y1.item()} =", delta)



    # unique prefix (prevents overwrites across Slurm array runs)
    run_tag = args.prefix or f"s{args.seed}_t{int(time.time())}"
    save_prefix = f"class{args.class_id}_{args.H}_{args.W}_{run_tag}"

    # --- sample with EMA + (optional) autocast + CFG ---
    amp = torch.autocast("cuda") if (args.autocast and device=="cuda") else torch.cuda.amp.autocast(enabled=False)
    from contextlib import nullcontext
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
