import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch, argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# put near the top of class2imgDMR.py
class _CatList(list):
    @property
    def shape(self):
        # make the sampler happy (reads batch size)
        return self[0].shape


def _fix_or_drop_mismatched_keys(model, sd):
    """
    Fix common attention projection mismatches (Conv1d [C,C,1] -> Conv2d [C,C,1,1]).
    Drop any remaining shape-mismatched tensors so load_state_dict won't error.
    """
    msd = model.state_dict()
    fixed, dropped = 0, 0
    for k in list(sd.keys()):
        if k in msd and torch.is_tensor(sd[k]) and torch.is_tensor(msd[k]):
            v, t = sd[k], msd[k]
            if v.shape != t.shape:
                # Try simple unsqueeze fix for 3D -> 4D 1x1 kernels
                if v.ndim == 3 and t.ndim == 4 and t.shape[-2:] == (1, 1) and v.shape[-1] == 1:
                    sd[k] = v.unsqueeze(-1)
                    fixed += 1
                else:
                    # Drop incompatible tensor
                    sd.pop(k)
                    dropped += 1
    return fixed, dropped

def load_model(cfg_path, ckpt_path, device):
    # Guard: YAML must be a text file
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    if os.path.splitext(cfg_path)[1].lower() not in (".yaml", ".yml"):
        raise ValueError(f"--config must be .yaml/.yml, got: {cfg_path}")

    # Load config
    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)

    # Conditioning debug
    print("[cfg] conditioning_key =", cfg.model.params.get("conditioning_key", None))
    try:
        print("[cfg] use_spatial_transformer =", cfg.model.params.unet_config.params.get("use_spatial_transformer"))
        print("[cfg] context_dim =", cfg.model.params.unet_config.params.get("context_dim"))
    except Exception:
        pass

    # If constructor didn't set it, inherit from cfg
    if getattr(model, "conditioning_key", None) is None:
        model.conditioning_key = cfg.model.params.get("conditioning_key", None)
    print("model.conditioning_key (effective):", getattr(model, "conditioning_key", None))

    # Check SpatialTransformer presence
    unet = getattr(getattr(model, "model", None), "diffusion_model", None)
    has_st = (unet is not None) and any(m.__class__.__name__.endswith("SpatialTransformer")
                                        for m in unet.modules())
    print("has SpatialTransformer:", has_st)

    # Load checkpoint with fixes
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        fixed, dropped = _fix_or_drop_mismatched_keys(model, sd)
        print(f"[load] attention params fixed: {fixed}, dropped: {dropped}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.to(device).eval()
    return model

def save_imgs(x, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
    for i, img in enumerate(x):
        Image.fromarray(img.numpy()).save(os.path.join(outdir, f"{prefix}_{i:04d}.png"))

def sample_ddim_cfg_crossattn(model, sampler, steps, shape, cond_ctx, uc_ctx, scale=7.5, eta=0.0, device="cuda"):
    """
    cond_ctx, uc_ctx are TENSORS from get_learned_conditioning({key: labels})
    """
    # Build schedule once
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)

    b, C, H, W = shape
    x = torch.randn(b, C, H, W, device=device)

    # Convert sampler's numpy schedules -> torch (on device, right dtype)
    alphas      = torch.as_tensor(sampler.ddim_alphas,       device=device, dtype=x.dtype)
    alphas_prev = torch.as_tensor(sampler.ddim_alphas_prev,  device=device, dtype=x.dtype)
    sigmas      = torch.as_tensor(sampler.ddim_sigmas,       device=device, dtype=x.dtype)
    timesteps   = torch.as_tensor(sampler.ddim_timesteps,    device=device, dtype=torch.long)

    for i in range(len(timesteps)):
        t = timesteps[i].repeat(b)  # [b]
        # CFG: stack uncond/cond along batch
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        cond_in = {"c_crossattn": [torch.cat([uc_ctx, cond_ctx], dim=0)]}

        # predict noise for [uncond; cond]
        e_t = model.apply_model(x_in, t_in, cond_in)
        e_u, e_c = e_t.chunk(2, dim=0)
        e_t = e_u + scale * (e_c - e_u)

        # DDIM update
        a_t     = alphas[i]       # scalar tensor
        a_prev  = alphas_prev[i]  # scalar tensor
        sigma_t = sigmas[i]       # scalar tensor

        sqrt_at    = torch.sqrt(a_t)
        sqrt_oma_t = torch.sqrt(1.0 - a_t)
        pred_x0 = (x - sqrt_oma_t * e_t) / sqrt_at

        # direction pointing to x_t
        dir_coeff = torch.sqrt(torch.clamp(1.0 - a_prev - sigma_t**2, min=0.0))
        noise  = sigma_t * torch.randn_like(x) if i < len(timesteps) - 1 else 0.0
        x = torch.sqrt(a_prev) * pred_x0 + dir_coeff * e_t + noise

    return x



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
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--scale", type=float, default=3.0)          # CFG guidance scale
    ap.add_argument("--sampler", choices=["ddim", "plms"], default="ddim")
    ap.add_argument("--autocast", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.config, args.ckpt, device)
    sampler = DDIMSampler(model) if args.sampler == "ddim" else PLMSSampler(model)

    # --- infer latent shape robustly (works for any VAE downsample factor) ---
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.H, args.W, device=device)
        posterior = model.encode_first_stage(dummy)
        z0 = model.get_first_stage_encoding(posterior)  # [1, C, H/f, W/f]
    C, H_z, W_z = z0.shape[1], z0.shape[2], z0.shape[3]
    assert H_z % 8 == 0 and W_z % 8 == 0, f"Pick H/W so latent is multiple of 8, got {H_z}x{W_z}."

    # --- build conditioning (ClassEmbedder expects a dict) ---
    key = getattr(model, "cond_stage_key", "class_label")  # 'class_id'
    labels = torch.full((args.n,), args.class_id, device=device, dtype=torch.long)
    ctx = model.get_learned_conditioning({key: labels})    # -> [B, 1, D]

    # --- unconditional context for CFG (zeros_like; NO -1 index) ---
    ucx = torch.zeros_like(ctx) if torch.is_tensor(ctx) else {k: torch.zeros_like(v) for k, v in ctx.items()}

    # --- cross-attn wrapper: pass a LIST-LIKE with .shape ---
    if getattr(model, "conditioning_key", None) == "crossattn":
        c  = {"c_crossattn": _CatList([ctx])}
        uc = {"c_crossattn": _CatList([ucx])}
    else:
        c, uc = ctx, ucx


    # unique prefix
    run_tag = args.prefix or f"s{args.seed}_t{int(time.time())}"
    save_prefix = f"class{args.class_id}_{args.H}_{args.W}_{run_tag}"

    # --- quick sanity delta (should be clearly > 0 once training has learned) ---
    with torch.no_grad():
        z = torch.randn(1, C, H_z, W_z, device=device)
        t = torch.tensor([max(1, args.steps // 2)], device=device)
        y0 = torch.tensor([args.class_id], device=device)
        y1 = torch.tensor([(args.class_id + 1) % 7], device=device)

        ctx0 = model.get_learned_conditioning({key: y0})
        ctx1 = model.get_learned_conditioning({key: y1})
        cond0 = {"c_crossattn": _CatList([ctx0])} if model.conditioning_key=="crossattn" else ctx0
        cond1 = {"c_crossattn": _CatList([ctx1])} if model.conditioning_key=="crossattn" else ctx1

        d = (model.apply_model(z, t, cond0) - model.apply_model(z, t, cond1)).abs().mean().item()
        print(f"[sanity] mean|Δε|={d:.6f}")


    # --- sample with EMA + (optional) autocast + CFG ---
    from contextlib import nullcontext
    amp_ctx = torch.amp.autocast('cuda') if (args.autocast and device == "cuda") else nullcontext()
    with torch.no_grad(), model.ema_scope("Sampling"), amp_ctx:
        # ctx/ucx are the TENSORS you already computed via get_learned_conditioning
        z = sample_ddim_cfg_crossattn(
            model, sampler,
            steps=args.steps,
            shape=(args.n, C, H_z, W_z),
            cond_ctx=ctx,
            uc_ctx=ucx,
            scale=args.scale,
            eta=args.eta,
            device=device
        )
        x = model.decode_first_stage(z)


    save_imgs(x, args.outdir, save_prefix)
