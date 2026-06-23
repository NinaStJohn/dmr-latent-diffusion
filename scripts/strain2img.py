import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


# -------------------------------
# Debug helpers (prints only)
# -------------------------------
def _dbg(debug: bool, *args, **kwargs):
    if debug:
        print(*args, **kwargs)

def _tstats(t: torch.Tensor, name: str, debug: bool, extra: str = ""):
    if not debug:
        return
    if t is None:
        print(f"[dbg] {name}: None")
        return
    if not torch.is_tensor(t):
        print(f"[dbg] {name}: type={type(t)} {extra}")
        return
    with torch.no_grad():
        finite = torch.isfinite(t)
        n_finite = int(finite.sum().item())
        n_total = t.numel()
        # Avoid .min/.max on empty/NaN-only tensors
        t_f = t[finite] if n_finite > 0 else None
        mn = float(t_f.min().item()) if t_f is not None else float("nan")
        mx = float(t_f.max().item()) if t_f is not None else float("nan")
        mean = float(t_f.mean().item()) if t_f is not None else float("nan")
        std = float(t_f.std(unbiased=False).item()) if (t_f is not None and n_finite > 1) else float("nan")
        print(
            f"[dbg] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} "
            f"finite={n_finite}/{n_total} {extra}"
        )

def _assert_same_shape(a: torch.Tensor, b: torch.Tensor, name_a: str, name_b: str, debug: bool):
    if not debug:
        return
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape:
            print(f"[dbg][WARN] shape mismatch: {name_a}.shape={tuple(a.shape)} vs {name_b}.shape={tuple(b.shape)}")

def _maybe_print_model_keys(model, debug: bool):
    if not debug:
        return
    # These attributes exist in many CompVis models but not all; guard safely
    for attr in ["conditioning_key", "cond_stage_key", "scale_factor", "downsample_factor"]:
        if hasattr(model, attr):
            print(f"[dbg] model.{attr} = {getattr(model, attr)}")

    # Try to print cond stage model type
    if hasattr(model, "cond_stage_model"):
        print(f"[dbg] cond_stage_model: {model.cond_stage_model.__class__.__name__}")
    if hasattr(model, "first_stage_model"):
        print(f"[dbg] first_stage_model: {model.first_stage_model.__class__.__name__}")


# -------------------------------
# Utilities
# -------------------------------
def _fix_or_drop_mismatched_keys(model, sd):
    """Fix some common 3D->4D attention weight shape mismatches; drop others."""
    msd = model.state_dict()
    fixed, dropped = 0, 0
    for k in list(sd.keys()):
        if k in msd and torch.is_tensor(sd[k]) and torch.is_tensor(msd[k]):
            v, t = sd[k], msd[k]
            if v.shape != t.shape:
                # Special-case: add singleton for (C, H*W, 1) -> (C, H*W, 1, 1)
                if v.ndim == 3 and t.ndim == 4 and t.shape[-2:] == (1, 1) and v.shape[-1] == 1:
                    sd[k] = v.unsqueeze(-1)
                    fixed += 1
                else:
                    sd.pop(k)
                    dropped += 1
    return fixed, dropped


def load_model(cfg_path, ckpt_path, device, debug=False):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"--config not found: {cfg_path}")
    if os.path.splitext(cfg_path)[1].lower() not in (".yaml", ".yml"):
        raise ValueError(f"--config must be .yaml/.yml, got: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)

    print("[cfg] conditioning_key =", cfg.model.params.get("conditioning_key", None))
    try:
        print("[cfg] use_spatial_transformer =", cfg.model.params.unet_config.params.get("use_spatial_transformer"))
        print("[cfg] context_dim =", cfg.model.params.unet_config.params.get("context_dim"))
    except Exception:
        pass

    if getattr(model, "conditioning_key", None) is None:
        model.conditioning_key = cfg.model.params.get("conditioning_key", None)
    print("model.conditioning_key (effective):", getattr(model, "conditioning_key", None))

    unet = getattr(getattr(model, "model", None), "diffusion_model", None)
    has_st = (unet is not None) and any(m.__class__.__name__.endswith("SpatialTransformer")
                                        for m in unet.modules())
    print("has SpatialTransformer:", has_st)

    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")
        pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        fixed, dropped = _fix_or_drop_mismatched_keys(model, sd)
        print(f"[load] attention params fixed: {fixed}, dropped: {dropped}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.to(device).eval()
    _dbg(debug, "[dbg] model loaded on", device)
    if debug:
        _maybe_print_model_keys(model, debug=True)
    return model


def save_imgs(x, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
    for i, img in enumerate(x):
        Image.fromarray(img.numpy()).save(os.path.join(outdir, f"{prefix}_{i:04d}.png"))


# -------------------------------
# Conditioning helpers
# -------------------------------
def make_class_conditioning(model, class_id: int, batch: int, device: torch.device, debug=False):
    """
    Builds class-label conditioning compatible with cross-attention pipelines.
    Returns:
      - for crossattn: {"c_crossattn": <tensor [B, T, C]>}
      - otherwise:     <tensor ...>
    """
    class_ids = torch.full((batch,), int(class_id), device=device, dtype=torch.long)

    key = getattr(model, "cond_stage_key", None) or "class_label"
    batch_dict = {key: class_ids}

    _dbg(debug, f"[dbg] make_class_conditioning: key='{key}', class_id={class_id}, batch={batch}")
    _tstats(class_ids, "class_ids", debug)

    # Obtain embedding tensor
    if hasattr(model, "get_learned_conditioning"):
        cond_embed = model.get_learned_conditioning(batch_dict)   # typically [B, T, C]
    elif hasattr(model, "cond_stage_model"):
        cond_embed = model.cond_stage_model(batch_dict)
    else:
        raise RuntimeError("Model has no conditioning interface for class labels.")

    _tstats(cond_embed, "cond_embed(class)", debug)

    # Wrap for cross-attn as dict of tensors (NOT list-wrapped)
    if getattr(model, "conditioning_key", None) == "crossattn":
        return {"c_crossattn": cond_embed}
    return cond_embed


def make_uncond(model, like_cond, batch: int, device: torch.device, debug=False):
    """
    Return an unconditional conditioning TENSOR when using cross-attn, because
    your DDIMSampler concatenates tensors: torch.cat([unconditional_conditioning, c]).
    """
    _dbg(debug, f"[dbg] make_uncond: batch={batch}, like_cond_type={type(like_cond)}")

    # If the model can produce its own unconditional embedding, use it
    if hasattr(model, "get_unconditional_conditioning"):
        uc = model.get_unconditional_conditioning(batch)  # tensor [B, T, C] for crossattn
        _tstats(uc, "uc(model.get_unconditional_conditioning)", debug)
        return uc  # <-- TENSOR (not dict)

    # Otherwise mirror structure of the *tensor value* inside cond
    if isinstance(like_cond, dict) and "c_crossattn" in like_cond:
        emb = like_cond["c_crossattn"]           # tensor [B, T, C]
        uc = torch.zeros_like(emb, device=device)  # <-- TENSOR
        _tstats(emb, "like_cond[c_crossattn]", debug)
        _tstats(uc, "uc(zeros_like)", debug)
        return uc

    if torch.is_tensor(like_cond):
        uc = torch.zeros_like(like_cond, device=device)
        _tstats(like_cond, "like_cond(tensor)", debug)
        _tstats(uc, "uc(zeros_like)", debug)
        return uc

    raise RuntimeError(f"Unsupported conditioning type for uncond: {type(like_cond)}")


@torch.no_grad()
def infer_latent_shape_from_vae(model, H, W, device, debug=False):
    """Run a dummy encode to get (C, h, w) so decode returns exactly HxWx3."""
    x = torch.zeros(1, 3, H, W, device=device)
    _tstats(x, "dummy_x(for latent shape)", debug, extra=f"(H={H}, W={W})")
    z = model.encode_first_stage(x)
    # CompVis VAEs return a DiagonalGaussianDistribution; get tensor latents
    if hasattr(model, "get_first_stage_encoding"):
        z = model.get_first_stage_encoding(z)  # -> (1, C, h, w)
    _tstats(z, "dummy_z(first_stage_encoding)", debug)
    return z.shape[1], z.shape[2], z.shape[3]  # C, h, w


def make_strain_conditioning(model, strain: float, batch: int, device: torch.device,
                             key: str = "strain_frac", debug=False):
    """
    Builds strain conditioning compatible with cross-attention pipelines.
    Returns:
      - for crossattn: {"c_crossattn": <tensor [B, T, C]>}
      - otherwise:     <tensor ...>
    """
    por = torch.full((batch,), float(strain), device=device, dtype=torch.float32)

    # StrainEmbedder expects batch[key] and does [:, None]
    batch_dict = {key: por}

    _dbg(debug, f"[dbg] make_strain_conditioning: key='{key}', strain={strain}, batch={batch}")
    _tstats(por, "strain_tensor(por)", debug)

    if hasattr(model, "get_learned_conditioning"):
        cond_embed = model.get_learned_conditioning(batch_dict)
    elif hasattr(model, "cond_stage_model"):
        cond_embed = model.cond_stage_model(batch_dict)
    else:
        raise RuntimeError("Model has no conditioning interface for Strain.")

    _tstats(cond_embed, "cond_embed(strain)", debug)

    if getattr(model, "conditioning_key", None) == "crossattn":
        return {"c_crossattn": cond_embed}
    return cond_embed


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--strain", type=float, required=True)
    ap.add_argument("--strain_key", default="strain_frac",
                    help="Key expected by the strain embedder (usually strain_frac)")
    ap.add_argument("--outdir", default="samples_poro")

    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None, help="Optional fixed seed; random if not set")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--scale", type=float, default=3.0)          # CFG guidance scale
    ap.add_argument("--sampler", choices=["ddim", "plms"], default="ddim")
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Enable verbose debug prints to diagnose conditioning/latents.")
    args = ap.parse_args()

    DEBUG = bool(args.debug)

    # Seed
    if args.seed is None:
        args.seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"[info] Random seed selected: {args.seed}")
    else:
        print(f"[info] Using fixed seed: {args.seed}")
    seed_everything(args.seed)
    _dbg(DEBUG, f"[dbg] seed_everything({args.seed})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = int(args.n)
    poro = float(args.strain)

    _dbg(DEBUG, f"[dbg] device={device}, batch(B)={B}, steps={args.steps}, eta={args.eta}, scale={args.scale}, sampler={args.sampler}")
    _dbg(DEBUG, f"[dbg] H={args.H}, W={args.W}, strain={poro}, strain_key='{args.strain_key}'")

    # Load model
    model = load_model(args.config, args.ckpt, device, debug=DEBUG)

    # Sampler
    sampler = DDIMSampler(model) if args.sampler == "ddim" else PLMSSampler(model)
    _dbg(DEBUG, f"[dbg] sampler={sampler.__class__.__name__}")

    # ---- latent shape inferred directly from VAE ----
    C, h, w = infer_latent_shape_from_vae(model, args.H, args.W, device, debug=DEBUG)
    shape = (int(C), int(h), int(w))
    print(f"[shape] inferred from VAE: C={C}, h={h}, w={w}  => recon {args.H}x{args.W}")
    _dbg(DEBUG, f"[dbg] sampler latent shape={shape}")

    # Conditioning
    cond = make_strain_conditioning(model, poro, B, device, key=args.strain_key, debug=DEBUG)

    use_cfg = float(args.scale) > 1.0
    uc = make_uncond(model, cond, B, device, debug=DEBUG) if use_cfg else None

    k = next(iter(cond)) if isinstance(cond, dict) else None
    print("COND kind:", type(cond), "inner:", (cond[k].shape if k else getattr(cond, 'shape', None)))
    print("UC   kind:", type(uc),   "shape:", getattr(uc, 'shape', None))

    # Convert conditioning to tensors for sampler.cat([uc, c])
    cond_t = cond["c_crossattn"] if isinstance(cond, dict) and "c_crossattn" in cond else cond
    uc_t   = uc  # already a tensor (same [B, T, C] shape)

    _tstats(cond_t, "cond_t(for sampler)", DEBUG)
    _tstats(uc_t, "uc_t(for sampler)", DEBUG)
    if use_cfg:
        _assert_same_shape(cond_t, uc_t, "cond_t", "uc_t", DEBUG)

    # Optional: warn if cond seems constant across batch (often fine, but useful signal)
    if DEBUG and torch.is_tensor(cond_t):
        with torch.no_grad():
            if cond_t.ndim >= 1 and B > 1:
                # compare first and last elements
                diff = (cond_t[0] - cond_t[-1]).abs().mean().item()
                print(f"[dbg] mean|cond_t[0]-cond_t[-1]| = {diff:.6g} (batch-const is OK if you intended it)")

    # Sampling
    _dbg(DEBUG, "[dbg] starting sampler.sample(...)")
    t0 = time.time()

    if args.autocast and device.type == "cuda":
        from torch.cuda.amp import autocast
        with autocast():
            samples_z, _ = sampler.sample(
                S=int(args.steps),
                conditioning=cond_t,
                batch_size=B,
                shape=shape,
                eta=float(args.eta),
                x_T=None,
                unconditional_guidance_scale=(float(args.scale) if use_cfg else 1.0),
                unconditional_conditioning=(uc_t if use_cfg else None),
            )
    else:
        samples_z, _ = sampler.sample(
            S=int(args.steps),
            conditioning=cond_t,
            batch_size=B,
            shape=shape,
            eta=float(args.eta),
            x_T=None,
            unconditional_guidance_scale=(float(args.scale) if use_cfg else 1.0),
            unconditional_conditioning=(uc_t if use_cfg else None),
        )

    _dbg(DEBUG, f"[dbg] sampler.sample done in {time.time()-t0:.3f}s")
    _tstats(samples_z, "samples_z(latents)", DEBUG)

    # Decode
    _dbg(DEBUG, "[dbg] decoding latents")
    x = model.decode_first_stage(samples_z.float())
    if True:  # or make it a flag like --force_grayscale
        # Convert to luminance in [-1,1] space
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        x = torch.cat([gray, gray, gray], dim=1)
    _tstats(x, "decoded_x", DEBUG)

    # Extra RGB diagnostics (only prints)
    if DEBUG and torch.is_tensor(x) and x.ndim == 4 and x.shape[1] == 3:
        with torch.no_grad():
            # compare channels (after decode, still in [-1,1])
            rg = (x[:, 0] - x[:, 1]).abs().mean().item()
            gb = (x[:, 1] - x[:, 2]).abs().mean().item()
            rb = (x[:, 0] - x[:, 2]).abs().mean().item()
            print(f"[dbg] mean|R-G|={rg:.6g} mean|G-B|={gb:.6g} mean|R-B|={rb:.6g} (decoded domain)")

            # per-channel stats
            for ci, cname in enumerate(["R", "G", "B"]):
                _tstats(x[:, ci:ci+1], f"decoded_x[{cname}]", DEBUG)

    # Save
    run_tag = args.prefix or f"s{args.seed}_t{int(time.time())}"
    save_prefix = f"poro{args.strain}_{args.H}_{args.W}_{run_tag}"
    save_imgs(x, args.outdir, save_prefix)
    print(f"[done] saved to {args.outdir} with prefix {save_prefix}")