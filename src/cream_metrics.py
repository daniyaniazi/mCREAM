"""
Post-hoc evaluation metrics for CREAM:
  - NEC / ANEC  (intervention efficiency)
  - CAM         (per-concept saliency maps via layer4 hook)
  - ADI         (Average Drop / Increase / Gain)

Usage:
    from src.cream_metrics import evaluate_all
    results = evaluate_all(model, datamodule, device, budgets=[5,10,15,20,25,30])
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple
import pytorch_lightning as pl


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_concept_cam_weights(model) -> Tensor:
    """
    Compute combined projection weights: u2c @ u2u  →  (num_concepts, backbone_dim)
    Works when u2c_model has 0 hidden layers (MaskedLinear / single Linear).
    """
    u2u_w = model.u_to_CY.u2u_model[0].weight.detach()   # (num_exogenous, backbone_dim)
    # u2c_model is a MaskedMLP / MaskedLinear — grab the first (and only) linear weight
    u2c_layer = model.u_to_CY.u2c_model
    # Walk through Sequential / MaskedMLP to find first weight matrix
    u2c_w = None
    for m in u2c_layer.modules():
        if hasattr(m, 'weight') and m.weight is not None:
            u2c_w = m.weight.detach()   # (num_concepts, num_exogenous - num_side_channel)
            break
    if u2c_w is None:
        raise RuntimeError("Could not find weight in u2c_model")
    num_concepts = u2c_w.shape[0]
    num_side     = model.u_to_CY.num_side_channel
    # u2c only sees Uc = u2u output[:, :num_exogenous-num_side]
    u2u_w_concept = u2u_w[:u2u_w.shape[0]-num_side, :]   # (340, 512/2048)
    cam_weights = u2c_w @ u2u_w_concept                   # (num_concepts, backbone_dim)
    return cam_weights                                     # (85, 512)


def _hook_layer4(model):
    """Register forward hook on layer4, return (handle, storage_dict)."""
    storage = {}
    def hook(module, inp, out):
        storage['feat'] = out  # (B, C, H, W)
    handle = model.x_to_u.resnet.layer4.register_forward_hook(hook)
    return handle, storage


def _compute_concept_cams(feat_map: Tensor, cam_weights: Tensor) -> Tensor:
    """
    feat_map   : (B, C, H, W)
    cam_weights: (num_concepts, C)
    returns    : (B, num_concepts, H, W)  — relu-normalized to [0,1]
    """
    B, C, H, W = feat_map.shape
    K = cam_weights.shape[0]
    # (B, C, H*W) × (C, K) → (B, K, H*W)
    cam = torch.einsum('bchw,kc->bkhw', feat_map, cam_weights)
    cam = F.relu(cam)
    # normalize each map to [0, 1]
    cam_min = cam.flatten(2).min(dim=2).values[:, :, None, None]
    cam_max = cam.flatten(2).max(dim=2).values[:, :, None, None]
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam   # (B, num_concepts, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# NEC / ANEC
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_nec_anec(
    model,
    dataloader,
    device: torch.device,
    budgets: List[int] = [5, 10, 15, 20, 25, 30],
    percentile_df=None,
) -> Dict:
    """
    NEC-k: % of initially wrong predictions that become correct
           after intervening on the k highest-error concepts.
    ANEC : mean NEC across all budgets.
    """
    model.eval()
    model.to(device)

    # Collect per-sample: was_wrong, becomes_correct_at_k
    initially_wrong = []
    corrected_at = {k: [] for k in budgets}

    # load percentile df into model if provided
    if percentile_df is not None:
        model.intervention_percentile_df = percentile_df

    for batch in dataloader:
        x, c_true, y_true = batch
        x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)

        # baseline prediction (no intervention)
        y_logits, c_pred = model(x)
        y_hat_base = y_logits.argmax(dim=1)
        wrong_mask = (y_hat_base != y_true)   # (B,)

        if wrong_mask.sum() == 0:
            continue

        # rank concepts by prediction error per sample  (B, num_concepts)
        concept_error = (c_pred - c_true.float()).abs()   # (B, K)

        for k in budgets:
            # top-k error concepts per sample
            top_k_idx = concept_error.topk(k, dim=1).indices   # (B, k)

            # build intervention mask
            B, K = c_pred.shape
            mask = torch.zeros(B, K, dtype=torch.bool, device=device)
            mask.scatter_(1, top_k_idx, True)

            # intervene: replace predicted concepts with GT
            c_intervened = c_pred.clone()
            # convert GT binary to soft if needed
            if hasattr(model, 'intervention_percentile_df') and \
               len(model.intervention_percentile_df) == K:
                p5  = torch.tensor(model.intervention_percentile_df['5th_percentile'].values,
                                   dtype=torch.float, device=device)
                p95 = torch.tensor(model.intervention_percentile_df['95th_percentile'].values,
                                   dtype=torch.float, device=device)
                c_gt_soft = c_true.float() * p95 + (1 - c_true.float()) * p5
            else:
                c_gt_soft = c_true.float()

            c_intervened[mask] = c_gt_soft[mask]

            # re-run last layer with intervened concepts
            num_side = model.u_to_CY.num_side_channel
            # we need Uy for side channel — re-run full forward to get it
            # use forward_with_interventions
            model.u_to_CY.group_interventions = False
            y_int, _ = model.u_to_CY.forward_with_interventions(
                x=model.x_to_u(x),
                true_concepts=c_true.float(),
                intervention_mask=mask,
            )
            y_hat_int = y_int.argmax(dim=1)

            for i in range(B):
                if wrong_mask[i]:
                    corrected_at[k].append(int(y_hat_int[i] == y_true[i]))

        for i in range(B):
            initially_wrong.append(int(wrong_mask[i]))

    n_wrong = sum(initially_wrong)
    nec = {}
    for k in budgets:
        corr = sum(corrected_at[k])
        nec[f'NEC-{k}'] = round(corr / max(n_wrong, 1), 4)

    anec = round(float(np.mean([nec[f'NEC-{k}'] for k in budgets])), 4)
    return {'NEC': nec, 'ANEC': anec, 'n_wrong': n_wrong}


# ──────────────────────────────────────────────────────────────────────────────
# ADI  (Average Drop / Increase / Gain)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_adi(
    model,
    dataloader,
    device: torch.device,
    img_size: int = 224,
) -> Dict:
    """
    For each active concept in each sample:
      1. Compute CAM for that concept
      2. Mask image with CAM
      3. Compare concept score before/after masking

    AD  (Average Drop)     ↓  lower is better
    AI  (Average Increase) ↓  lower is better
    AG  (Average Gain)     ↑  higher is better
    """
    model.eval()
    model.to(device)

    cam_weights = _get_concept_cam_weights(model).to(device)  # (K, C)

    drops, increases, gains = [], [], []
    total_active = 0

    for batch in dataloader:
        x, c_true, y_true = batch
        x, c_true = x.to(device), c_true.to(device)

        # ── forward with layer4 hook ──────────────────────────────────────
        handle, storage = _hook_layer4(model)
        _, c_pred = model(x)
        handle.remove()

        feat_map = storage['feat']                         # (B, C, H, W)
        cams = _compute_concept_cams(feat_map, cam_weights)  # (B, K, 7, 7)

        B, K, H, W = cams.shape

        # upsample CAM to image size
        cams_up = F.interpolate(cams, size=(img_size, img_size),
                                mode='bilinear', align_corners=False)  # (B, K, 224, 224)

        # original concept scores (sigmoid)
        c_scores_orig = torch.sigmoid(c_pred)   # (B, K)

        for b in range(B):
            active_concepts = c_true[b].nonzero(as_tuple=True)[0]
            for ci in active_concepts:
                ci = ci.item()
                mask = cams_up[b, ci]          # (224, 224) in [0,1]
                # mask image: keep only salient region
                masked_img = x[b:b+1] * mask[None, None, :, :]  # (1, 3, 224, 224)

                # forward masked image
                handle2, storage2 = _hook_layer4(model)
                _, c_pred_masked = model(masked_img)
                handle2.remove()

                c_scores_masked = torch.sigmoid(c_pred_masked)  # (1, K)

                s_orig   = c_scores_orig[b, ci].item()
                s_masked = c_scores_masked[0, ci].item()

                drop = max(0.0, (s_orig - s_masked) / (s_orig + 1e-8))
                inc  = float(s_masked > s_orig)
                gain = max(0.0, s_masked - s_orig)

                drops.append(drop)
                increases.append(inc)
                gains.append(gain)
                total_active += 1

    return {
        'AD':  round(float(np.mean(drops)),     4),
        'AI':  round(float(np.mean(increases)), 4),
        'AG':  round(float(np.mean(gains)),     4),
        'n_active_concepts': total_active,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Saliency map visualization
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_concept_saliency_maps(
    model,
    dataloader,
    device: torch.device,
    concept_names: List[str],
    save_dir: str,
    n_images: int = 10,
    img_size: int = 224,
):
    """
    For the first n_images test samples, save per-concept CAM overlays as PNGs.
    Output: save_dir/sample_{i}/concept_{name}.png
    """
    import os
    import matplotlib.pyplot as plt

    model.eval().to(device)
    cam_weights = _get_concept_cam_weights(model).to(device)  # (K, C)

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    saved = 0
    for batch in dataloader:
        if saved >= n_images:
            break
        x, c_true, _ = batch
        x = x.to(device)

        handle, storage = _hook_layer4(model)
        model(x)
        handle.remove()

        feat_map = storage['feat']                                    # (B, C, H, W)
        cams = _compute_concept_cams(feat_map, cam_weights)           # (B, K, 7, 7)
        cams_up = F.interpolate(cams, size=(img_size, img_size),
                                mode='bilinear', align_corners=False) # (B, K, 224, 224)

        for b in range(x.shape[0]):
            if saved >= n_images:
                break
            sample_dir = os.path.join(save_dir, f'sample_{saved}')
            os.makedirs(sample_dir, exist_ok=True)

            # denormalize image for display
            img = x[b].cpu() * IMAGENET_STD + IMAGENET_MEAN
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()

            active_concepts = c_true[b].nonzero(as_tuple=True)[0].tolist()

            for ci in range(len(concept_names)):
                cam_map = cams_up[b, ci].cpu().numpy()  # (224, 224)
                name = concept_names[ci]
                active = ci in active_concepts

                _, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(img); axes[0].set_title('Image'); axes[0].axis('off')
                axes[1].imshow(img)
                axes[1].imshow(cam_map, alpha=0.5, cmap='jet')
                title = f'{name} ({"active" if active else "inactive"})'
                axes[1].set_title(title); axes[1].axis('off')
                plt.tight_layout()
                fname = os.path.join(sample_dir, f'concept_{ci:03d}_{name}.png')
                plt.savefig(fname, dpi=80, bbox_inches='tight')
                plt.close()

            saved += 1

    print(f"Saved saliency maps for {saved} samples to {save_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Combined runner
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    model,
    datamodule: pl.LightningDataModule,
    device: torch.device,
    budgets: List[int] = [5, 10, 15, 20, 25, 30],
    run_adi: bool = True,
) -> Dict:
    """Run NEC/ANEC and optionally ADI on the test set."""
    datamodule.setup(stage='test')
    loader = datamodule.test_dataloader()

    print("Computing NEC/ANEC...")
    nec_results = compute_nec_anec(model, loader, device, budgets)
    print(f"  ANEC={nec_results['ANEC']}  NEC: {nec_results['NEC']}")

    adi_results = {}
    if run_adi:
        print("Computing ADI (may take a while)...")
        adi_results = compute_adi(model, loader, device)
        print(f"  AD={adi_results['AD']}  AI={adi_results['AI']}  AG={adi_results['AG']}")

    return {**nec_results, **adi_results}
