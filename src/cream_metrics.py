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
    u2u_w = model.u_to_CY.u2u_model[0].weight.detach()   # (num_exogenous, backbone_dim=512/2048)
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
    """Register forward hook on layer4, return (handle, storage_dict).

    x_to_u is concept_extractor = nn.Sequential(conv1, bn1, relu, maxpool,
    layer1, layer2, layer3, layer4, avgpool, flatten)
    layer4 is at index 7.
    """
    storage = {}
    def hook(module, inp, out):
        storage['feat'] = out  # (B, C, H, W)
    layer4 = model.x_to_u[7]  # layer4 in the Sequential
    handle = layer4.register_forward_hook(hook)
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

def _compute_class_cam(feat_map: Tensor, cam_weights: Tensor, last_layer_weight: Tensor, y_idx: int) -> Tensor:
    """
    Class-level CAM: weighted sum of concept CAMs by last_layer weights for class y_idx.
    last_layer_weight: (num_classes, num_concepts)  — only concept part (no side channel)
    Returns: (B, H, W)
    """
    class_weights = last_layer_weight[y_idx]                        # (K,)
    class_weights = F.relu(class_weights)                           # keep positive contributions
    class_weights = class_weights / (class_weights.sum() + 1e-8)    # normalize
    # concept cams: (B, K, H, W)
    cams = _compute_concept_cams(feat_map, cam_weights)
    # weighted sum over concepts
    class_cam = (cams * class_weights[None, :, None, None]).sum(dim=1)  # (B, H, W)
    # normalize to [0,1]
    cam_min = class_cam.flatten(1).min(dim=1).values[:, None, None]
    cam_max = class_cam.flatten(1).max(dim=1).values[:, None, None]
    return (class_cam - cam_min) / (cam_max - cam_min + 1e-8)


@torch.no_grad()
def compute_adi(
    model,
    dataloader,
    device: torch.device,
    img_size: int = 224,
) -> Dict:
    """
    Sc — concept-level ADI: mask with per-concept CAM, measure concept score change
    Sy — class-level ADI:   mask with class CAM, measure class score change

    AD  (Average Drop)     ↓  lower is better
    AI  (Average Increase) ↓  lower is better
    AG  (Average Gain)     ↑  higher is better
    """
    model.eval()
    model.to(device)

    cam_weights = _get_concept_cam_weights(model).to(device)  # (K, backbone_dim)

    # last_layer weight for class CAM — shape (num_classes, num_concepts+num_side)
    # grab only the concept columns
    num_concepts = model.u_to_CY.num_concepts
    last_layer_w = model.u_to_CY.last_layer.weight.detach().to(device)  # (C, K+side)
    last_layer_w_concepts = last_layer_w[:, :num_concepts]               # (C, K)

    sc_drops, sc_incs, sc_gains = [], [], []
    sy_drops, sy_incs, sy_gains = [], [], []

    for batch in dataloader:
        x, c_true, y_true = batch
        x, c_true, y_true = x.to(device), c_true.to(device), y_true.to(device)

        handle, storage = _hook_layer4(model)
        y_logits, c_pred = model(x)
        handle.remove()

        feat_map = storage['feat']                                        # (B, backbone_dim, H, W)
        cams     = _compute_concept_cams(feat_map, cam_weights)           # (B, K, 7, 7)
        cams_up  = F.interpolate(cams, size=(img_size, img_size),
                                 mode='bilinear', align_corners=False)    # (B, K, 224, 224)

        c_scores_orig = torch.sigmoid(c_pred)                             # (B, K)
        y_scores_orig = torch.softmax(y_logits, dim=1)                    # (B, num_classes)
        y_pred        = y_logits.argmax(dim=1)                            # (B,)

        for b in range(x.shape[0]):
            active_concepts = c_true[b].nonzero(as_tuple=True)[0].tolist()
            if not active_concepts:
                continue

            # ── Sc: per-concept ────────────────────────────────────────────
            for ci in active_concepts:
                concept_mask = cams_up[b, ci, None, None, :]              # (1, 1, 224, 224)
                masked_img   = x[b:b+1] * concept_mask
                _, c_masked  = model(masked_img)
                c_s_masked   = torch.sigmoid(c_masked)

                s0 = c_scores_orig[b, ci].item()
                sm = c_s_masked[0, ci].item()

                sc_drops.append(max(0.0, (s0 - sm) / (s0 + 1e-8)))
                sc_incs.append(float(sm > s0))
                sc_gains.append(max(0.0, sm - s0))

            # ── Sy: class-level ────────────────────────────────────────────
            yi = y_pred[b].item()
            class_cam_b = _compute_class_cam(
                feat_map[b:b+1], cam_weights, last_layer_w_concepts, yi
            )   # (1, 7, 7)
            class_cam_up = F.interpolate(
                class_cam_b.unsqueeze(0), size=(img_size, img_size),
                mode='bilinear', align_corners=False
            )[0, 0]                                                        # (224, 224)

            masked_img_y  = x[b:b+1] * class_cam_up[None, None, :, :]
            y_logits_m, _ = model(masked_img_y)
            y_s_masked     = torch.softmax(y_logits_m, dim=1)

            s0y = y_scores_orig[b, yi].item()
            smy = y_s_masked[0, yi].item()

            sy_drops.append(max(0.0, (s0y - smy) / (s0y + 1e-8)))
            sy_incs.append(float(smy > s0y))
            sy_gains.append(max(0.0, smy - s0y))

    return {
        'Sc_AD': round(float(np.mean(sc_drops)), 4),
        'Sc_AI': round(float(np.mean(sc_incs)),  4),
        'Sc_AG': round(float(np.mean(sc_gains)), 4),
        'Sy_AD': round(float(np.mean(sy_drops)), 4),
        'Sy_AI': round(float(np.mean(sy_incs)),  4),
        'Sy_AG': round(float(np.mean(sy_gains)), 4),
        'n_concept_samples': len(sc_drops),
        'n_class_samples':   len(sy_drops),
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
    save_pt: bool = True,
):
    """
    For the first n_images test samples, save per-concept CAM overlays as PNGs.
    Output: save_dir/sample_{i}/concept_{name}.png and sample_{i}_heatmaps.pt.
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
                safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
                fname = os.path.join(sample_dir, f'concept_{ci:03d}_{safe_name}.png')
                plt.savefig(fname, dpi=80, bbox_inches='tight')
                plt.close()

            if save_pt:
                pt_path = os.path.join(sample_dir, f'sample_{saved}_heatmaps.pt')
                torch.save(
                    {
                        'sample_index': saved,
                        'image_normalized': x[b].detach().cpu(),
                        'image_display': torch.from_numpy(img).permute(2, 0, 1),
                        'true_concepts': c_true[b].detach().cpu(),
                        'active_concept_indices': active_concepts,
                        'concept_names': concept_names,
                        'cams_layer4': cams[b].detach().cpu(),
                        'cams_up': cams_up[b].detach().cpu(),
                    },
                    pt_path,
                )

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
        print(f"  Sc: AD={adi_results['Sc_AD']} AI={adi_results['Sc_AI']} AG={adi_results['Sc_AG']}")
        print(f"  Sy: AD={adi_results['Sy_AD']} AI={adi_results['Sy_AI']} AG={adi_results['Sy_AG']}")

    return {**nec_results, **adi_results}
