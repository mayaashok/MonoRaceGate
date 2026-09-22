"""
inference.py - run the trained GateNet on validation images and save a
side-by-side comparison: input image | predicted mask | ground-truth mask | overlay.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

# run opencv and torch on single threads to prevent conflict segfault
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
torch.set_num_threads(1)

from dataset import build_datasets
from model import GateNet

from rich.progress import Progress

def to_uint8_bgr(img_tensor):
    """(3,H,W) float tensor in [0,1], RGB -> (H,W,3) uint8, BGR (for cv2 saving)."""
    img = img_tensor.numpy().transpose(1, 2, 0)          # CHW -> HWC
    img = (img[..., ::-1] * 255).astype(np.uint8)        # RGB -> BGR, scale to 0-255
    return img

# ----------------------------------------------------------------------------
# Sanity check: python gatenet/inference.py
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/best.pt")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--val_folders", nargs="+", default=["dhl_s1_f1", "marina_s1_f1"])
    ap.add_argument("--crop_mode", default="resize", choices=["resize", "crop"])
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--n", type=int, default=8)  # how many validation samples to show
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load the trained model ----
    print("loading model...")
    model = GateNet(f=args.f).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()   # turns off batch-norm running-stat updates and any dropout
    print("model loaded")

    # ---- Load validation data (no augmentation, since train=False inside build_datasets) ----
    print("building dataset...")
    _, val_ds = build_datasets(args.data_root, tuple(args.val_folders), crop_mode=args.crop_mode)
    print(f"val: {len(val_ds)} images, checkpoint: {args.checkpoint}, device: {device}")
    print("dataset built")

    n = min(args.n, len(val_ds))
    indices = np.linspace(0, len(val_ds) - 1, n, dtype=int)  # spread picks across the val set

    rows = []
    ious = []
    with torch.no_grad():   # no gradients needed for inference
        with Progress() as progress:
            task = progress.add_task("[green] Visualization Rows Built...", total=n)
            for idx in indices:
                img, gt_mask = val_ds[idx]                  # img: (3,384,384), gt_mask: (1,384,384)
                img_batch = img.unsqueeze(0).to(device)     # add batch dim -> (1,3,384,384)

                preds = model(img_batch)                    # [y0..y4]
                pred = preds[-1][0, 0].cpu()                # y4 (highest-res, deployed output), drop batch+channel dims

                pred_bin = (pred > args.threshold).float()
                gt = gt_mask[0]                             # drop channel dim

                # IoU for this single sample (same formula as train.py's iou_score)
                inter = (pred_bin * gt).sum()
                union = pred_bin.sum() + gt.sum() - inter
                iou = ((inter + 1e-6) / (union + 1e-6)).item()
                ious.append(iou)

                # ---- Build the visualization row ----
                img_bgr = to_uint8_bgr(img)

                pred_vis = (pred_bin.numpy() * 255).astype(np.uint8)
                pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

                gt_vis = (gt.numpy() * 255).astype(np.uint8)
                gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

                # Overlay: prediction in red, ground truth in green, overlap shows as yellow-ish
                overlay = img_bgr.copy()
                overlay[..., 2] = np.maximum(overlay[..., 2], pred_bin.numpy() * 255)   # red channel = prediction
                overlay[..., 1] = np.maximum(overlay[..., 1], gt.numpy() * 255)          # green channel = ground truth

                # Label this row with its IoU
                row = np.hstack([img_bgr, pred_vis, gt_vis, overlay])
                cv2.putText(row, f"IoU: {iou:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                rows.append(row)
                progress.update(task, advance=1)

    grid = np.vstack(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_check.png")
    cv2.imwrite(out_path, grid)

    print(f"mean IoU over shown samples: {np.mean(ious):.4f}")
    print(f"wrote {out_path}")
    print("columns: input | predicted mask | ground-truth mask | overlay (red=pred, green=gt, yellow=both)")


if __name__ == "__main__":
    main()