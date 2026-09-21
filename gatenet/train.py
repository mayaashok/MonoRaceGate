"""
train.py - training loop for GateNet (MonoRace, arXiv:2601.15222).

Paper references (GateNet "Training setup"):
  - Resolution 384 x 384, f = 4 (see model.py).
  - Xavier uniform init is applied inside GateNet itself (model.py), so nothing to do here.
  - 100 epochs, AdamW optimizer, batch size 16.
  - Base learning rate 1e-3, reduced by sqrt(0.1) at epochs 10, 33, 66 and 90
    -> final learning rate 1e-5 (verified: 1e-3 * sqrt(0.1)^4 = 1e-5).
  - Per-output loss: L_i = Dice(y_i, gt_i) + 2 * BCE(y_i, gt_i), where gt_i is the
    ground-truth mask downscaled to y_i's resolution.
  - Combined loss: L_tot = 4*L0 + 2*L1 + L2 + L3 + L4

The weights [1, 1, 1, 2, 4] are applied in order of preds = [y0, y1, y2, y3, y4] going from
LOWEST resolution (y0, bottleneck side) to HIGHEST (y4, full 384x384, the deployed output).
Matches paper saying to apply scaling factors to "emphasize higher-resolution predictions" even
if opposite of formula (assuming L0 = y4, L1 = y3, L2, = y2, L3 = y1, L4 = y0).

Not specified by the paper (my choices, confirm with your lab):
  - AdamW's weight_decay and betas (left at PyTorch defaults).
  - Dice loss epsilon (for numerical stability, doesn't noticeably affect training).
  - Which single metric to use for "best model" checkpointing (I use validation IoU on y4).
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import GateNet

from rich.progress import Progress

# Weight on each output's loss, in [y0 (coarsest) ... y4 (finest)] order (opposite of paper's L_tot formula)
WEIGHTS = [1, 1, 1, 2, 4]


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------
def dice_loss(pred, gt, eps=1e-6):
    """Soft Dice loss between a predicted probability map and a binary ground-truth mask.

    Dice = (2 * intersection) / (sum of areas). Loss = 1 - Dice, so 0 is a perfect match.
    `eps` avoids a 0/0 when both pred and gt are entirely empty (no gate in the crop).
    """
    pred = pred.reshape(pred.shape[0], -1)   # flatten each sample to a 1D vector
    gt = gt.reshape(gt.shape[0], -1)
    intersection = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return (1 - dice).mean()                 # average over the batch


def gatenet_loss(preds, mask, weights=WEIGHTS):
    """Combined GateNet loss: L_tot = 4*L0 + 2*L1 + L2 + L3 + L4, L_i = Dice_i + 2*BCE_i.

    preds : list [y0..y4] of predicted probability maps from GateNet.forward().
    mask  : ground-truth mask at full (384x384) resolution, shape (B, 1, H, W), values in {0, 1}.
    """
    total = 0.0
    per_output = []  # kept for logging/debugging, not used in the backward pass directly
    for y_i, w_i in zip(preds, weights):
        # Downscale the full-resolution ground truth to this output's resolution.
        # "nearest" keeps the mask binary (no blurred/fractional edge values).
        gt_i = F.interpolate(mask, size=y_i.shape[-2:], mode="nearest")

        # Dice loss (region overlap)
        d = dice_loss(y_i, gt_i)
        # Binary cross-entropy loss (per-pixel classification), weighted x2 as in the paper
        b = F.binary_cross_entropy(y_i, gt_i)

        l_i = d + 2 * b
        per_output.append(l_i.item())
        total = total + w_i * l_i

    return total, per_output


# ----------------------------------------------------------------------------
# Metric: IoU on the full-resolution output (y4), for validation only
# ----------------------------------------------------------------------------
@torch.no_grad()
def iou_score(pred, gt, threshold=0.5, eps=1e-6):
    """Intersection-over-Union between a thresholded prediction and the ground-truth mask."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * gt).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


# ----------------------------------------------------------------------------
# One epoch of training / validation
# ----------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    
    with Progress() as progress:
        task = progress.add_task("Training..." if train else "Validation...",total=len(loader))
        
        with torch.set_grad_enabled(train):
            for img, mask in loader:
                img, mask = img.to(device), mask.to(device)

                preds = model(img)                            # [y0..y4]
                loss, _ = gatenet_loss(preds, mask)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_iou += iou_score(preds[-1], mask)        # y4 is the highest-resolution, deployed output
                n_batches += 1

                progress.update(task, advance=1)

    return total_loss / n_batches, total_iou / n_batches


# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--val_folders", nargs="+", default=["dhl_s1_f1", "marina_s1_f1"])
    ap.add_argument("--crop_mode", default="resize", choices=["resize", "crop"])
    ap.add_argument("--epochs", type=int, default=100)          # paper: 100 epochs
    ap.add_argument("--batch_size", type=int, default=16)       # paper: batch size 16
    ap.add_argument("--lr", type=float, default=1e-3)           # paper: base learning rate 1e-3
    ap.add_argument("--f", type=int, default=4)                 # paper: f = 4
    ap.add_argument("--out_dir", default="./checkpoints")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Data ----
    train_ds, val_ds = build_datasets(args.data_root, tuple(args.val_folders), crop_mode=args.crop_mode)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # ---- Model ----
    model = GateNet(f=args.f).to(device)   # Xavier init already applied inside GateNet.__init__

    # ---- Optimizer + LR schedule ----
    # AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Learning rate reduced by sqrt(0.1) at epochs 10, 33, 66, 90 -> final lr = 1e-5
    scheduler = MultiStepLR(optimizer, milestones=[10, 33, 66, 90], gamma=0.1 ** 0.5)

    # ---- Training loop ----
    best_val_iou = 0.0

    with Progress() as progress:
        epoch_task = progress.add_task("[green] Epochs", total=args.epochs)

        for epoch in range(args.epochs):
            t0 = time.time()

            train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, train=True)
            val_loss, val_iou = run_epoch(model, val_loader, optimizer, device, train=False)

            scheduler.step()   # advance the LR schedule by one epoch
            lr_now = optimizer.param_groups[0]["lr"]

            dt = time.time() - t0

            progress.update(epoch_task,advance=1,
                description=(
                    f"[green]Epoch {epoch+1}/{args.epochs} "
                    f"lr={lr_now:.2e} "
                    f"train_loss={train_loss:.4f} "
                    f"train_iou={train_iou:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_iou={val_iou:.4f} "
                    f"({dt:.1f}s)"
                )
            )

            # Save the most recent weights every epoch (cheap safety net if training is interrupted)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))

            # Save a separate copy whenever validation IoU improves
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
                progress.console.print(f"  -> new best val_iou {best_val_iou:.4f}, saved best.pt")

    print(f"done. best val_iou = {best_val_iou:.4f}")


if __name__ == "__main__":
    main()