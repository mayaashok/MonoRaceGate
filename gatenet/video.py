"""
video.py - run GateNet on every image in one flight folder, in the correct temporal
order, and stitch a 4-panel comparison (input | predicted mask | ground-truth mask |
overlay, with IoU) into a video, same layout as inference.py's pred_check.png but as
a playable sequence instead of one stacked image.

Usage (default: sort by the number in the filename - NOT true flight order, see below):
    python gatenet/video.py --checkpoint checkpoints/best.pt --folder data/red_s1_f1

Usage (manually verified order, e.g. gatenet/orders/red_s1_f1_order.txt):
    python gatenet/video.py --checkpoint checkpoints/best.pt --folder data/red_s1_f1 \
        --order_file gatenet/orders/red_s1_f1_order.txt

KNOWN LIMITATION when NOT using --order_file: the numeric order of img_N.png files in
this repo's labeled subset does NOT correspond to true flight/temporal order (confirmed:
consecutive numbers show the drone in very different positions, and file timestamps /
corners.csv row order don't recover the real sequence either). --order_file exists so a
manually-verified viewing order (one image number per line, e.g. from looking through the
images by hand) can be used instead, which is the only reliable way we've found to get a
sequence that actually looks like continuous flight.
"""

import argparse
import glob
import os
import re

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
torch.set_num_threads(1)

from dataset import geometric_transform
from model import GateNet


def frame_number(path):
    """Extract the integer number from a filename like '.../img_123.png' -> 123.
       Used only to get a stable, reproducible ordering (NOT a guarantee of true flight order -
       see the module docstring above).
    """
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_frames(folder):
    """Return every img_* file in `folder` that also has a matching mask_* file,
       sorted by the number in the filename. NOT true flight order - see module docstring.
    """
    img_paths = glob.glob(os.path.join(folder, "img_*"))
    pairs = []
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            continue
        folder_, base = os.path.split(img_path)
        mask_path = os.path.join(folder_, base.replace("img_", "mask_", 1))
        if os.path.isfile(mask_path):
            pairs.append((img_path, mask_path))
    pairs.sort(key=lambda p: frame_number(p[0]))
    return pairs


def list_frames_from_order_file(folder, order_path, ext=".png"):
    """Return (img_path, mask_path) pairs in the EXACT order given by `order_path`, a text
       file with one image number per line (e.g. "21\n17\n23\n..."), as produced by manually
       looking through a folder's images and writing down the true viewing order.

       A number whose img_* or mask_* file is missing is skipped with a printed warning,
       rather than crashing, so a hand-written list doesn't need to be perfectly clean.
    """
    with open(order_path) as f:
        numbers = [line.strip() for line in f if line.strip()]

    pairs = []
    skipped = []
    for n in numbers:
        img_path = os.path.join(folder, f"img_{n}{ext}")
        mask_path = os.path.join(folder, f"mask_{n}{ext}")
        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            pairs.append((img_path, mask_path))
        else:
            skipped.append(n)

    print(f"[video] order file: {len(pairs)}/{len(numbers)} frames found "
          f"({len(skipped)} skipped, missing img/mask)")
    if skipped:
        print(f"  skipped numbers: {skipped}")
    return pairs


def label_panel(panel, text):
    """Stamp a text label in the top-left corner of one panel (in place).
       A filled dark rectangle behind the text keeps it readable over any background,
       light or dark.
    """
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(panel, (0, 0), (tw + 12, th + 16), (0, 0, 0), -1)
    cv2.putText(panel, text, (6, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return panel


def make_panel_row(img_bgr, pred_bin, gt_bin, iou, panel_size):
    """Build a 2x2 grid: [input | prediction] on top, [ground truth | overlay] on bottom.
       All four panels are resized to `panel_size` = (width, height) and labeled in
       their top-left corner so it's clear which panel is which.
    """
    img_p = label_panel(cv2.resize(img_bgr, panel_size), "Input")

    pred_vis = cv2.resize(pred_bin, panel_size, interpolation=cv2.INTER_NEAREST)
    pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
    pred_vis = label_panel(pred_vis, "Predicted mask")

    if gt_bin is not None:
        gt_vis = cv2.resize(gt_bin, panel_size, interpolation=cv2.INTER_NEAREST)
    else:
        gt_vis = np.zeros(panel_size[::-1], np.uint8)   # blank if no ground truth for this frame
    gt_vis_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
    gt_vis_bgr = label_panel(gt_vis_bgr, "Ground truth")

    overlay = cv2.resize(img_bgr, panel_size).copy()
    overlay[..., 2] = np.maximum(overlay[..., 2], cv2.resize(pred_bin, panel_size, interpolation=cv2.INTER_NEAREST))
    if gt_bin is not None:
        overlay[..., 1] = np.maximum(overlay[..., 1], gt_vis)
    iou_text = f"Overlay - IoU: {iou:.3f}" if iou is not None else "Overlay - IoU: n/a"
    overlay = label_panel(overlay, iou_text)

    top_row = np.hstack([img_p, pred_vis])          # input | prediction
    bottom_row = np.hstack([gt_vis_bgr, overlay])   # ground truth | overlay
    grid = np.vstack([top_row, bottom_row])          # stack the two rows into a 2x2 grid
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/best.pt")
    ap.add_argument("--folder", required=True, help="one dataset folder, e.g. ./data/red_s1_f1")
    ap.add_argument("--crop_mode", default="resize", choices=["resize", "crop"])
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--panel_width", type=int, default=320, help="width of EACH of the 4 panels")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: gatenet/flight.mp4)")
    ap.add_argument("--order_file", default=None,
                    help="text file with one image number per line, giving a manually-verified "
                         "viewing order (e.g. gatenet/orders/red_s1_f1_order.txt). If omitted, "
                         "falls back to sorting by the number in the filename (NOT true flight order).")
    args = ap.parse_args()

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "flight.mp4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model...")
    model = GateNet(f=args.f).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("model loaded")

    if args.order_file:
        pairs = list_frames_from_order_file(args.folder, args.order_file)
    else:
        pairs = list_frames(args.folder)
        print("WARNING: no --order_file given, falling back to filename-number order, "
              "which is NOT true flight order (see module docstring).")
    print(f"using {len(pairs)} labeled frames (img+mask pairs) from {args.folder}")
    if not pairs:
        raise SystemExit(f"No usable img_*/mask_* pairs found for {args.folder}")

    writer = None
    n_written = 0
    ious = []

    with torch.no_grad():
        for img_path, mask_path in pairs:
            img_orig = cv2.imread(img_path)
            gt_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img_orig is None or gt_orig is None:
                print(f"  skipping unreadable pair: {img_path}")
                continue
            orig_h, orig_w = img_orig.shape[:2]
            if gt_orig.shape[:2] != (orig_h, orig_w):
                gt_orig = cv2.resize(gt_orig, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Deterministic (no-augmentation) preprocessing to the network's 384x384 input,
            # same as validation uses.
            img_384, _ = geometric_transform(img_orig, gt_orig, 384, args.crop_mode, train=False)
            img_tensor = torch.from_numpy(
                np.ascontiguousarray(img_384[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
            ).unsqueeze(0).to(device)

            preds = model(img_tensor)
            pred = preds[-1][0, 0].cpu().numpy()             # y4, (384,384) probability map
            pred_bin_384 = (pred > args.threshold).astype(np.uint8) * 255
            pred_native = cv2.resize(pred_bin_384, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            gt_bin_native = (gt_orig > 127).astype(np.uint8) * 255

            # IoU at native resolution, same formula used throughout the project
            p = (pred_native > 127)
            g = (gt_bin_native > 127)
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou = (inter + 1e-6) / (union + 1e-6)
            ious.append(iou)

            panel_h = int(args.panel_width * orig_h / orig_w)
            row = make_panel_row(img_orig, pred_native, gt_bin_native, iou, (args.panel_width, panel_h))

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                frame_h, frame_w = row.shape[:2]   # remember the size WE chose; the writer can't tell us later
                writer = cv2.VideoWriter(out_path, fourcc, args.fps, (frame_w, frame_h))
                print(f"video: {frame_w}x{frame_h} @ {args.fps} fps -> {out_path}")
            else:
                # Force every row to the SAME size as the first frame - this is what fixes the
                # "Failed to write frame" warnings, which happen whenever a frame's size doesn't
                # match the size the writer was created with (source images aren't all identical
                # resolution). NOTE: cv2.VideoWriter.get(CAP_PROP_FRAME_WIDTH/HEIGHT) is NOT
                # reliable for querying this back (many backends just return 0), so we keep our
                # own frame_w/frame_h variables instead of asking the writer.
                if row.shape[1] != frame_w or row.shape[0] != frame_h:
                    row = cv2.resize(row, (frame_w, frame_h))

            writer.write(row)
            n_written += 1

    if writer is not None:
        writer.release()

    print(f"wrote {n_written} frames to {out_path}")
    print(f"mean IoU across written frames: {np.mean(ious):.4f}")
    print("panels (left to right): input | predicted mask | ground-truth mask | overlay (red=pred, green=gt)")
    if args.crop_mode == "crop":
        print("NOTE: crop_mode='crop' predictions are shown resized back to native size, not "
              "re-placed at their true crop offset. Use crop_mode='resize' for a geometrically "
              "correct comparison.")


if __name__ == "__main__":
    main()