"""
dataset.py - image/mask dataset for GateNet (MonoRace, arXiv:2601.15222).

Reads img_* / mask_* pairs from the MonoRaceGate `data/` folder (the masks are
created once by generate_masks.py, which `make` already ran). Nothing here calls
generate_masks.py.

Paper references (GateNet section):
  - "Training setup": 384 x 384 input.
  - "Data augmentations": random affine, HSV jitter, motion blur (5-15 px kernel,
    random orientation), thermal/Gaussian noise, kernel-based blurring.
  - Fig. 6B caption: HSV colour, directional brightness gradient, Gaussian blur,
    additive Gaussian noise, lens distortion, rolling-shutter (motion) kernel blur.

NOTE: the paper does NOT give exact ranges or probabilities for the augmentations.
The values in AUG below are my own reasonable defaults - tune them / confirm
with your lab.

All randomness uses Python's `random` module because PyTorch seeds it separately
in every DataLoader worker (so workers don't produce identical augmentations).
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------
# Augmentation settings (my choices - the paper only names the augmentation types)
# ----------------------------------------------------------------------------
AUG = dict(
    # geometric (applied to image AND mask)
    rot_deg=15.0,                 # random rotation in [-rot, +rot] degrees
    scale_range=(0.8, 1.25),      # random zoom
    trans_frac=0.10,              # random translation, fraction of output size
    lens_p=0.3,                   # probability of lens distortion
    lens_k_range=(-0.25, 0.25),   # radial distortion strength (barrel/pincushion)
    # photometric (applied to image only)
    hue_shift_deg=15.0,           # +/- hue shift in degrees (OpenCV float hue is 0-360)
    sat_range=(0.6, 1.4),         # saturation multiplier
    val_range=(0.6, 1.4),         # value (brightness) multiplier
    gradient_p=0.5,               # probability of directional brightness gradient
    gradient_strength=(0.1, 0.4),
    motion_blur_p=0.5,            # probability of motion blur
    motion_blur_ksize=(5, 15),    # kernel size in pixels (from the paper)
    gauss_blur_p=0.3,             # probability of lens-style Gaussian blur
    gauss_blur_ksizes=(3, 5),
    noise_p=0.7,                  # probability of additive Gaussian noise
    noise_var=(0.0, 144.0),       # variance in (0-255 pixel units)^2, sampled per image
)


# ----------------------------------------------------------------------------
# Finding image/mask pairs
# ----------------------------------------------------------------------------
def collect_samples(data_root, folders):
    """Return a list of (img_path, mask_path) for the given dataset folders.

    Image names are read from each folder's corners.csv (first column), exactly
    like generate_masks.py does. The mask is the same path with 'img_' replaced
    by 'mask_' in the file name. Images without a mask on disk are skipped.
    """
    samples, missing = [], 0
    for ds in folders:
        csv_path = os.path.join(data_root, ds, "corners.csv")
        if not os.path.exists(csv_path):
            print(f"[dataset] WARNING: {csv_path} not found, skipping folder")
            continue

        names = []
        with open(csv_path) as f:
            for row in csv.reader(f, delimiter=","):
                if row:
                    names.append(row[0])

        for name in dict.fromkeys(names):  # unique, keeps order
            img_path = os.path.join(data_root, ds, name.lstrip("/"))
            folder, base = os.path.split(img_path)
            mask_path = os.path.join(folder, base.replace("img_", "mask_", 1))
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                samples.append((img_path, mask_path))
            else:
                missing += 1

    print(f"[dataset] {len(samples)} pairs found in {list(folders)} ({missing} skipped: missing image or mask)")
    return samples


# ----------------------------------------------------------------------------
# Geometric augmentations (must be applied identically to image and mask)
# ----------------------------------------------------------------------------
def _to_3x3(m2x3):
    m = np.eye(3, dtype=np.float64)
    m[:2] = m2x3
    return m


def geometric_transform(img, mask, out_size, crop_mode, train, cfg=AUG):
    """Map the source image to an out_size x out_size training window.

    crop_mode:
      "resize" - whole image squashed to out_size x out_size (paper's 'Resized' option,
                 keeps the full field of view, changes the aspect ratio).
      "crop"   - native-resolution out_size x out_size window (random position when
                 training, centred when validating). Stand-in for the paper's adaptive
                 cropping, which needs the state estimate and can't be used in training.
    """
    h, w = img.shape[:2]
    c = out_size / 2.0

    # Base transform B: source image -> output window
    if crop_mode == "resize":
        B = np.array([[out_size / w, 0, 0], [0, out_size / h, 0], [0, 0, 1]], dtype=np.float64)
    else:
        if train:
            x0 = random.uniform(0, max(w - out_size, 0))
            y0 = random.uniform(0, max(h - out_size, 0))
        else:
            x0, y0 = max((w - out_size) / 2, 0), max((h - out_size) / 2, 0)
        B = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)

    M = B
    if train:
        # Random affine transformations (rotation, translation, scaling)
        angle = random.uniform(-cfg["rot_deg"], cfg["rot_deg"])
        scale = random.uniform(*cfg["scale_range"])
        R = _to_3x3(cv2.getRotationMatrix2D((c, c), angle, scale))  # rotate + scale about window centre
        tx = random.uniform(-cfg["trans_frac"], cfg["trans_frac"]) * out_size
        ty = random.uniform(-cfg["trans_frac"], cfg["trans_frac"]) * out_size
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        M = T @ R @ B

    # Zero padding outside the image (reflection would create unlabeled ghost gates)
    img_o = cv2.warpAffine(img, M[:2], (out_size, out_size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_o = cv2.warpAffine(mask, M[:2], (out_size, out_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_o, mask_o


def random_lens_distortion(img, mask, k_range):
    """Simulated radial (barrel / pincushion) lens distortion, applied to image AND mask."""
    h, w = img.shape[:2]
    k = random.uniform(*k_range)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)  # normalised coords in [-1, 1]
    f = 1.0 + k * (xn ** 2 + yn ** 2)
    map_x, map_y = xn * f * (w / 2) + w / 2, yn * f * (h / 2) + h / 2
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img, mask


# ----------------------------------------------------------------------------
# Photometric augmentations (image only - the mask stays as the clean label)
# ----------------------------------------------------------------------------
def hsv_jitter(img01, cfg=AUG):
    """HSV-based colour augmentation: random shifts in hue, saturation and value."""
    hsv = cv2.cvtColor(img01, cv2.COLOR_BGR2HSV)  # float input: H in [0,360), S,V in [0,1]
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-cfg["hue_shift_deg"], cfg["hue_shift_deg"])) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*cfg["sat_range"]), 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*cfg["val_range"]), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def brightness_gradient(img255, cfg=AUG):
    """Directional brightness gradient (Fig. 6B-c): linear light ramp at a random angle."""
    h, w = img255.shape[:2]
    theta = random.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)
    ramp = xn * np.cos(theta) + yn * np.sin(theta)
    gain = 1.0 + random.uniform(*cfg["gradient_strength"]) * ramp
    return img255 * gain[..., None]


def motion_blur(img, ksize_range):
    """Artificial motion blur: averaging kernel with random size (5-15 px) and orientation.

    The paper says 'square averaging kernel of random size and random orientation'.
    I implement it as a line-shaped averaging kernel rotated to a random angle
    (the usual way to simulate directional blur) - confirm this interpretation.
    """
    k = random.randint(*ksize_range)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0                                   # horizontal line
    rot = cv2.getRotationMatrix2D(((k - 1) / 2.0, (k - 1) / 2.0), random.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))              # random orientation
    s = kernel.sum()
    if s < 1e-6:
        return img
    return cv2.filter2D(img, -1, kernel / s, borderType=cv2.BORDER_REFLECT)


def photometric_augment(img_u8, cfg=AUG):
    """Apply the image-only augmentations in a plausible 'camera' order; returns float32 [0,1]."""
    img01 = img_u8.astype(np.float32) / 255.0

    # HSV-based colour augmentation (hue / saturation / value shifts)
    img01 = hsv_jitter(img01, cfg)
    img = img01 * 255.0

    # Directional brightness gradient
    if random.random() < cfg["gradient_p"]:
        img = brightness_gradient(img, cfg)

    # Artificial motion blur (5-15 px kernel, random orientation)
    if random.random() < cfg["motion_blur_p"]:
        img = motion_blur(img, cfg["motion_blur_ksize"])

    # Kernel-based (Gaussian) blurring to mimic lens-induced degradation
    if random.random() < cfg["gauss_blur_p"]:
        k = random.choice(cfg["gauss_blur_ksizes"])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Additive Gaussian noise with randomly sampled variance (thermal / sensor noise)
    if random.random() < cfg["noise_p"]:
        rng = np.random.default_rng(random.getrandbits(32))
        variance = random.uniform(*cfg["noise_var"])
        sigma = np.sqrt(variance)   
        noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
        img = img + noise

    return np.clip(img, 0, 255).astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class GateDataset(Dataset):
    """Yields (image, mask):
         image: float32 tensor (3, S, S), RGB, values in [0, 1]
         mask:  float32 tensor (1, S, S), values in {0, 1}
       Multi-scale (downscaled) masks for GateNet's auxiliary losses are NOT made here;
       do that in the loss (e.g. F.interpolate / avg-pool the mask to each output size).

       `samples` is a list of (img_path, mask_path). To mix in synthetic data later
       (paper: 3500 synthetic : 500 real), build a second GateDataset from the synthetic
       samples and combine with torch.utils.data.ConcatDataset - both then go through
       this same pipeline, matching the paper's 'unified dataloader'.
    """

    def __init__(self, samples, out_size=384, train=True, crop_mode="resize", cfg=AUG):
        assert crop_mode in ("resize", "crop")
        self.samples, self.out_size = samples, out_size
        self.train, self.crop_mode, self.cfg = train, crop_mode, cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)                # BGR uint8
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)          # uint8 0/255
        if img is None or mask is None:
            raise FileNotFoundError(f"Could not read {img_path} or {mask_path}")
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Geometric part: affine (train only) + crop/resize to 384 x 384, same for image and mask
        img, mask = geometric_transform(img, mask, self.out_size, self.crop_mode, self.train, self.cfg)

        if self.train:
            # Lens distortion (geometric, so image and mask together)
            if random.random() < self.cfg["lens_p"]:
                img, mask = random_lens_distortion(img, mask, self.cfg["lens_k_range"])
            # Photometric augmentations, image only
            img = photometric_augment(img, self.cfg)
        else:
            img = img.astype(np.float32) / 255.0

        img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))   # BGR->RGB, HWC->CHW
        mask = (mask > 127).astype(np.float32)[None]                    # binarise, add channel dim
        return torch.from_numpy(img), torch.from_numpy(mask)


# ----------------------------------------------------------------------------
# Convenience: train/val split BY FOLDER (frames in one flight are near-duplicates,
# so a random frame-level split would leak and inflate the IoU)
# ----------------------------------------------------------------------------
ALL_FOLDERS = ["adnec_s1_f1", "adnec_s1_f2", "dhl_s1_f1", "marina_s1_f1",
               "red_s1_f1", "red_s1_f2", "red_s1_f3"]


def build_datasets(data_root="./data", val_folders=("dhl_s1_f1", "marina_s1_f1"), out_size=384, crop_mode="resize"):
    train_folders = [f for f in ALL_FOLDERS if f not in val_folders]
    train_ds = GateDataset(collect_samples(data_root, train_folders), out_size, train=True, crop_mode=crop_mode)
    val_ds = GateDataset(collect_samples(data_root, list(val_folders)), out_size, train=False, crop_mode=crop_mode)
    return train_ds, val_ds


# ----------------------------------------------------------------------------
# Sanity check: python gatenet/dataset.py
# Saves aug_check.png: rows of [augmented image | mask | overlay]. Verify that the
# red overlay still sits exactly on the gate borders after augmentation.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--val_folders", nargs="+", default=["dhl_s1_f1", "marina_s1_f1"])
    ap.add_argument("--crop_mode", default="resize", choices=["resize", "crop"])
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    train_ds, val_ds = build_datasets(args.data_root, args.val_folders, crop_mode=args.crop_mode)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    rows = []
    for _ in range(args.n):
        idx = random.randrange(len(train_ds))
        img_path, mask_path = train_ds.samples[idx]

        # Original: same deterministic resize/crop as validation, no augmentation
        orig = cv2.imread(img_path)
        orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orig, _ = geometric_transform(orig, orig_mask, 384, args.crop_mode, train=False)

        # Augmented version (random each call)
        img, mask = train_ds[idx]
        img = (img.numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)   # back to BGR uint8
        m = (mask[0].numpy() * 255).astype(np.uint8)
        overlay = img.copy()
        overlay[..., 2] = np.maximum(overlay[..., 2], m)

        rows.append(np.hstack([orig, img, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), overlay]))
    cv2.imwrite("gatenet/aug_check.png", np.vstack(rows))
    print("wrote aug_check.png")