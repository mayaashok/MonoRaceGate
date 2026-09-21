"""
model.py - GateNet (MonoRace, arXiv:2601.15222), a U-Net-style gate segmentation network.

Paper description (GateNet section):
  - Encoder-decoder with skip connections from each encoder level to the matching decoder level.
  - Channel widths are divided by a scaling factor f (paper uses f = 4).
  - Every block = double 3x3 convolution, each followed by batch norm + ReLU.
  - inc      : initial double-conv block
  - down_k   : max-pool, then a double-conv block with k output channels
  - up_k     : transposed conv + batch norm, ADD the encoder skip, then a double-conv block with k channels
  - outc     : 1x1 conv to a single channel + sigmoid
  - Five output maps y0..y4 at progressively increasing resolution (deep supervision).
    Only the highest-resolution map (y4) is used at deployment.
  - Xavier uniform weight initialisation.

Layout with f = 4 and a 384 x 384 input (channels @ resolution):

  Encoder                           Decoder                        Outputs
  inc   16  @ 384  ----skip--------> up4  16  @ 384  ------------> y4 (384 x 384)
  down1 32  @ 192  ----skip--------> up3  16  @ 192  ------------> y3 (192 x 192)
  down2 64  @  96  ----skip--------> up2  32  @  96  ------------> y2 ( 96 x  96)
  down3 128 @  48  ----skip--------> up1  64  @  48  ------------> y1 ( 48 x  48)
  down4 128 @  24  (bottleneck) ---------------------------------> y0 ( 24 x  24)

Choices the paper does NOT specify (my assumptions - confirm with your lab):
  - Max-pool is 2x2 with stride 2; transposed conv has kernel 2 and stride 2 (standard U-Net).
  - 3x3 convs use padding=1, so they keep the spatial size.
  - Convs followed by batch norm have no bias (batch norm makes it redundant).
  - No ReLU after the transposed conv's batch norm (the paper only lists BN there).
  - The input height and width must be divisible by 16 (four max-pools).
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two (3x3 conv -> batch norm -> ReLU) layers in a row."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # First 3x3 convolutional layer
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # Batch normalization
            nn.BatchNorm2d(out_ch),
            # ReLU activation
            nn.ReLU(inplace=True),
            # Second 3x3 convolutional layer
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # Batch normalization
            nn.BatchNorm2d(out_ch),
            # ReLU activation
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """down_k: max-pool (halve the resolution), then a double-conv block with k output channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 2x2 max pooling (halves height and width)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Double 3x3 convolution block
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """up_k: transposed conv + batch norm, add the encoder skip, then a double-conv block.

    The skip is ADDED (not concatenated), so the transposed conv must output
    exactly `skip_ch` channels for the shapes to match.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Transposed convolution (doubles height and width, maps in_ch -> skip_ch channels)
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2, bias=False)
        # Batch normalization
        self.bn = nn.BatchNorm2d(skip_ch)
        # Double 3x3 convolution block (after the skip has been added)
        self.conv = DoubleConv(skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.bn(self.up(x))       # upsample + batch norm
        x = x + skip                  # skip connection from the encoder (element-wise addition)
        return self.conv(x)


# ----------------------------------------------------------------------------
# GateNet
# ----------------------------------------------------------------------------
class GateNet(nn.Module):
    """U-Net-style gate segmentation network with five multi-scale outputs.

    forward() returns a list [y0, y1, y2, y3, y4] of sigmoid probability maps,
    each of shape (B, 1, H_i, W_i), ordered from LOWEST resolution (y0, H/16)
    to HIGHEST (y4, H). At deployment use only y4.
    """

    def __init__(self, f=4, in_channels=3):
        super().__init__()
        # Channel widths divided by the scaling factor f (f=4 -> 16, 32, 64, 128, 128)
        c = [64 // f, 128 // f, 256 // f, 512 // f, 512 // f]

        # ---------------- Encoder ----------------
        self.inc = DoubleConv(in_channels, c[0])    # inc-64/f
        self.down1 = Down(c[0], c[1])               # down1-128/f
        self.down2 = Down(c[1], c[2])               # down2-256/f
        self.down3 = Down(c[2], c[3])               # down3-512/f
        self.down4 = Down(c[3], c[4])               # down4-512/f (bottleneck)

        # ---------------- Decoder ----------------
        # Up(in_ch, skip_ch, out_ch): skip_ch is the channel count of the encoder skip it adds
        self.up1 = Up(c[4], c[3], c[2])             # up1-256/f, skip from down3
        self.up2 = Up(c[2], c[2], c[1])             # up2-128/f, skip from down2
        self.up3 = Up(c[1], c[1], c[0])             # up3-64/f,  skip from down1
        self.up4 = Up(c[0], c[0], c[0])             # up4-64/f,  skip from inc

        # ---------------- Output heads ----------------
        # 1x1 convolutions mapping to a single channel (sigmoid is applied in forward)
        self.outc0 = nn.Conv2d(c[4], 1, kernel_size=1)      # from bottleneck
        self.outc1 = nn.Conv2d(c[2], 1, kernel_size=1)      # from up1
        self.outc2 = nn.Conv2d(c[1], 1, kernel_size=1)      # from up2
        self.outc3 = nn.Conv2d(c[0], 1, kernel_size=1)      # from up3
        self.outc4 = nn.Conv2d(c[0], 1, kernel_size=1)      # from up4

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all (transposed) conv weights, as in the paper."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            raise ValueError(f"Input height/width must be divisible by 16, got {tuple(x.shape[-2:])}")

        # ---- Encoder (features kept as skip connections BEFORE the next down block) ----
        x0 = self.inc(x)        # (B,  16, 384, 384)
        x1 = self.down1(x0)     # (B,  32, 192, 192)
        x2 = self.down2(x1)     # (B,  64,  96,  96)
        x3 = self.down3(x2)     # (B, 128,  48,  48)
        x4 = self.down4(x3)     # (B, 128,  24,  24)  bottleneck

        # ---- Decoder (each stage upsamples and adds the matching encoder features) ----
        u1 = self.up1(x4, x3)   # (B,  64,  48,  48)
        u2 = self.up2(u1, x2)   # (B,  32,  96,  96)
        u3 = self.up3(u2, x1)   # (B,  16, 192, 192)
        u4 = self.up4(u3, x0)   # (B,  16, 384, 384)

        # ---- Five sigmoid output maps, lowest to highest resolution ----
        y0 = torch.sigmoid(self.outc0(x4))   # (B, 1,  24,  24)
        y1 = torch.sigmoid(self.outc1(u1))   # (B, 1,  48,  48)
        y2 = torch.sigmoid(self.outc2(u2))   # (B, 1,  96,  96)
        y3 = torch.sigmoid(self.outc3(u3))   # (B, 1, 192, 192)
        y4 = torch.sigmoid(self.outc4(u4))   # (B, 1, 384, 384)
        return [y0, y1, y2, y3, y4]


# ----------------------------------------------------------------------------
# Sanity check: python gatenet/model.py
# Builds the network, pushes a random batch through it, and checks all output shapes.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    model = GateNet(f=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GateNet(f=4): {n_params:,} parameters")

    x = torch.randn(2, 3, 384, 384)      # fake batch of 2 RGB images
    outs = model(x)

    expected_sizes = [24, 48, 96, 192, 384]
    for i, (y, s) in enumerate(zip(outs, expected_sizes)):
        print(f"y{i}: {tuple(y.shape)}")
        assert y.shape == (2, 1, s, s), f"y{i} has wrong shape"
        assert 0.0 <= y.min().item() and y.max().item() <= 1.0, f"y{i} not in [0, 1]"
    print("All output shapes OK")