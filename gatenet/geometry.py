"""
geometry.py - gate-corner geometry helpers for adaptive cropping.

Reads the same corners.csv format generate_masks.py uses (image_name, x1,y1,x2,y2,x3,y3,x4,y4
per gate, one row per gate, multiple rows per image for multi-gate images), and picks a crop
window around the two closest visible gates, per the paper's rule.

NOTE on "closest": we use each gate's apparent size in the image (its bounding-circle radius)
as a proxy for physical distance - a gate that looks bigger is assumed to be closer. This is
an approximation, not a true distance measurement (that would need camera calibration / known
gate size), but it's a reasonable stand-in given what's in the labeled data.

NOTE on ground truth: this uses the CSV's labeled corners directly, i.e. an "oracle" version of
adaptive cropping. The real deployed system doesn't know gate corners in advance; it uses a
state estimate instead. See our discussion: use this to find the ceiling on how much adaptive
cropping could help, not as a deployment-realistic result.
"""

import csv
import os


def parse_csv_gate(row_values, n_vars_per_gate=8):
    """Same parsing as generate_masks.py: 8 numbers -> 4 (x, y) corner tuples."""
    if len(row_values) != n_vars_per_gate:
        return []
    gate = []
    for i in range(0, n_vars_per_gate, 2):
        x = int(round(float(row_values[i])))
        y = int(round(float(row_values[i + 1])))
        gate.append((x, y))
    return gate


def read_gates_csv(csv_path):
    """Same grouping as generate_masks.py's read_csv: {image_name: [gate, gate, ...]}."""
    gates_by_image = {}
    if not os.path.exists(csv_path):
        return gates_by_image
    with open(csv_path) as f:
        for row in csv.reader(f, delimiter=","):
            if not row:
                continue
            gate = parse_csv_gate(row[1:])
            gates_by_image.setdefault(row[0], [])
            if gate:
                gates_by_image[row[0]].append(gate)
    return gates_by_image


def gate_center_and_size(gate):
    """Center point and average corner-to-center distance (apparent size / closeness proxy)."""
    cx = sum(c[0] for c in gate) / 4.0
    cy = sum(c[1] for c in gate) / 4.0
    size = sum(((c[0] - cx) ** 2 + (c[1] - cy) ** 2) ** 0.5 for c in gate) / 4.0
    return (cx, cy), size


def two_closest_gates(gates):
    """Return up to 2 gates, largest-apparent-size first (our "closest" proxy)."""
    if not gates:
        return []
    scored = [(gate_center_and_size(g), g) for g in gates]
    scored.sort(key=lambda item: item[0][1], reverse=True)
    return [g for (_center_size, g) in scored[:2]]


def adaptive_crop_box(gates, img_w, img_h, out_size):
    """Compute an out_size x out_size crop box (x0, y0) containing the two closest visible
       gates, clamped so the box stays inside the image. Falls back to a center crop if no
       gates are visible in this image.
    """
    chosen = two_closest_gates(gates)
    if not chosen:
        # No visible gates: fall back to a center crop (same as crop_mode="crop" validation case)
        x0 = max((img_w - out_size) / 2, 0)
        y0 = max((img_h - out_size) / 2, 0)
        return int(x0), int(y0)

    centers = [gate_center_and_size(g)[0] for g in chosen]
    cx = sum(c[0] for c in centers) / len(centers)   # midpoint between the (up to 2) gate centers
    cy = sum(c[1] for c in centers) / len(centers)

    x0 = cx - out_size / 2
    y0 = cy - out_size / 2
    # Clamp so the crop window stays fully inside the image
    x0 = min(max(x0, 0), max(img_w - out_size, 0))
    y0 = min(max(y0, 0), max(img_h - out_size, 0))
    return int(x0), int(y0)