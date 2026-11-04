"""
Step-by-step visualization of AnomalyDINO's zero-shot foreground masking.

Mirrors DINOv2Wrapper.compute_background_mask (src/backbones.py):
  1) DINOv2 patch tokens -> PCA(n_components=1) fit on THIS image's patches
  2) raw mask = PC1 > threshold (default 10)
  3) adaptive flip: if the center crop (border=0.2) keeps <= 35% patches,
     use  -PC1 > threshold  instead (PCA sign is arbitrary)
  4) dilate (3x3) + morphological close (3x3)

Usage:
  # single / few images
  python visualize_masking.py --images path/a.png path/b.png
  # MVTec-style folder: N images from <data_path>/<object>/{train/good, test/*}
  python visualize_masking.py --data_path data/mvtec_anomaly_detection --objects hazelnut screw pill --n 3
Outputs go to --out_dir (default: mask_vis/).
"""
import argparse, glob, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.backbones import get_model


def masking_steps(tokens, grid_size, threshold=10, kernel_size=3, border=0.2):
    pc1 = PCA(n_components=1, svd_solver="randomized").fit_transform(tokens.astype(np.float32)).squeeze()
    H, W = grid_size
    pc1_map = pc1.reshape(H, W)

    raw = pc1_map > threshold
    c = raw[int(H * border):int(H * (1 - border)), int(W * border):int(W * (1 - border))]
    center_ratio = c.sum() / c.size
    flipped = center_ratio <= 0.35
    oriented = (-pc1_map > threshold) if flipped else raw

    k = np.ones((kernel_size, kernel_size), np.uint8)
    final = cv2.dilate(oriented.astype(np.uint8), k)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, k).astype(bool)
    return dict(pc1=pc1_map, raw=raw, oriented=oriented, final=final,
                center_ratio=center_ratio, flipped=flipped)


def pca_rgb(tokens, grid_size):
    r = PCA(n_components=3, svd_solver="randomized").fit_transform(tokens.astype(np.float32))
    r = (r - r.min(0)) / (r.max(0) - r.min(0) + 1e-8)
    return r.reshape(*grid_size, 3)


def up(m, shape):
    return cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def visualize(model, img_path, out_path, threshold=10, border=0.2):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t, grid = model.prepare_image(img)
    tokens = model.extract_features(t)
    s = masking_steps(tokens, grid, threshold, border=border)
    # image exactly as the model sees it (resized + cropped), de-normalized
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_r = np.clip(t.permute(1, 2, 0).cpu().numpy() * std + mean, 0, 1)

    pc = -s["pc1"] if s["flipped"] else s["pc1"]
    v = np.abs(pc).max()
    H, W = grid
    y0, y1, x0, x1 = int(H * border), int(H * (1 - border)), int(W * border), int(W * (1 - border))

    fig, ax = plt.subplots(1, 6, figsize=(26, 4.8))
    ax[0].imshow(img_r); ax[0].set_title(os.path.relpath(img_path)[-45:], fontsize=9)
    ax[1].imshow(pca_rgb(tokens, grid)); ax[1].set_title("PCA top-3 (RGB)")
    im = ax[2].imshow(pc, cmap="RdBu_r", vmin=-v, vmax=v)
    ax[2].contour(pc, levels=[threshold], colors="k", linewidths=1)
    ax[2].add_patch(plt.Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5, ls="--"))
    ax[2].set_title(f"PC1{' (sign flipped)' if s['flipped'] else ''}  — contour = {threshold}")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].imshow(s["raw"], cmap="gray")
    ax[3].add_patch(plt.Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5, ls="--"))
    ax[3].set_title(f"raw PC1>{threshold}\ncenter keep={s['center_ratio']:.2f} → {'FLIP' if s['flipped'] else 'keep'}")
    ax[4].imshow(s["final"], cmap="gray")
    ax[4].set_title(f"final (dilate+close)\nfg {s['final'].sum()}/{s['final'].size} patches")
    ax[5].imshow(img_r)
    ax[5].imshow(np.ma.masked_where(up(s["final"], img_r.shape) > 0, np.ones(img_r.shape[:2])),
                 cmap=matplotlib.colors.ListedColormap(["red"]), alpha=0.45)
    ax[5].set_title("overlay (red = discarded background)")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return s


def collect(data_path, obj, n):
    paths = sorted(glob.glob(f"{data_path}/{obj}/train/good/*.png"))[:n]
    for d in sorted(glob.glob(f"{data_path}/{obj}/test/*")):
        paths += sorted(glob.glob(f"{d}/*.png"))[:max(1, n // 2)]
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="dinov2_vits14")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--device", default="cuda")
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--data_path", default=None)
    p.add_argument("--objects", nargs="*", default=["hazelnut"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--threshold", type=float, default=10)
    p.add_argument("--out_dir", default="mask_vis")
    a = p.parse_args()

    model = get_model(a.model_name, a.device, smaller_edge_size=a.resolution)
    jobs = [("custom", x) for x in a.images]
    if a.data_path:
        for o in a.objects:
            jobs += [(o, x) for x in collect(a.data_path, o, a.n)]

    for obj, path in jobs:
        d = os.path.join(a.out_dir, obj); os.makedirs(d, exist_ok=True)
        name = "_".join(path.replace("\\", "/").split("/")[-3:]).replace(".png", "") + ".png"
        s = visualize(model, path, os.path.join(d, name), a.threshold)
        print(f"{obj:12s} {path[-40:]:40s} flip={s['flipped']!s:5s} center={s['center_ratio']:.2f} "
              f"fg={s['final'].mean():.2%}")
