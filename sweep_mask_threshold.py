"""
Compare foreground masks under different PC1 thresholds / dilation settings.
Rows = images, columns = (threshold, kernel) settings, plus an Otsu (adaptive) column.

  python sweep_mask_threshold.py --data_path data/mvtec_anomaly_detection --object screw \
      --thresholds 10 15 20 25 --n 4
Output: mask_vis/sweep_<object>.png
"""
import argparse, glob, os
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.backbones import get_model
from visualize_masking import masking_steps

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def otsu_mask(tokens, grid, kernel_size=3, border=0.2):
    """Adaptive threshold: Otsu on PC1 (oriented with the same center rule), then same morphology."""
    pc1 = PCA(n_components=1, svd_solver="randomized").fit_transform(tokens.astype(np.float32)).squeeze().reshape(grid)
    u8 = cv2.normalize(pc1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = u8 > thr
    H, W = grid
    c = m[int(H * border):int(H * (1 - border)), int(W * border):int(W * (1 - border))]
    if c.mean() <= 0.35:
        m = ~m
    k = np.ones((kernel_size, kernel_size), np.uint8)
    if kernel_size > 1:
        m = cv2.dilate(m.astype(np.uint8), k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def overlay(ax, img, m, title):
    up = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ax.imshow(img)
    ax.imshow(np.ma.masked_where(up > 0, np.ones(up.shape)), cmap=matplotlib.colors.ListedColormap(["red"]), alpha=0.45)
    ax.set_title(title, fontsize=9); ax.axis("off")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--object", default="screw")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--thresholds", type=float, nargs="+", default=[10, 15, 20, 25])
    p.add_argument("--kernels", type=int, nargs="+", default=[3, 1], help="3 = original dilate+close, 1 = no morphology")
    p.add_argument("--model_name", default="dinov2_vits14")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="mask_vis")
    a = p.parse_args()

    paths = sorted(glob.glob(f"{a.data_path}/{a.object}/train/good/*.png"))[:a.n // 2 or 1]
    paths += [sorted(glob.glob(f"{d}/*.png"))[0] for d in sorted(glob.glob(f"{a.data_path}/{a.object}/test/*"))
              if os.path.basename(d) != "good"][:a.n - len(paths)]

    model = get_model(a.model_name, a.device, smaller_edge_size=a.resolution)
    settings = [(t, k) for k in a.kernels for t in a.thresholds]
    fig, axes = plt.subplots(len(paths), len(settings) + 2, figsize=(3 * (len(settings) + 2), 3.2 * len(paths)), squeeze=False)

    print(f"{'image':38s} " + " ".join(f"t={t:g},k={k}".rjust(10) for t, k in settings) + "      otsu")
    for r, path in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        t, grid = model.prepare_image(img)
        tok = model.extract_features(t)
        img_r = np.clip(t.permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)
        axes[r, 0].imshow(img_r); axes[r, 0].axis("off"); axes[r, 0].set_title(path.split("/")[-2] + "/" + path.split("/")[-1], fontsize=9)
        row = []
        for c, (thr, k) in enumerate(settings, start=1):
            s = masking_steps(tok, grid, threshold=thr, kernel_size=k)
            m = s["final"] if k > 1 else s["oriented"]
            row.append(m.mean())
            overlay(axes[r, c], img_r, m, f"thr={thr:g} {'dilate+close' if k > 1 else 'no morph'}\nfg={m.mean():.0%}{'  FLIP' if s['flipped'] else ''}")
        mo = otsu_mask(tok, grid)
        overlay(axes[r, -1], img_r, mo, f"Otsu + dilate+close\nfg={mo.mean():.0%}")
        print(f"{path[-38:]:38s} " + " ".join(f"{x:10.1%}" for x in row) + f"{mo.mean():10.1%}")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"sweep_{a.object}.png")
    fig.tight_layout(); fig.savefig(out, dpi=90); print("saved", out)
