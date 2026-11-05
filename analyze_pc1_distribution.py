"""
Look at the PC1 distribution of every image, per category, to decide when a
fixed threshold vs. a dynamic (Otsu) threshold makes sense.

For each image:
  - PCA(1) on its own patch tokens (same as AnomalyDINO)
  - orientation decided WITHOUT a threshold: flip if mean(PC1 in center) < mean(PC1 on border)
  - raw-unit histogram of the oriented PC1, fixed thresholds and the image's Otsu threshold
  - statistics that describe the distribution shape (see STATS below)

  python analyze_pc1_distribution.py --data_path data/mvtec_anomaly_detection
  python analyze_pc1_distribution.py --data_path data/mvtec_anomaly_detection --objects screw cable hazelnut

Outputs (in --out_dir, default pc1_dist/):
  overview.png         one panel per category, all its images' PC1 histograms overlaid
  <object>.png         per-category detail: histograms + per-image spatial PC1 maps
  pc1_stats.csv        one row per image
  pc1_summary.csv      per-category median of the stats

STATS
  pc1_std            spread of PC1 (raw units) -> is "10" big or small for this image?
  otsu_thr           Otsu threshold in raw PC1 units
  otsu_fg            fraction of patches above Otsu (class balance)
  otsu_eta           Otsu separability = between-class var / total var (0..1, higher = cleaner 2 groups)
  bimodality_coef    BC = (skew^2+1)/(kurt + 3(n-1)^2/((n-2)(n-3))); > 0.555 suggests bi/multi-modal
  n_peaks            number of KDE peaks (>= 3 = multi-level, e.g. cable: wires / sheath / background)
  fg@T               fraction of patches with oriented PC1 > T for each fixed T
  oldrule_flip@T     would the original center-crop rule flip at threshold T?
  orient_agree@T     does the original rule agree with the mean-based orientation at T?
"""
import argparse, csv, glob, os
import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew
from sklearn.decomposition import PCA

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
         "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def otsu_float(x, bins=256):
    """Otsu on raw float values. Returns (threshold, eta)."""
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m = np.cumsum(p * centers); mT = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mT * w0 - m) ** 2 / (w0 * w1)
    sb = np.nan_to_num(sb)
    best = np.flatnonzero(sb >= sb.max() * (1 - 1e-6))   # plateau -> take its middle
    i = int(best[len(best) // 2])
    return edges[i + 1], sb[i] / (x.var() + 1e-12)


def center_border(grid, border=0.2):
    H, W = grid
    c = np.zeros(grid, bool)
    c[int(H * border):int(H * (1 - border)), int(W * border):int(W * (1 - border))] = True
    return c


def analyze(tokens, grid, thresholds, border=0.2):
    pc1 = PCA(n_components=1, svd_solver="randomized").fit_transform(tokens.astype(np.float32)).squeeze().reshape(grid)
    c = center_border(grid, border)
    sign = 1.0 if pc1[c].mean() >= pc1[~c].mean() else -1.0
    x = sign * pc1                                  # oriented: object side = positive (by mean rule)
    v = x.ravel()

    thr, eta = otsu_float(v)
    n = v.size
    bc = (skew(v) ** 2 + 1) / (kurtosis(v) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    pad = 0.15 * (v.max() - v.min())                      # so edge modes become real peaks
    grid_x = np.linspace(v.min() - pad, v.max() + pad, 512)
    dens = gaussian_kde(v)(grid_x)
    peaks, _ = find_peaks(dens, prominence=0.02 * dens.max())

    st = dict(pc1_std=v.std(), otsu_thr=thr, otsu_fg=(v > thr).mean(), otsu_eta=eta,
              bimodality_coef=bc, n_peaks=len(peaks))
    for T in thresholds:
        st[f"fg@{T:g}"] = (v > T).mean()
        raw = pc1 > T                               # original rule on the UNoriented PC1
        old_flip = raw[c].mean() <= 0.35
        st[f"oldrule_flip@{T:g}"] = int(old_flip)
        old_sign = -1.0 if old_flip else 1.0
        st[f"orient_agree@{T:g}"] = int(old_sign == sign)
    return x, st, (grid_x, dens, peaks)


def collect(data_path, obj, n_train, n_per_type):
    paths = sorted(glob.glob(f"{data_path}/{obj}/train/good/*.png"))[:n_train]
    for d in sorted(glob.glob(f"{data_path}/{obj}/test/*")):
        paths += sorted(glob.glob(f"{d}/*.png"))[:n_per_type]
    if not paths:  # VisA-style layout fallback
        paths = sorted(glob.glob(f"{data_path}/{obj}/**/*.JPG", recursive=True))[:n_train + 4]
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset", default="MVTec", choices=["MVTec", "VisA"])
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--n_train", type=int, default=5)
    ap.add_argument("--n_per_type", type=int, default=1)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 15, 20, 25])
    ap.add_argument("--model_name", default="dinov2_vits14")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="pc1_dist")
    a = ap.parse_args()

    from src.backbones import get_model
    objects = a.objects or (MVTEC if a.dataset == "MVTec" else VISA)
    os.makedirs(a.out_dir, exist_ok=True)
    model = get_model(a.model_name, a.device, smaller_edge_size=a.resolution)
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    rows, per_obj = [], {}
    for obj in objects:
        paths = collect(a.data_path, obj, a.n_train, a.n_per_type)
        if not paths:
            print(f"[skip] {obj}: no images"); continue
        per_obj[obj] = []
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            t, grid = model.prepare_image(img)
            x, st, kde = analyze(model.extract_features(t), grid, a.thresholds)
            img_r = np.clip(t.permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)
            rel = "/".join(p.replace("\\", "/").split("/")[-3:])
            per_obj[obj].append((rel, x, st, kde, img_r))
            rows.append(dict(object=obj, image=rel, **{k: round(float(v), 4) for k, v in st.items()}))
        print(f"{obj:12s} n={len(paths):3d}  " + "  ".join(
            f"{k}={np.median([s[k] for _, _, s, _, _ in per_obj[obj]]):.2f}"
            for k in ["pc1_std", "otsu_thr", "otsu_fg", "otsu_eta", "bimodality_coef", "n_peaks"]))

        # ---- per-category detail figure: histogram panel + spatial maps of up to 6 images
        items = per_obj[obj]
        show = items[:6]
        fig = plt.figure(figsize=(4 + 2.6 * len(show), 6.4))
        gs = fig.add_gridspec(2, len(show) + 2)
        ax = fig.add_subplot(gs[:, :2])
        cmap = plt.get_cmap("tab10")
        for i, (rel, x, st, (gx, dens, pk), _) in enumerate(items):
            col = cmap(i % 10)
            ax.plot(gx, dens, color=col, lw=1, alpha=0.8, label=rel)
            ax.axvline(st["otsu_thr"], color=col, lw=0.8, ls=":")
        for T in a.thresholds:
            ax.axvline(T, color="k", lw=1, ls="--", alpha=0.5)
            ax.text(T, ax.get_ylim()[1] * 0.97, f"{T:g}", ha="center", fontsize=7)
        ax.set_title(f"{obj}: oriented PC1 density (dotted = per-image Otsu, dashed = fixed)", fontsize=9)
        ax.set_xlabel("PC1 (raw units, object side positive)")
        ax.legend(fontsize=6, loc="upper right")
        for j, (rel, x, st, _, img_r) in enumerate(show):
            a0 = fig.add_subplot(gs[0, j + 2]); a0.imshow(img_r); a0.axis("off")
            a0.set_title(rel, fontsize=7)
            a1 = fig.add_subplot(gs[1, j + 2]); vmax = np.abs(x).max()
            a1.imshow(x, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            a1.contour(x, levels=[st["otsu_thr"]], colors="lime", linewidths=1)
            a1.contour(x, levels=[a.thresholds[0]], colors="k", linewidths=0.8, linestyles="--")
            a1.axis("off")
            a1.set_title(f"otsu={st['otsu_thr']:.1f} η={st['otsu_eta']:.2f}\npeaks={st['n_peaks']} BC={st['bimodality_coef']:.2f}", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"{obj}.png"), dpi=100); plt.close(fig)

    # ---- overview: one panel per category
    objs = list(per_obj)
    nc = 5; nr = int(np.ceil(len(objs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for ax, obj in zip(axes.ravel(), objs):
        for rel, x, st, (gx, dens, pk), _ in per_obj[obj]:
            ax.plot(gx, dens / dens.max(), color="tab:blue", lw=0.8, alpha=0.5)
        for T in a.thresholds:
            ax.axvline(T, color="k", lw=0.8, ls="--", alpha=0.4)
        med = {k: np.median([s[k] for _, _, s, _, _ in per_obj[obj]]) for k in ["otsu_thr", "otsu_eta", "otsu_fg", "n_peaks"]}
        ax.axvline(med["otsu_thr"], color="tab:red", lw=1.5)
        ax.set_title(f"{obj}  η={med['otsu_eta']:.2f} fg_otsu={med['otsu_fg']:.0%} peaks={med['n_peaks']:.0f}", fontsize=8)
        ax.set_yticks([])
    for ax in axes.ravel()[len(objs):]:
        ax.axis("off")
    fig.suptitle("Oriented PC1 density per image (blue), fixed thresholds (dashed), median Otsu (red)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "overview.png"), dpi=100); plt.close(fig)

    # ---- CSVs
    with open(os.path.join(a.out_dir, "pc1_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = [k for k in rows[0] if k not in ("object", "image")]
    with open(os.path.join(a.out_dir, "pc1_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["object", "n"] + [f"median_{k}" for k in keys])
        for obj in objs:
            r = [x for x in rows if x["object"] == obj]
            w.writerow([obj, len(r)] + [round(float(np.median([x[k] for x in r])), 4) for k in keys])
    print("saved to", a.out_dir)
