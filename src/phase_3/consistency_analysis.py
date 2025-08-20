import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# -----------------------------
# Helpers
# -----------------------------
def load_and_normalize(img_path):
    """Load heatmap (Grad-CAM/LIME/IG) and normalize to [0,1]."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def top_k_mask(arr, k=0.2):
    """Binary mask of top-k percent salient pixels."""
    thresh = np.quantile(arr, 1-k)
    return (arr >= thresh).astype(np.uint8)

def compute_metrics(hm1, hm2):
    """Compute Pearson correlation, IoU, and SSIM between two maps."""
    # Flatten for correlation
    r, _ = pearsonr(hm1.flatten(), hm2.flatten())

    # IoU on top-20% salient pixels
    mask1 = top_k_mask(hm1, 0.2)
    mask2 = top_k_mask(hm2, 0.2)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / (union + 1e-8)

    # SSIM
    ssim_val = ssim(hm1, hm2, data_range=1.0)

    return r, iou, ssim_val

# -----------------------------
# Main Analysis
# -----------------------------
def analyze_consistency(base_dir="src/phase_2/outputs", save_csv="src/phase_3/outputs/metrics.csv"):
    methods = ["gradcam", "lime", "integrated_gradients"]
    subsets = ["tp", "fp", "fn", "tn"]

    results = []

    for subset in subsets:
        # Collect filenames available in all methods for this subset
        files = []
        for method in methods:
            folder = os.path.join(base_dir, method, subset)
            imgs = set(os.path.basename(f) for f in glob(os.path.join(folder, "*.jpg")))
            files.append(imgs)
        common_files = set.intersection(*files)

        for fname in common_files:
            hmaps = {}
            for method in methods:
                path = os.path.join(base_dir, method, subset, fname)
                hmaps[method] = load_and_normalize(path)

            # Pairwise comparisons
            for (m1, m2) in [("gradcam","lime"), ("gradcam","integrated_gradients"), ("lime","integrated_gradients")]:
                r, iou, ssim_val = compute_metrics(hmaps[m1], hmaps[m2])
                results.append({
                    "subset": subset,
                    "file": fname,
                    "pair": f"{m1}_vs_{m2}",
                    "pearson_r": r,
                    "IoU_top20": iou,
                    "SSIM": ssim_val
                })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved metrics to {save_csv}")
    print(df.groupby(["subset","pair"]).mean())

if __name__ == "__main__":
    analyze_consistency()
