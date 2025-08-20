import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

# --------------------------
# 1. Perturbation functions
# --------------------------
def perturb_gaussian(img, sigma=0.1):
    noise = np.random.normal(0, sigma*255, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy

def perturb_brightness(img, factor=1.2):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def perturb_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def perturb_occlusion(img, size=50):
    h, w, _ = img.shape
    x, y = np.random.randint(0, w-size), np.random.randint(0, h-size)
    occluded = img.copy()
    occluded[y:y+size, x:x+size, :] = 0
    return occluded

def perturb_rotation(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

perturbations = {
    "gaussian": perturb_gaussian,
    "brightness": perturb_brightness,
    "blur": perturb_blur,
    "occlusion": perturb_occlusion,
    "rotation": perturb_rotation
}

# --------------------------
# 2. Similarity metrics
# --------------------------
def compute_metrics(expl1, expl2, topk=0.1):
    """Compare two heatmaps (numpy arrays in [0,1])."""
    expl1 = (expl1 - expl1.min()) / (expl1.max() - expl1.min() + 1e-8)
    expl2 = (expl2 - expl2.min()) / (expl2.max() - expl2.min() + 1e-8)

    # Flatten
    f1, f2 = expl1.flatten(), expl2.flatten()

    # Pearson correlation
    corr = np.corrcoef(f1, f2)[0,1]

    # IoU of top-k%
    k = int(len(f1) * topk)
    idx1, idx2 = np.argsort(f1)[-k:], np.argsort(f2)[-k:]
    intersection = len(set(idx1) & set(idx2))
    union = len(set(idx1) | set(idx2))
    iou = intersection / union if union > 0 else 0

    # SSIM (structural similarity)
    ssim_score = ssim(expl1, expl2, data_range=1.0)

    return corr, iou, ssim_score

# --------------------------
# 3. Perturbation engine
# --------------------------
def run_perturbation_engine(model, explain_fn, preds_csv, img_dir, save_csv, class_idx=12):
    """
    model: trained model
    explain_fn: function(img_tensor) -> heatmap (H,W)
    preds_csv: predictions CSV with labels + probs
    """
    preds = pd.read_csv(preds_csv)

    results = []
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    for _, row in tqdm(preds.iterrows(), total=len(preds)):
        fname = row["Image Index"]
        img_path = os.path.join(img_dir, fname)
        orig_img = Image.open(img_path).convert("RGB").resize((224,224))
        orig_np = np.array(orig_img)
        orig_tensor = transform(orig_img).unsqueeze(0).to(next(model.parameters()).device)

        # get original prediction
        with torch.no_grad():
            logits = model(orig_tensor)
            prob = torch.sigmoid(logits)[0, class_idx].item()
        orig_class = int(prob >= 0.5)

        # get original explanation
        expl_orig = explain_fn(orig_tensor)

        for p_name, perturb_fn in perturbations.items():
            pert_np = perturb_fn(orig_np)

            # convert back to tensor
            pert_tensor = transform(Image.fromarray(pert_np)).unsqueeze(0).to(next(model.parameters()).device)

            # check prediction unchanged
            with torch.no_grad():
                pert_prob = torch.sigmoid(model(pert_tensor))[0, class_idx].item()
            pert_class = int(pert_prob >= 0.5)

            if pert_class != orig_class:
                continue  # skip if prediction flips

            # get explanation
            expl_pert = explain_fn(pert_tensor)

            # compute metrics
            corr, iou, ssim_score = compute_metrics(expl_orig, expl_pert)

            results.append({
                "Image": fname,
                "Perturbation": p_name,
                "Method": explain_fn.__name__,
                "Correlation": corr,
                "IoU": iou,
                "SSIM": ssim_score
            })

    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print(f"✅ Saved perturbation results to {save_csv}")
    return df
