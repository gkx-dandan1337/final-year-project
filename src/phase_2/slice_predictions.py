# file: slice_predictions.py
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "src/phase2/outputs/results.csv"   # path to your saved predictions
OUTPUT_DIR = "src/phase2/outputs/slices"               # where to save subsets
SAMPLES_PER_GROUP = 30                      # how many per class to keep

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load predictions
# ----------------------------
df = pd.read_csv(CSV_PATH)

# ----------------------------
# Identify TP / TN / FP / FN
# ----------------------------
tp = df[(df["true_label"] == 1) & (df["predicted_label"] == 1)]
tn = df[(df["true_label"] == 0) & (df["predicted_label"] == 0)]
fp = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)]
fn = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)]

print(f"Counts → TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}")

# ----------------------------
# Sample balanced subset
# ----------------------------
tp_sample = tp.sample(min(SAMPLES_PER_GROUP, len(tp)), random_state=42)
tn_sample = tn.sample(min(SAMPLES_PER_GROUP, len(tn)), random_state=42)
fp_sample = fp.sample(min(SAMPLES_PER_GROUP, len(fp)), random_state=42)
fn_sample = fn.sample(min(SAMPLES_PER_GROUP, len(fn)), random_state=42)

# ----------------------------
# Save each slice
# ----------------------------
tp_sample.to_csv(os.path.join(OUTPUT_DIR, "tp.csv"), index=False)
tn_sample.to_csv(os.path.join(OUTPUT_DIR, "tn.csv"), index=False)
fp_sample.to_csv(os.path.join(OUTPUT_DIR, "fp.csv"), index=False)
fn_sample.to_csv(os.path.join(OUTPUT_DIR, "fn.csv"), index=False)

# Also save one merged subset for convenience
subset = pd.concat([tp_sample, tn_sample, fp_sample, fn_sample])
subset.to_csv(os.path.join(OUTPUT_DIR, "subset_balanced.csv"), index=False)

print("✅ Slices saved in:", OUTPUT_DIR)
