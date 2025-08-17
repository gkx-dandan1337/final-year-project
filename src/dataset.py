import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------
# Paths
# -----------------------
LABELS_CSV = "data/labels_onehot.csv"   # <- your new one-hot file
OUTPUT_DIR = "data"

# -----------------------
# Step 1. Load CSV
# -----------------------
df = pd.read_csv(LABELS_CSV)

# -----------------------
# Step 2. Patient-wise split
# -----------------------
# Get unique patients
patients = df["Patient ID"].unique()

# Split patients into train/val/test
train_patients, test_patients = train_test_split(patients, test_size=0.20, random_state=42)
train_patients, val_patients  = train_test_split(train_patients, test_size=0.125, random_state=42)
# (0.125 of 0.80 = 0.10 → so final split is 70/10/20)

# Helper function
def subset_by_patients(df, patient_ids):
    return df[df["Patient ID"].isin(patient_ids)]

train_df = subset_by_patients(df, train_patients)
val_df   = subset_by_patients(df, val_patients)
test_df  = subset_by_patients(df, test_patients)

# -----------------------
# Step 3. Save splits
# -----------------------
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print(f"✅ train.csv, val.csv, test.csv created in {OUTPUT_DIR}")
print(f"Train: {len(train_df)} images, Val: {len(val_df)} images, Test: {len(test_df)} images")

# Optional: check label distributions (average prevalence per class)
diseases = train_df.columns[2:]  # skip Image Index + Patient ID
print("\nLabel prevalence per set:")
print("Train:\n", train_df[diseases].mean())
print("Val:\n", val_df[diseases].mean())
print("Test:\n", test_df[diseases].mean())
