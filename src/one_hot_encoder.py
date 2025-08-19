import pandas as pd

# Path to original CSV
CSV_PATH = "final-year-project\data\Data_Entry_2017_v2020 (1).csv"
OUTPUT_PATH = "data/labels_onehot.csv"

# The 14 disease labels (CheXNet setup)
diseases = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# -------------------------
# Step 1: Read CSV
# -------------------------
df = pd.read_csv(CSV_PATH)

# -------------------------
# Step 2: Create one-hot columns
# -------------------------
for disease in diseases:
    df[disease] = df["Finding Labels"].apply(
        lambda x: 1 if disease in x else 0
    )

# Special handling: "No Finding" → all zeros
df.loc[df["Finding Labels"] == "No Finding", diseases] = 0

# -------------------------
# Step 3: Save processed CSV
# -------------------------
keep_cols = ["Image Index", "Patient ID"] + diseases
df_out = df[keep_cols]
df_out.to_csv(OUTPUT_PATH, index=False)

print("✅ Saved one-hot labels at", OUTPUT_PATH)
print(df_out.head(10))
