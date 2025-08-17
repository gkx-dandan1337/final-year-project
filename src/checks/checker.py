import pandas as pd

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("Train class balance:\n", train_df['label'].value_counts())
