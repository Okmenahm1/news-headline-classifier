from datasets import load_dataset
import pandas as pd

# 1) تحميل الداتا من HuggingFace
dataset = load_dataset("wangrongsheng/ag_news")

# 2) تحويل train و test إلى DataFrame
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# 3) دمجهم مع بعض (اختياري)
full_df = pd.concat([train_df, test_df], ignore_index=True)

print("Columns:", full_df.columns)
print("First rows:")
print(full_df.head())

# 4) حفظ الداتا كـ CSV
full_df.to_csv("ag_news.csv", index=False)

print("\n✅ Dataset saved as ag_news.csv")
