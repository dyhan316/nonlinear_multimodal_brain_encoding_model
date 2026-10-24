"""
Fetch and compile BabyLM shared task benchmark scores for models used in this analysis.

Sources:
- BabyLM 2023: https://github.com/babylm/submissions2023 (model_summary.csv)
- BabyLM 2024: Tables from "BabyHGRN" paper (arxiv.org/abs/2412.15978)
  which reports official baseline scores from the 2nd BabyLM Challenge.
"""

import pandas as pd

# ============================================================
# BabyLM 2023 Baseline Scores (from official submissions2023 repo)
# Metrics: BLiMP, GLUE, MSGS, Aggregate
# Note: 2023 used GLUE (not SuperGLUE) and MSGS (not EWoK)
# Aggregate = weighted: BLiMP+BLiMP-Supp 50%, GLUE 30%, MSGS 20%
# ============================================================

data_2023 = [
    # Model ID in this project, Display Name, Track, BLiMP, GLUE, MSGS, Aggregate
    ("babylm/opt-125m-strict-2023",       "OPT 125M (100M)",           "Strict (100M)",      0.731, 0.702, 0.133, 0.603),
    ("babylm/opt-125m-strict-small-2023", "OPT 125M (10M)",            "Strict-Small (10M)", 0.597, 0.624, 0.096, 0.505),
    ("babylm/t5-base-strict-2023",        "T5 Base 220M (100M)",       "Strict (100M)",      0.537, 0.721, 0.069, 0.498),
    ("babylm/t5-base-strict-small-2023",  "T5 Base 220M (10M)",        "Strict-Small (10M)", 0.537, 0.583, -0.064, 0.431),
]

# ============================================================
# BabyLM 2024 Baseline Scores (from official evaluation)
# Metrics: BLiMP, BLiMP-Supp, EWoK, SuperGLUE, Text Average
# Source: Table 5 & 6 in BabyHGRN paper (arxiv 2412.15978)
# ============================================================

data_2024 = [
    # Model ID in this project, Display Name, Track, BLiMP, BLiMP-Supp, EWoK, SuperGLUE, Text Avg
    ("babylm/babyllama-100m-2024",   "BabyLLaMA 58M (100M)",   "Strict (100M)",      73.1, 60.6, 52.1, 69.0, 63.7),
    ("babylm-babyllama-10m-2024",    "BabyLLaMA 58M (10M)",    "Strict-Small (10M)", 69.8, 59.5, 50.7, 63.3, 60.8),
    ("babylm/ltgbert-100m-2024",     "LTGBERT BASE (100M)",    "Strict (100M)",      69.2, 66.5, 51.9, 68.4, 64.0),
    ("babylm/ltgbert-10m-2024",      "LTGBERT BASE (10M)",     "Strict-Small (10M)", 60.6, 60.8, 48.9, 60.3, 57.7),
]

# Build 2023 DataFrame
df_2023 = pd.DataFrame(data_2023, columns=[
    "model_id", "display_name", "track",
    "BLiMP", "GLUE", "MSGS", "Aggregate_2023"
])
# Convert to percentage scale for consistency
for col in ["BLiMP", "GLUE", "MSGS", "Aggregate_2023"]:
    df_2023[col] = df_2023[col] * 100

# Build 2024 DataFrame
df_2024 = pd.DataFrame(data_2024, columns=[
    "model_id", "display_name", "track",
    "BLiMP", "BLiMP_Supplement", "EWoK", "SuperGLUE", "Text_Average"
])

# ============================================================
# Combined DataFrame with all models
# ============================================================

# For a unified view, create a combined df with common columns
records = []

# 2023 models
for _, row in df_2023.iterrows():
    records.append({
        "model_id": row["model_id"],
        "display_name": row["display_name"],
        "track": row["track"],
        "challenge_year": 2023,
        "BLiMP": row["BLiMP"],
        "BLiMP_Supplement": None,  # Not reported separately in 2023
        "GLUE_or_SuperGLUE": row["GLUE"],
        "MSGS_or_EWoK": row["MSGS"],
        "Aggregate": row["Aggregate_2023"],
    })

# 2024 models
for _, row in df_2024.iterrows():
    records.append({
        "model_id": row["model_id"],
        "display_name": row["display_name"],
        "track": row["track"],
        "challenge_year": 2024,
        "BLiMP": row["BLiMP"],
        "BLiMP_Supplement": row["BLiMP_Supplement"],
        "GLUE_or_SuperGLUE": row["SuperGLUE"],
        "MSGS_or_EWoK": row["EWoK"],
        "Aggregate": row["Text_Average"],
    })

df_babylm = pd.DataFrame(records)

# Print results
print("=" * 90)
print("BabyLM Shared Task Benchmark Scores for Models Used in This Analysis")
print("=" * 90)

print("\n--- BabyLM 2023 Baselines (scores in %) ---")
print("Evaluation: BLiMP (zero-shot), GLUE (finetuned), MSGS (zero-shot)")
print("Aggregate = BLiMP+BLiMP-Supp 50% + GLUE 30% + MSGS 20%\n")
print(df_2023.to_string(index=False))

print("\n--- BabyLM 2024 Baselines (scores in %) ---")
print("Evaluation: BLiMP, BLiMP-Supplement, EWoK, SuperGLUE")
print("Text Average = mean of BLiMP, BLiMP-Supp, EWoK, SuperGLUE\n")
print(df_2024.to_string(index=False))

print("\n--- Combined View (all models, scores in %) ---")
print("Note: 2023 used GLUE+MSGS; 2024 used SuperGLUE+EWoK\n")
print(df_babylm.to_string(index=False))

# Save to CSV
df_babylm.to_csv("babylm_benchmark_scores.csv", index=False)
print("\nSaved to babylm_benchmark_scores.csv")
