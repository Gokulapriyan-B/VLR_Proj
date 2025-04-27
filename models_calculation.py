import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# 1) Load your routing test results (NN classifier) and extract basenames
routing_csv = '/home/ubuntu/STEAD/outputs/model4_results.csv'
routing_df  = pd.read_csv(routing_csv)
routing_df['basename'] = routing_df['name'].apply(os.path.basename)

# 2) Load model1 results and extract basenames
model1_csv = '/home/ubuntu/STEAD/outputs/model4_results.csv'
m1_df      = pd.read_csv(model1_csv)
m1_df['basename'] = m1_df['name'].apply(os.path.basename)

# 3) Keep only samples in the routing test split
merged = pd.merge(
    routing_df[['basename']],
    m1_df,
    on='basename',
    how='inner'
)

# 4) Pull out arrays for metrics:
y_true  = merged['ground_truth'].astype(int).values
y_score = merged['prediction'].values

# 5) Apply a 0.5 threshold for accuracy
y_pred = (y_score > 0.5).astype(int)

# 6) Compute metrics
acc     = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_score)
pr_auc  = average_precision_score(y_true, y_score)

print(f"Model1 on NN test split:")
print(f"  Accuracy:             {acc:.4f}")
print(f"  ROC AUC:              {roc_auc:.4f}")
print(f"  Precision‑Recall AUC: {pr_auc:.4f}")

# (Optional) also compute ROC/PR curve points
fpr, tpr, _         = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)
