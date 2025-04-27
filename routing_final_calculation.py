import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

# --- File paths (adjust as needed) ---
pred_csv    = '/home/ubuntu/STEAD/outputs/routing_test_results_nn.csv'
model1_csv  = '/home/ubuntu/STEAD/outputs/model1_results.csv'
model2_csv  = '/home/ubuntu/STEAD/outputs/model2_results.csv'
model3_csv  = '/home/ubuntu/STEAD/outputs/model3_results.csv'

# --- 1. Load routing predictions ---
pred_df = pd.read_csv(pred_csv)
pred_df['basename'] = pred_df['name'].apply(os.path.basename)

# --- 2. Load each model's results ---
m1 = (pd.read_csv(model1_csv)
        .assign(basename=lambda df: df['name'].map(os.path.basename))
        .set_index('basename')[['prediction','ground_truth']]
        .rename(columns={'prediction':'pred1','ground_truth':'gt'}))
m2 = (pd.read_csv(model2_csv)
        .assign(basename=lambda df: df['name'].map(os.path.basename))
        .set_index('basename')[['prediction']]
        .rename(columns={'prediction':'pred2'}))
m3 = (pd.read_csv(model3_csv)
        .assign(basename=lambda df: df['name'].map(os.path.basename))
        .set_index('basename')[['prediction']]
        .rename(columns={'prediction':'pred3'}))

# --- 3. Merge routing + model results on basename ---
df = (pred_df
      .merge(m1, left_on='basename', right_index=True, how='left')
      .merge(m2, left_on='basename', right_index=True, how='left')
      .merge(m3, left_on='basename', right_index=True, how='left'))

# --- 4. Select prediction from the routed model ---
def select_score(row):
    rt = row['predicted_routing']
    if rt == 0:
        return row['pred1']
    elif rt == 1:
        return row['pred2']
    elif rt == 2:
        return row['pred3']

df['y_score'] = df.apply(select_score, axis=1)

# --- 5. Derive binary predictions by thresholding at 0.5 ---
df['y_pred'] = (df['y_score'] >= 0.5).astype(int)

# --- 6. Ground‑truth labels ---
# use model1's ground_truth as the true labels
df['y_true'] = df['gt'].astype(int)

print(df['y_pred'], df['y_true'])

# --- 7. Compute metrics ---
acc     = accuracy_score(df['y_true'], df['y_pred'])
roc_auc = roc_auc_score(df['y_true'], df['y_score'])
pr_auc  = average_precision_score(df['y_true'], df['y_score'])

print(f"Accuracy:             {acc:.4f}")
print(f"ROC AUC:              {roc_auc:.4f}")
print(f"Precision‑Recall AUC: {pr_auc:.4f}")

# (Optional) compute curve points
fpr, tpr, _         = roc_curve(df['y_true'], df['y_score'])
precision, recall, _ = precision_recall_curve(df['y_true'], df['y_score'])

# --- 8. Save merged evaluation table ---
out_csv = '/home/ubuntu/STEAD/outputs/routing_eval_with_model_predictions.csv'
df.to_csv(out_csv, index=False)
print(f"Saved detailed results to: {out_csv}")
