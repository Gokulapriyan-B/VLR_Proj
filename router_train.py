# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import option
# from dataset import Dataset

# # Parse arguments
# args   = option.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- 1. Load raw inputs + sample names ---
# dataset = Dataset(args, test_mode=True)
# loader  = DataLoader(dataset,
#                      batch_size=args.batch_size,
#                      shuffle=False,
#                      num_workers=0,
#                      pin_memory=False)

# all_inputs = []
# all_names  = []
# for inputs, _, names in loader:
#     flat = inputs.view(inputs.size(0), -1)  # flatten [B, D_flat]
#     all_inputs.append(flat.cpu().numpy())
#     all_names.extend(names)

# X = np.vstack(all_inputs)
# all_names = np.array(all_names)

# # --- 2. Load routing labels and align by name ---
# df_labels = (
#     pd.read_csv('/home/ubuntu/STEAD/outputs/routing_data.csv')
#       .set_index('name')['routing']
# )
# y = np.array([df_labels.loc[os.path.basename(n)] for n in all_names])

# # --- 3. Train/test split (80/20 stratified) including names ---
# X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
#     X, y, all_names,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )
# print(y_train)
# # --- 4. Create a WeightedRandomSampler for balanced batches ---
# #    (keeps oversampling minorities so model sees them often)
# class_counts = np.bincount(y_train)
# print(class_counts)
# class_weights = 1.0 / class_counts
# sample_weights = class_weights[y_train]
# sampler = WeightedRandomSampler(weights=sample_weights,
#                                 num_samples=len(sample_weights),
#                                 replacement=True)

# # --- 5. Prepare DataLoaders ---
# train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                          torch.tensor(y_train, dtype=torch.long))
# test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
#                          torch.tensor(y_test, dtype=torch.long))

# train_loader = DataLoader(train_ds,
#                           batch_size=64,
#                           sampler=sampler,
#                           num_workers=0)
# test_loader  = DataLoader(test_ds,
#                           batch_size=64,
#                           shuffle=False,
#                           num_workers=0)

# # --- 6. Define a simple MLP classifier ---
# class MLPClassifier(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x):
#         return self.net(x)

# num_classes = len(np.unique(y_train))
# model       = MLPClassifier(input_dim=X_train.shape[1],
#                             hidden_dim=512,
#                             num_classes=num_classes).to(device)

# # --- 7. Unweighted loss & optimizer ---
# criterion = torch.nn.CrossEntropyLoss()  # no class weights
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# num_epochs = 20

# # --- 8. Training loop ---
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#     avg_loss = running_loss / len(train_loader.dataset)
#     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# # --- 9. Evaluate on test set ---
# model.eval()
# all_preds = []
# with torch.no_grad():
#     for xb, _ in test_loader:
#         xb = xb.to(device)
#         preds = torch.argmax(model(xb), dim=1).cpu().numpy()
#         all_preds.extend(preds)
# print(y_test, all_preds)
# print("Test Accuracy:", accuracy_score(y_test, all_preds))
# print("\nClassification Report:\n", classification_report(y_test, all_preds))

# # --- 10. Save results ---
# df_out = pd.DataFrame({
#     'name':            names_test,
#     'true_routing':    y_test,
#     'predicted_routing': all_preds
# })
# output_path = '/home/ubuntu/STEAD/outputs/routing_test_results_nn.csv'
# df_out.to_csv(output_path, index=False)
# print(f"Saved NN test results to {output_path}")

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report

# import option
# from dataset import Dataset

# # --- Parse args & device ---
# args   = option.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   # synchronous CUDA errors

# # --- 1. Load raw inputs + names ---
# dataset = Dataset(args, test_mode=True)
# loader  = DataLoader(dataset,
#                      batch_size=args.batch_size,
#                      shuffle=False,
#                      num_workers=0,
#                      pin_memory=False)

# all_inputs, all_names = [], []
# for inputs, _, names in loader:
#     flat = inputs.view(inputs.size(0), -1)  # flatten [B, D_flat]
#     all_inputs.append(flat.cpu().numpy())
#     all_names.extend(names)

# X         = np.vstack(all_inputs)
# all_names = np.array(all_names)

# # --- 2. Load routing labels by basename ---
# df_labels = (
#     pd.read_csv('/home/ubuntu/STEAD/outputs/routing_data.csv')
#       .set_index('name')['routing']
# )
# y = np.array([df_labels.loc[os.path.basename(n)] for n in all_names])

# # --- 3. Train/Val/Test split (80/10/10 stratified) ---
# X_train_val, X_test, y_train_val, y_test, names_train_val, names_test = train_test_split(
#     X, y, all_names,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )
# X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
#     X_train_val, y_train_val, names_train_val,
#     test_size=0.1,
#     stratify=y_train_val,
#     random_state=42
# )

# # --- 4. Feature normalization ---
# scaler    = StandardScaler()
# X_train   = scaler.fit_transform(X_train)
# X_val     = scaler.transform(X_val)
# X_test    = scaler.transform(X_test)

# # --- 5. Safe class‐weights for loss & sampler ---
# class_counts = np.bincount(y_train).astype(np.float32) + 1e-6
# class_weights = 1.0 / class_counts
# class_weights[class_counts <= 1e-6] = 0
# # normalize sum of weights to num_classes (optional)
# class_weights *= len(class_counts) / class_weights.sum()

# # for CrossEntropyLoss
# weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

# # for sampler (oversample rare classes)
# sample_weights = class_weights[y_train]
# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(sample_weights),
#     replacement=True
# )

# # --- 6. Prepare DataLoaders ---
# train_ds = TensorDataset(
#     torch.tensor(X_train, dtype=torch.float32),
#     torch.tensor(y_train, dtype=torch.long)
# )
# val_ds = TensorDataset(
#     torch.tensor(X_val, dtype=torch.float32),
#     torch.tensor(y_val, dtype=torch.long)
# )
# test_ds = TensorDataset(
#     torch.tensor(X_test, dtype=torch.float32),
#     torch.tensor(y_test, dtype=torch.long)
# )

# train_loader = DataLoader(train_ds,
#                           batch_size=64,
#                           sampler=sampler,
#                           num_workers=0)
# val_loader   = DataLoader(val_ds,
#                           batch_size=64,
#                           shuffle=False,
#                           num_workers=0)
# test_loader  = DataLoader(test_ds,
#                           batch_size=64,
#                           shuffle=False,
#                           num_workers=0)

# # --- 7. Define MLP with Dropout & BatchNorm ---
# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, x):
#         return self.net(x)

# num_classes = len(np.unique(y_train))
# model = MLPClassifier(
#     input_dim=X_train.shape[1],
#     hidden_dim=512,
#     num_classes=num_classes
# ).to(device)

# # --- 8. Loss, optimizer, scheduler ---
# criterion = nn.CrossEntropyLoss(weight=weight_tensor)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.5,
#     patience=3,
#     verbose=True
# )

# num_epochs = 20

# # --- 9. Training + validation loop ---
# for epoch in range(num_epochs):
#     # ---- train ----
#     model.train()
#     train_loss = 0.0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         # sanity check
#         assert 0 <= yb.min() and yb.max() < num_classes

#         optimizer.zero_grad()
#         logits = model(xb)
#         loss   = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * xb.size(0)

#     avg_train_loss = train_loss / len(train_loader.dataset)

#     # ---- validate ----
#     model.eval()
#     val_loss = 0.0
#     val_preds, val_targets = [], []
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             logits = model(xb)
#             loss   = criterion(logits, yb)
#             val_loss += loss.item() * xb.size(0)

#             val_preds.extend(logits.argmax(dim=1).cpu().numpy())
#             val_targets.extend(yb.cpu().numpy())

#     avg_val_loss = val_loss / len(val_loader.dataset)
#     val_acc      = accuracy_score(val_targets, val_preds)

#     print(f"Epoch {epoch+1}/{num_epochs} "
#           f"| Train Loss: {avg_train_loss:.4f} "
#           f"| Val Loss: {avg_val_loss:.4f} "
#           f"| Val Acc:  {val_acc:.4f}")

#     scheduler.step(avg_val_loss)

# # --- 10. Final test evaluation ---
# model.eval()
# all_preds = []
# with torch.no_grad():
#     for xb, _ in test_loader:
#         xb    = xb.to(device)
#         preds = model(xb).argmax(dim=1).cpu().numpy()
#         all_preds.extend(preds)

# print("Test Accuracy:", accuracy_score(y_test, all_preds))
# print("\nClassification Report:\n", classification_report(y_test, all_preds))

# # --- 11. Save results ---
# df_out = pd.DataFrame({
#     'name':             names_test,
#     'true_routing':     y_test,
#     'predicted_routing': all_preds
# })
# output_path = '/home/ubuntu/STEAD/outputs/routing_test_results_nn.csv'
# df_out.to_csv(output_path, index=False)
# print(f"Saved NN test results to {output_path}")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import option
from dataset import Dataset

# --- Parse args & device ---
args   = option.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force synchronous CUDA errors for easier debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- 1. Load raw inputs + names ---
dataset = Dataset(args, test_mode=True)
loader  = DataLoader(dataset,
                     batch_size=args.batch_size,
                     shuffle=False,
                     num_workers=0,
                     pin_memory=False)

all_inputs, all_names = [], []
for inputs, _, names in loader:
    flat = inputs.view(inputs.size(0), -1)  # flatten [B, D_flat]
    all_inputs.append(flat.cpu().numpy())
    all_names.extend(names)

X         = np.vstack(all_inputs)
all_names = np.array(all_names)

# --- 2. Load raw routing labels by basename ---
df_labels = (
    pd.read_csv('/home/ubuntu/STEAD/outputs/routing_data.csv')
      .set_index('name')['routing']
)
raw_y = np.array([df_labels.loc[os.path.basename(n)] for n in all_names])

# --- 2b. Remap to 0...C-1 ---
#   This ensures no label >= num_classes or < 0
classes, y = np.unique(raw_y, return_inverse=True)
print(f"Original classes: {classes}, remapped into 0...{len(classes)-1}")

# --- 3. Train/Val/Test split (80/10/10 stratified) ---
X_train_val, X_test, y_train_val, y_test, names_train_val, names_test = train_test_split(
    X, y, all_names,
    test_size=0.20,
    stratify=y,
    random_state=42
)
X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
    X_train_val, y_train_val, names_train_val,
    test_size=0.10,
    stratify=y_train_val,
    random_state=42
)

# --- 4. Feature normalization ---
scaler    = StandardScaler()
X_train   = scaler.fit_transform(X_train)
X_val     = scaler.transform(X_val)
X_test    = scaler.transform(X_test)

# --- 5. Safe class‑weights & sampler ---
num_classes   = len(classes)
class_counts  = np.bincount(y_train).astype(np.float32) + 1e-6
class_weights = 1.0 / class_counts
class_weights[class_counts <= 1e-6] = 0
# optional: renormalize so weights sum to num_classes
class_weights *= num_classes / class_weights.sum()

# for the loss fn
weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

# for the sampler
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# --- 6. Build DataLoaders ---
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds,
                          batch_size=64,
                          sampler=sampler,
                          num_workers=0)
val_loader   = DataLoader(val_ds,
                          batch_size=64,
                          shuffle=False,
                          num_workers=0)
test_loader  = DataLoader(test_ds,
                          batch_size=64,
                          shuffle=False,
                          num_workers=0)

# --- 7. Define MLP with Dropout & BatchNorm ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLPClassifier(
    input_dim=X_train.shape[1],
    hidden_dim=512,
    num_classes=num_classes
).to(device)

# --- 8. Loss, optimizer, scheduler (no verbose warning) ---
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

num_epochs = 100

# --- 9. Train + validate ---
for epoch in range(num_epochs):
    # Train
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # Sanity check: now guaranteed 0 <= yb < num_classes
        assert 0 <= yb.min() and yb.max() < num_classes

        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # Validate
    model.eval()
    total_val_loss = 0.0
    val_preds, val_targs = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            total_val_loss += loss.item() * xb.size(0)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_targs.extend(yb.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_acc      = accuracy_score(val_targs, val_preds)

    print(f"Epoch {epoch+1}/{num_epochs}  "
          f"| Train Loss: {avg_train_loss:.4f}  "
          f"| Val Loss: {avg_val_loss:.4f}  "
          f"| Val Acc:  {val_acc:.4f}")

    scheduler.step(avg_val_loss)

# --- 10. Test evaluation ---
model.eval()
test_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        test_preds.extend(model(xb).argmax(dim=1).cpu().numpy())

print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("\nClassification Report:\n", classification_report(y_test, test_preds))

# --- 11. Save results ---
df_out = pd.DataFrame({
    'name':              names_test,
    'true_routing':      y_test,
    'predicted_routing': test_preds
})
output_path = '/home/ubuntu/STEAD/outputs/routing_test_results_nn.csv'
df_out.to_csv(output_path, index=False)
print(f"Saved NN test results to {output_path}")
