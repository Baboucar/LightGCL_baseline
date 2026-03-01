import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data as data

from parser import args
from model import LightGCL
from utils import TrnData, scipy_sparse_mat_to_torch_sparse_tensor

# --------------------------
# Device
# --------------------------
device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)

# --------------------------
# Hyperparameters
# --------------------------
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
svd_q = args.q

# --------------------------
# Load data
# --------------------------
path = os.path.join("data", args.data) + os.sep

with open(path + "trnMat.pkl", "rb") as f:
    train = pickle.load(f)

train_csr = (train != 0).astype(np.float32)

with open(path + "tstMat.pkl", "rb") as f:
    test = pickle.load(f)

print("Data loaded.")
print("user_num:", train.shape[0], "item_num:", train.shape[1], "eval_mode:", args.eval_mode)

# --------------------------
# Normalize adjacency
# --------------------------
train = train.tocoo()
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()

deg_u = rowD[train.row]
deg_i = colD[train.col]
train.data = train.data / np.sqrt(deg_u * deg_i + 1e-12)

# --------------------------
# Data loader
# --------------------------
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(torch_device)
print("Adj matrix normalized.")

# --------------------------
# SVD reconstruction
# --------------------------
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(torch_device)
print("Performing SVD...")
svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del s
print("SVD done.")

# --------------------------
# Process test set
# --------------------------
test = test.tocoo()
test_labels = [[] for _ in range(test.shape[0])]
for idx in range(len(test.data)):
    u = test.row[idx]
    i = test.col[idx]
    test_labels[u].append(i)
print("Test data processed.")

# --------------------------
# Model + optimizer
# --------------------------
model = LightGCL(
    adj_norm.shape[0], adj_norm.shape[1], d,
    u_mul_s, v_mul_s, svd_u.T, svd_v.T,
    train_csr, adj_norm, l, temp, lambda_1, lambda_2,
    dropout, batch_user, device
).to(torch_device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0, lr=lr)

# --------------------------
# NEG99 loader for ML1M
# --------------------------
def load_ml1m_neg99_candidates(path):
    """
    Parses ml-1m.test.negative where each line is:
      (u,pos)  neg1 neg2 ... neg99
    We extract integers from the whole line to survive weird spacing.
    Returns dict user -> (pos, [negs])
    """
    neg_path = os.path.join(path, "ml-1m.test.negative")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"Missing: {neg_path}")

    out = {}
    with open(neg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract ALL integers from the line:
            # For "(0,25) 1064 174 ..." => [0,25,1064,174,...]
            ints = [int(x) for x in __import__("re").findall(r"\d+", line)]
            if len(ints) < 3:
                continue

            u = ints[0]
            pos = ints[1]
            negs = ints[2:]

            out[u] = (pos, negs)

    return out

# --------------------------
# Evaluation: NEG99
# --------------------------
@torch.no_grad()
def evaluate_neg99(k=10):
    model.eval()
    cand = load_ml1m_neg99_candidates(path)

    total = 0
    hr = 0.0
    ndcg = 0.0

    for u, (pos, negs) in cand.items():
        # candidate set: 1 pos + 99 neg
        items = [pos] + [i for i in negs if i != pos]
        if len(items) > 100:
            items = items[:100]

        u_tensor = torch.tensor([u], dtype=torch.long, device=torch_device)

        # full scores, masked for train items in model
        scores_all = model(u_tensor, None, None, None, test=True)  # [1, n_items]
        scores = scores_all[0, items].detach().cpu().numpy()       # [C]

        order = np.argsort(-scores)
        ranked = [items[i] for i in order]

        total += 1
        if pos in ranked[:k]:
            hr += 1.0
            rank = ranked.index(pos)
            ndcg += 1.0 / np.log2(rank + 2)

    if total == 0:
        return 0.0, 0.0
    return hr / total, ndcg / total

# --------------------------
# Train loop
# --------------------------
for epoch in range(epoch_no):
    model.train()

    if (epoch + 1) % 50 == 0:
        os.makedirs("saved_model", exist_ok=True)
        torch.save(model.state_dict(), f"saved_model/saved_model_epoch_{epoch}.pt")
        torch.save(optimizer.state_dict(), f"saved_model/saved_optim_epoch_{epoch}.pt")

    train_loader.dataset.neg_sampling()

    epoch_loss = 0.0
    epoch_loss_r = 0.0
    epoch_loss_s = 0.0

    for batch in tqdm(train_loader):
        uids, pos, neg = batch
        uids = uids.long().to(torch_device)
        pos = pos.long().to(torch_device)
        neg = neg.long().to(torch_device)
        iids = torch.concat([pos, neg], dim=0)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_r, loss_s = model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.detach().cpu())
        epoch_loss_r += float(loss_r.detach().cpu())
        epoch_loss_s += float(loss_s.detach().cpu())

    bn = len(train_loader)
    epoch_loss /= bn
    epoch_loss_r /= bn
    epoch_loss_s /= bn

    print(f"Epoch: {epoch} Loss: {epoch_loss:.6f} Loss_r: {epoch_loss_r:.6f} Loss_s: {epoch_loss_s:.6f}")

    # eval every 3 epochs
    if epoch % 3 == 0:
        if args.eval_mode == "neg99":
            hr10, nd10 = evaluate_neg99(k=args.eval_k)
            print("-------------------------------------------")
            print(f"Test of epoch {epoch} : HR@{args.eval_k}: {hr10} Ndcg@{args.eval_k}: {nd10}")
        else:
            print("eval_mode=full not implemented in this snippet (keep your full-ranking eval if you want it).")

# --------------------------
# Final evaluation
# --------------------------
if args.eval_mode == "neg99":
    hr10, nd10 = evaluate_neg99(k=args.eval_k)
    print("-------------------------------------------")
    print(f"Final test: HR@{args.eval_k}: {hr10} Ndcg@{args.eval_k}: {nd10}")