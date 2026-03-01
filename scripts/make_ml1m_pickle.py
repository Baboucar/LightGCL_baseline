import pickle
import numpy as np
import scipy.sparse as sp
import os

train_path = "data/ml1m/ml-1m.train.rating"
test_path  = "data/ml1m/ml-1m.test.rating"
neg_path   = "data/ml1m/ml-1m.test.negative"
neg_out    = "data/ml1m/ml-1m.test.negative.mapped"
rating_out = "data/ml1m/ml-1m.test.rating.mapped"

def read_train_ids(path):
    U, I = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u, i, r, ts = line.strip().split("\t")
            U.append(int(u))
            I.append(int(i))
    return np.array(U, dtype=np.int32), np.array(I, dtype=np.int32)

# 1) build maps from train
tr_u_raw, tr_i_raw = read_train_ids(train_path)

uniq_u = np.unique(tr_u_raw)
uniq_i = np.unique(tr_i_raw)

user_map = {u: idx for idx, u in enumerate(uniq_u)}
item_map = {i: idx for idx, i in enumerate(uniq_i)}

n_users = len(user_map)
n_items = len(item_map)

# 2) map train
tr_u = np.array([user_map[u] for u in tr_u_raw], dtype=np.int32)
tr_i = np.array([item_map[i] for i in tr_i_raw], dtype=np.int32)
trn = sp.coo_matrix((np.ones(len(tr_u), np.float32), (tr_u, tr_i)), shape=(n_users, n_items))

# 3) map test rating (drop any test user/item unseen in train)
te_u, te_i = [], []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        u, i, r, ts = line.strip().split("\t")
        u = int(u); i = int(i)
        if u in user_map and i in item_map:
            te_u.append(user_map[u])
            te_i.append(item_map[i])

tst = sp.coo_matrix((np.ones(len(te_u), np.float32), (np.array(te_u), np.array(te_i))), shape=(n_users, n_items))

# 4) map test negatives + save mapped files for candidate eval
#    Each line: user \t neg1 \t neg2 ...
with open(neg_path, "r", encoding="utf-8") as fin, open(neg_out, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        u_raw = int(parts[0])
        if u_raw not in user_map:
            continue
        u = user_map[u_raw]
        mapped_negs = []
        ok = True
        for x in parts[1:]:
            i_raw = int(x)
            if i_raw not in item_map:
                ok = False
                break
            mapped_negs.append(str(item_map[i_raw]))
        if ok:
            fout.write(str(u) + "\t" + "\t".join(mapped_negs) + "\n")

# Also save mapped rating (so your eval reads mapped user/item)
with open(test_path, "r", encoding="utf-8") as fin, open(rating_out, "w", encoding="utf-8") as fout:
    for line in fin:
        u, i, r, ts = line.strip().split("\t")
        u_raw = int(u); i_raw = int(i)
        if u_raw in user_map and i_raw in item_map:
            fout.write(f"{user_map[u_raw]}\t{item_map[i_raw]}\n")

# 5) save pkl
with open("data/ml1m/trnMat.pkl", "wb") as f:
    pickle.dump(trn, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/ml1m/tstMat.pkl", "wb") as f:
    pickle.dump(tst, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved PKLs with consistent mapping.")
print("Shapes:", trn.shape, tst.shape)
print("Train nnz:", trn.nnz, "Test nnz:", tst.nnz)
print("Wrote:", neg_out, "and", rating_out)