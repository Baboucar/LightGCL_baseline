# utils.py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


# --------------------------
# Candidate-eval metrics (1 pos + N neg)
# --------------------------

def hr_ndcg_at_k(gt_item, ranked_items, k: int):
    """
    gt_item: int
    ranked_items: list[int] length >= k
    Returns (HR@k, NDCG@k)
    """
    topk = ranked_items[:k]
    if gt_item in topk:
        rank = topk.index(gt_item)
        return 1.0, 1.0 / np.log2(rank + 2)
    return 0.0, 0.0


# --------------------------
# Full-ranking metrics (optional)
# --------------------------

def recall_ndcg_full_rank(uids, topk_items, k: int, test_labels):
    """
    uids: np array [B]
    topk_items: np array [B, k] (item ids already sorted by score desc)
    test_labels: list[list[int]]
    Returns (Recall@k, NDCG@k)
    """
    uids = np.asarray(uids, dtype=np.int64)
    user_num = 0
    all_recall = 0.0
    all_ndcg = 0.0

    for r, u in enumerate(uids):
        gt = test_labels[u]
        if not gt:
            continue
        pred = list(topk_items[r])

        hit = 0
        dcg = 0.0
        idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(k, len(gt)))])
        if idcg <= 0:
            continue

        for item in gt:
            if item in pred:
                hit += 1
                loc = pred.index(item)
                dcg += 1.0 / np.log2(loc + 2)

        all_recall += hit / len(gt)
        all_ndcg += dcg / idcg
        user_num += 1

    if user_num == 0:
        return 0.0, 0.0
    return all_recall / user_num, all_ndcg / user_num


# --------------------------
# Sparse helpers
# --------------------------

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    scipy sparse -> torch sparse COO tensor (no deprecation warning)
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_dropout(mat, dropout: float):
    if dropout <= 0.0:
        return mat
    mat = mat.coalesce()
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout, training=True)
    return torch.sparse_coo_tensor(indices, values, mat.size(), device=mat.device)


# --------------------------
# Training dataset
# --------------------------

class TrnData(data.Dataset):
    def __init__(self, coomat):
        coomat = coomat.tocoo()
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows), dtype=np.int32)

    def neg_sampling(self):
        n_items = self.dokmat.shape[1]
        for idx in range(len(self.rows)):
            u = int(self.rows[idx])
            while True:
                i_neg = np.random.randint(n_items)
                if (u, i_neg) not in self.dokmat:
                    self.negs[idx] = i_neg
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return int(self.rows[idx]), int(self.cols[idx]), int(self.negs[idx])