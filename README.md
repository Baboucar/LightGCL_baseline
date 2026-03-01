# LightGCL: Reproducible Baseline Implementation (PyTorch)

This repository provides a clean, reproducible PyTorch implementation
of:

Cai et al., *LightGCL: Simple Yet Effective Graph Contrastive Learning
for Recommendation*, ICLR 2023.\
Paper: https://openreview.net/forum?id=FKXVK9dyMM

This version is structured for academic reproducibility and controlled
baseline evaluation. It includes corrected score-based evaluation and
explicit support for the standard ML-1M 99-negative protocol.

------------------------------------------------------------------------

## 1. Purpose of This Repository

This implementation is designed to:

-   Provide a stable research baseline for recommender system
    experiments
-   Ensure correct ranking evaluation (score-based, not index-based)
-   Support ML-1M evaluation under the 1 positive + 99 negatives
    protocol
-   Produce reproducible logs and checkpoints

The training objective is:

Total Loss = BPR Loss + lambda1 \* Contrastive Loss + lambda2 \* L2
Regularization

------------------------------------------------------------------------

## 2. Repository Structure

LightGCL/ │ ├── data/ │ └── ml1m/ │ ├── ml-1m.train.rating │ ├──
ml-1m.test.rating │ ├── ml-1m.test.negative │ ├── trnMat.pkl │ └──
tstMat.pkl │ ├── log/ (created automatically) ├── saved_model/ (created
automatically) │ ├── main.py ├── model.py ├── utils.py ├── parser.py └──
README.md

------------------------------------------------------------------------

## 3. Environment

Tested with:

Python 3.10\
PyTorch \>= 1.12\
NumPy\
SciPy\
Pandas\
tqdm

Install dependencies:

pip install numpy scipy pandas tqdm\
pip install torch

------------------------------------------------------------------------

## 4. Dataset Preparation (ML-1M)

Required files:

-   trnMat.pkl : Sparse training matrix
-   tstMat.pkl : Sparse test matrix
-   ml-1m.test.negative : 99-negative evaluation file

Important constraints:

-   User and item IDs must be 0-indexed
-   Sparse matrices must have shape \[n_users, n_items\]
-   The negative file must follow the official ML-1M format: Each line
    begins with (user,item) followed by 99 negative item IDs

------------------------------------------------------------------------

## 5. Running Experiments

### A. Full Ranking Evaluation

Ranks each user against all items while masking training interactions:

**python main.py --data ml1m --epoch 50**

------------------------------------------------------------------------

### B. ML-1M 99-Negative Evaluation (Recommended Baseline Setting)

This is the standard protocol used in many graph recommendation papers.

It evaluates each user on: 1 positive item + 99 sampled negatives

To reproduce baseline results using 50 epochs:

python main.py --data ml1m --eval_mode neg99 --epoch 50

Recommended full configuration:

python main.py\
--data ml1m\
--eval_mode neg99\
--epoch 50\
--d 64\
--gnn_layer 2\
--q 5\
--lr 0.001\
--lambda1 0.2\
--lambda2 1e-7\
--temp 0.2\
--cuda 0

This configuration trains for 50 epochs and reports:

-   HR@10
-   NDCG@10

under the official 99-negative evaluation protocol.

------------------------------------------------------------------------

## 6. Key Arguments

--data Dataset folder name\
--epoch Number of training epochs\
--d Embedding dimension\
--gnn_layer Number of GNN layers\
--q Rank for SVD reconstruction\
--lambda1 Contrastive loss weight\
--lambda2 L2 regularization weight\
--temp Temperature for contrastive loss\
--dropout Edge dropout rate\
--lr Learning rate\
--cuda GPU index\
--eval_mode full or neg99

------------------------------------------------------------------------

## 7. Evaluation Protocol Details

In test mode:

-   The model returns raw item scores
-   Training interactions are masked
-   Rankings are computed from scores
-   In neg99 mode, ranking is performed only within the candidate set (1
    positive + 99 negatives)

This ensures comparability with published ML-1M results.

------------------------------------------------------------------------

## 8. Citation

If you use this implementation in academic work, please cite:

@inproceedings{caisimple, title={LightGCL: Simple Yet Effective Graph
Contrastive Learning for Recommendation}, author={Cai, Xuheng and Huang,
Chao and Xia, Lianghao and Ren, Xubin}, booktitle={The Eleventh
International Conference on Learning Representations}, year={2023} }

------------------------------------------------------------------------

## 9. Reproducibility Notes

For consistent results:

-   Fix random seeds in PyTorch and NumPy if required
-   Use the same negative sampling file for ML-1M
-   Report whether evaluation is full-ranking or neg99

This repository is intended for controlled research experimentation and
reproducible baseline comparison.