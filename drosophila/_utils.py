# Mirrored and adapted from: https://github.com/PeARSearch/PeARS-fruit-fly/blob/main/dense_fruit_fly/utils.py

import numpy as np
from scipy.sparse import csr_matrix, vstack


def wta_vectorized(feature_mat, k):
    # thanks https://stackoverflow.com/a/59405060
    m, n = feature_mat.shape
    k = int(k * n / 100)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(feature_mat, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = feature_mat[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = feature_mat < kth_vals[:, None]
    # replace mask by 0
    feature_mat[is_smaller_than_kth] = 0
    return feature_mat


def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    kc_use = np.squeeze(kc_mat.toarray().sum(axis=0, keepdims=1))
    kc_use = kc_use / sum(kc_use)
    kc_sorted_ids = np.argsort(kc_use)[:-kc_use.shape[0] - 1:-1]  # Give sorted list from most to least used KCs
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i + 2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon, kc_use, kc_sorted_ids


def hash_dataset_(dataset_mat, weight_mat, percent_hash):
    dataset_mat = csr_matrix(dataset_mat)
    hs, kc_use, kc_sorted_ids = hash_input_vectorized_(dataset_mat, weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs, kc_use, kc_sorted_ids
