"""The main Fly class"""

import random

import numpy as np
from scipy.sparse import lil_matrix

from ._utils import hash_dataset_


class Fly:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, init_method=None, eval_method=None,
                 proj_store=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.init_method = init_method
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        if self.init_method == "random":
            weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        else:
            weight_mat, self.shuffled_idx = self.projection_store(proj_store)

        self.projections = lil_matrix(weight_mat)
        self.val_score = 0
        self.is_evaluated = False
        self.kc_use_sorted = None
        self.kc_in_hash_sorted = None

    def hash_space(self, m):
        hs, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=m,
                                                      weight_mat=self.projections,
                                                      percent_hash=self.wta)
        hs = (hs > 0).astype(np.int_)
        return hs

    def create_projections(self, proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0, len(idx), proj_size):
                p = idx[i:i + proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c += 1
                if c >= self.kc_size:
                    break
            random.shuffle(idx)  # reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def projection_store(self, proj_store):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        self.proj_store = proj_store.copy()
        proj_size = len(self.proj_store[0])
        random.shuffle(self.proj_store)
        sidx = [pn for p in self.proj_store for pn in p]
        idx = list(range(self.pn_size))
        used_idx = sidx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(len(self.proj_store)):
                p = self.proj_store[i]
                for j in p:
                    weight_mat[c][j] = 1
                c += 1
                if c >= self.kc_size:
                    break
            random.shuffle(idx)  # add random if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]