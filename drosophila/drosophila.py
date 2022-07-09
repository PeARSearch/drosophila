from sklearn.base import BaseEstimator, TransformerMixin

from ._fly import Fly


class FFA(BaseEstimator, TransformerMixin):
    def __init__(self, kc_size: int = 100, wta: int = 5, proj_size: int = 10, random_state=None):
        """
        The Fruit Fly Algorithm for Embeddings.

        Args:
            kc_size: Number of kenyon cells.
            wta: Percentage of Kenyon cells retained in hash.
            proj_size: Size of projections
        """
        super().__init__()
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.random_state = random_state
        self._fly = None

    def fit(self, X, y=None, **fit_params):
        pn_size = X.shape[1]
        self._fly = Fly(pn_size, self.kc_size, self.wta, self.proj_size, init_method="random")
        return self

    def transform(self, x):
        return self._fly.hash_space(x)
