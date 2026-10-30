"""SMOTER-style synthetic generation for a regression target.

Holds ``_SMOTERGenerator``: interpolation between a row and one of its
nearest neighbors, the target interpolated with the same weight, optionally
density-weighted. The module must not import the other oversampling modules.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class _SMOTERGenerator:
    """SMOTER-style synthetic sample generator for regression targets.

    imbalanced-learn does not expose the (row, neighbor, gap) triplets it
    uses internally, so the engine implements its own generator following the
    SMOTER scheme (Torgo et al.): for a synthetic sample between row ``i``
    and its neighbor ``n`` (from the same bin) with a random gap ``u`` in
    ``(0, 1)``, both the features and the target are interpolated:

    ``x = x_i + u * (x_n - x_i)`` and ``y = y_i + u * (y_n - y_i)``.

    Neighbors are found with ``NearestNeighbors`` within each bin, which
    keeps the synthetic targets honest: the interpolated ``y`` always lies
    strictly between the two parents' values instead of collapsing every
    synthetic row onto a bin midpoint.

    Categorical columns follow the SMOTENC rules: a category that differs
    between the two parents keeps the first parent's (row ``i``) value;
    a shared category is kept as is.

    ``density_weighted=True`` switches the parent selection to the ADASYN
    scheme: the per-parent number of generated rows is proportional to its
    "density" — the share of k-NN neighbors (over the whole dataset) that
    belong to OTHER bins. Rows in sparse regions of the feature space are
    therefore copied more often than rows in dense regions, which visibly
    differs from the uniform random parent selection of SMOTE.
    """

    def __init__(
        self,
        *,
        random_state: int | None = None,
        categorical_features: list[int] | None = None,
        density_weighted: bool = False,
    ):
        """Initialize the generator.

        Args:
            random_state (int | None): Random seed for reproducibility.
            categorical_features (list[int] | None): Indices of categorical
                columns; they are never interpolated (SMOTENC
                copy-from-first-parent rules apply instead).
            density_weighted (bool): Use ADASYN-style density weighting of
                the parent selection instead of uniform random choice.
        """
        self.random_state = random_state
        self.categorical_features = set(categorical_features or [])
        self.density_weighted = density_weighted

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binned: np.ndarray,
        n_to_generate: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ``n_to_generate[bin]`` synthetic rows per quantile bin.

        Args:
            X (np.ndarray): Numeric (already encoded) feature matrix.
            y (np.ndarray): The real regression target values per row.
            y_binned (np.ndarray): Quantile-bin code per row (``-1`` —
                missing target, never used as a synthetic parent).
            n_to_generate (dict[int, int]): Number of synthetic rows to
                create per bin code.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of the stacked synthetic
                feature rows and the interpolated synthetic target values.
        """
        rng = np.random.RandomState(self.random_state)
        X_synth: list[np.ndarray] = []
        y_synth: list[np.ndarray] = []

        # ADASYN-style density: for every row, the share of its k-NN
        # neighbors (over the whole dataset) that belong to OTHER bins.
        # Rows whose neighborhood is dominated by other bins are "harder"
        # and receive proportionally more synthetic clones.
        density: np.ndarray | None = None
        if self.density_weighted and len(X) >= 2:
            k_global = min(5, len(X) - 1)
            nn_global = NearestNeighbors(n_neighbors=k_global + 1).fit(X)
            neigh_global = nn_global.kneighbors(X, return_distance=False)[:, 1:]
            # Neighbor "belongs to another bin" when its bin differs from
            # the row's; missing-target rows (code -1) count as different.
            other = y_binned[neigh_global] != y_binned[:, None]
            density = other.sum(axis=1) / k_global

        for _bin, n in n_to_generate.items():
            if n <= 0:
                continue
            mask = y_binned == _bin
            if mask.sum() < 2:
                continue
            bin_X = X[mask]
            bin_y = y[mask]
            n_bin = len(bin_X)
            k = min(5, n_bin - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(bin_X)
            neigh = nn.kneighbors(bin_X, return_distance=False)[:, 1:]
            # Rows with a k-NN query that returns no neighbor cannot be
            # synthesized (SMOTE contract: n_samples >= k_neighbors + 1).
            queryable = neigh.sum(axis=1) >= 1
            if not queryable.any():
                continue
            pool_idx = np.flatnonzero(queryable)
            n_gen = int(n)
            if self.density_weighted and density is not None:
                # ADASYN: distribute the bin's synthetic budget among the
                # queryable parents proportionally to their density.
                parents_mask = np.zeros(len(bin_X), dtype=bool)
                parents_mask[pool_idx] = True
                w = density[mask][parents_mask]
                w_sum = w.sum()
                if w_sum <= 0:
                    # No queryable parent has a cross-bin neighbor (constant
                    # or well-separated features): fall back to the uniform
                    # parent selection of SMOTE.
                    chosen = rng.randint(0, len(pool_idx), size=n_gen)
                    rows = pool_idx[chosen]
                else:
                    w = w / w_sum
                    counts = rng.multinomial(n_gen, w)
                    rows = np.repeat(pool_idx, counts)
            else:
                chosen = rng.randint(0, len(pool_idx), size=n_gen)
                rows = pool_idx[chosen]
            neigh_chosen = neigh[rows, 0]
            gaps = rng.uniform(0.0, 1.0, size=len(rows)).reshape(-1, 1)
            feat_diff = bin_X[neigh_chosen] - bin_X[rows]
            new_X = bin_X[rows] + gaps * feat_diff
            # Categorical columns: SMOTENC rule — a category differing
            # between the parents keeps the first parent's value.
            if self.categorical_features:
                new_X = new_X.copy()
                for col in self.categorical_features:
                    new_X[:, col] = bin_X[rows, col]
            new_y = bin_y[rows] + gaps.ravel() * (bin_y[neigh_chosen] - bin_y[rows])
            X_synth.append(new_X)
            y_synth.append(new_y)
        if not X_synth:
            return np.empty((0, X.shape[1])), np.empty(0)
        return np.vstack(X_synth), np.concatenate(y_synth)
