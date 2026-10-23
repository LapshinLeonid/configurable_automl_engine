from __future__ import annotations

import numpy as np
import pandas as pd


def create_mock_df(rows=100, cols=5, target="target"):
    """
    Generates a synthetic DataFrame for testing.

    Parameters:
    ----------
    rows : int
        Number of rows in the generated dataset.
    cols : int
        Number of features (columns prefixed with 'feature_').
    target : str
        Name of the target variable.

    Returns:
    -----------
    pd.DataFrame
        DataFrame object containing random floating-point numbers.
    """

    # Fix the seed for test reproducibility
    np.random.seed(42)

    # Generate column names: feature_0, feature_1, ...
    column_names = [f"feature_{i}" for i in range(cols)]

    # Generate the feature matrix (normal distribution)
    data = np.random.randn(rows, cols)

    # Create the DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # Generate the target variable (binary classification or regression)
    # In this case, generate a random continuous value
    df[target] = np.random.rand(rows)

    return df


if __name__ == "__main__":
    # Usage example for verification
    test_df = create_mock_df(rows=10, cols=3)
    print("Generated DataFrame:")
    print(test_df.head())
    print(f"\nData format: {test_df.shape}")


# ──────────────────────────────────────────────────────────────────────────────
#  Reference and degenerate datasets for the regression
#  benchmark "before/after" the Sanity Gate. All generators use fixed seeds,
#  so run reproducibility is guaranteed (sensitive real data is not added to
#  git).
#
#  Returned format: (X: np.ndarray (n, p), y: np.ndarray (n,)).
# ──────────────────────────────────────────────────────────────────────────────


def make_reference_noisy_r2_03(n: int = 300, p: int = 8, seed: int = 42):
    """Reference noisy dataset with a theoretical R²≈0.3.

    Linear signal (sum of the first 4 features) plus noise tuned so that
    R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3 exactly. Honest linear
    models (ridge/lasso/elasticnet) must pass all the gate contours:
    diversity ≈ R² ≈ 0.3 > 0.15, gap ≤ 1.5.

    Args:
        n: Number of rows.
        p: Number of features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    signal = X[:, :4].sum(axis=1)
    noise_var = 4.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)
    return X, y


def make_reference_wide(
    n: int = 300, p: int = 200, k: int = 10, w: float = 0.6, seed: int = 42
):
    """Reference wide dataset: hundreds of features, few informative ones.

    Weak linear relationship through ``k`` informative features against p−k
    noise columns. Sparse-linear models (lasso/elasticnet) honestly win;
    ridge with the default search space is noticeably worse and does not
    make it into the finalist pool. The gate must not fire.

    Args:
        n: Number of rows.
        p: Number of features (hundreds).
        k: Number of informative features.
        w: Weight of informative features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = w * X[:, :k].sum(axis=1) + rng.randn(n)
    return X, y


def make_reference_small_n(n: int = 80, p: int = 6, seed: int = 42):
    """Reference small-N dataset (small samples).

    Clean linear signal with moderate noise: at N=80 honest linear models
    stably pass the gate (diversity ≫ 0.15, gap ≤ 1.5), and the scatter
    of results is suppressed by the fixed seed.

    Args:
        n: Number of rows (small N).
        p: Number of features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


def make_reference_boundary_diversity(
    n: int = 400, p: int = 4, signal: float = 0.46, seed: int = 42
):
    """Reference boundary dataset: diversity of an honest model near 0.15.

    Weak signal: OOF diversity of a trained Ridge ≈ 0.155 with the threshold
    ``min_prediction_diversity=0.15`` (margin < 5%). Documented tolerance for
    "deterioration ≤ ε": a model sitting at the threshold boundary must NOT be
    disqualified (equality/strict shortfall is the only failure point), and
    the winner must not change.

    Args:
        n: Number of rows.
        p: Number of features.
        signal: Signal amplitude (defines diversity ≈ signal²/(signal²+1)).
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = signal * X[:, 0] + rng.randn(n)
    return X, y


def make_degenerate_flat_shelf(n: int = 150, p: int = 10, seed: int = 42):
    """Degenerate "shelf" dataset: strong regularization zeroes the weights.

    Weak linear relationship: ElasticNet with alpha=500 zeroes ALL coefficients
    (``coef_ == 0``) and produces flat (constant) OOF predictions — the gate
    must fire (contours A/B/C). An honest Ridge on the same data passes.

    Args:
        n: Number of rows.
        p: Number of features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 0.5 * X[:, 0] + rng.randn(n)
    return X, y


def make_degenerate_tanh_plateau(n: int = 200, p: int = 6, seed: int = 42):
    """Degenerate "tanh plateau" dataset: SVR-sigmoid hits a plateau.

    Linear signal with noise: SVR(kernel='sigmoid', gamma=0.0, coef0=0.0)
    produces almost constant predictions (scatter ≪ scatter of y), so its
    R²_oof is negative — the model is worse than a constant prediction and
    the gate must fire (contour A, the adaptive negative-R² test). An honest
    SVR-RBF (C=1.0) passes. Note: a slightly non-zero gamma (e.g. 0.001)
    yields a small positive R²_oof and is thus NOT degenerate anymore —
    under the adaptive diversity circuit such a model is honestly better
    than the constant baseline.

    Args:
        n: Number of rows.
        p: Number of features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 1.2
    return X, y


def make_degenerate_overfit(n: int = 120, p: int = 6, seed: int = 42):
    """Degenerate "overfit" dataset: a tree with RMSE_train≈0.

    Linear signal with noise: DecisionTreeRegressor(max_depth=20) learns the
    train set perfectly (RMSE_full≈0) — the gate must fire (contour D,
    division by zero). An honest Ridge passes.

    Args:
        n: Number of rows.
        p: Number of features.
        seed: Random generator seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.5
    return X, y
