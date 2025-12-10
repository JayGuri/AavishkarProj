"""
Dynamic feature-level bridge generation between B4 and A5 using
Mahalanobis-based duration and Brownian Bridge stochastic pathing.

The goal is to create a physiologically plausible transition in FEATURE space
that:
- Adapts duration to the statistical distance between states
- Preserves variance/texture via a Brownian Bridge
- Adds jitter to avoid overfitting
- Provides a soft label (0..1) along the transition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.spatial.distance import mahalanobis


@dataclass
class DynamicBridgeParams:
    step_sec: float
    base_time: float = 5.0
    alpha: float = 1.0
    min_time: float = 5.0
    max_time: float = 20.0
    jitter: float = 0.15
    random_state: Optional[int] = None


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding timeline/label columns."""
    exclude = {"time", "segment", "label", "soft_label"}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _brownian_bridge(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a standard Brownian bridge B(t) for t in [0,1] sampled at n points.
    Returns array of shape (n,).
    """
    if n <= 1:
        return np.zeros(n, dtype=float)
    # Standard Brownian motion increments
    w = rng.normal(0.0, 1.0, size=n)
    w = np.cumsum(w)
    t = np.linspace(0.0, 1.0, n)
    # Bridge B_t = W_t - t * W_1
    b = w - t * w[-1]
    # Normalize to avoid degenerate scaling for small n
    std = b.std() or 1.0
    return b / std


def generate_dynamic_bridge(
    features_b4: pd.DataFrame,
    features_a5: pd.DataFrame,
    *,
    params: DynamicBridgeParams,
    start_time: float,
) -> pd.DataFrame:
    """
    Generate a dynamic, stochastic feature bridge using Mahalanobis distance
    and a Brownian Bridge.

    Args:
        features_b4: Tail windows from B4 (DataFrame with numeric feature columns)
        features_a5: Head windows from A5 (DataFrame with numeric feature columns)
        params:    Dynamic bridge parameters (timing and jitter controls)
        start_time: Time (seconds) for the first bridge window (ensures continuity)

    Returns:
        bridge_df: DataFrame containing synthesized bridge feature windows with
                   columns matching numeric feature columns plus: 'time',
                   'segment'=='bridge', 'soft_label' in [0,1], and 'label' as NaN.
    """
    if len(features_b4) == 0 or len(features_a5) == 0:
        raise ValueError("Both features_b4 and features_a5 must be non-empty")

    cols = _numeric_feature_columns(features_b4)
    if not cols:
        raise ValueError("No numeric feature columns found to bridge")

    # Compute centroids and pooled covariance for Mahalanobis distance
    mu_b4 = features_b4[cols].mean().values
    mu_a5 = features_a5[cols].mean().values

    pooled = pd.concat([features_b4[cols], features_a5[cols]], axis=0)
    cov = np.cov(pooled.values, rowvar=False)
    # Regularize covariance to avoid singularities
    cov += np.eye(cov.shape[0]) * 1e-6
    VI = pinv(cov)
    dist = mahalanobis(mu_b4, mu_a5, VI)

    # Map distance to duration, then to number of windows
    T = params.base_time + params.alpha * dist
    T = _clamp(T, params.min_time, params.max_time)
    n_windows = max(3, int(round(T / params.step_sec)))

    rng = np.random.default_rng(params.random_state)
    t01 = np.linspace(0.0, 1.0, n_windows)

    # Feature-wise Brownian Bridge with variance matched to tails/heads
    start_std = features_b4[cols].std(ddof=0).values
    end_std = features_a5[cols].std(ddof=0).values
    var_interp = (1.0 - t01)[:, None] * start_std[None, :] + t01[:, None] * end_std[None, :]

    paths = np.zeros((n_windows, len(cols)), dtype=float)
    for j in range(len(cols)):
        s0 = mu_b4[j]
        s1 = mu_a5[j]
        bb = _brownian_bridge(n_windows, rng)
        # Weighted variance along the path with jitter control
        sigma = params.jitter * var_interp[:, j]
        paths[:, j] = (1.0 - t01) * s0 + t01 * s1 + sigma * bb

    # Assemble bridge dataframe
    bridge_df = pd.DataFrame(paths, columns=cols)
    bridge_df['time'] = start_time + np.arange(n_windows) * params.step_sec
    bridge_df['segment'] = 'bridge'
    # Soft labels from 0 (hungry) to 1 (satiated)
    bridge_df['soft_label'] = t01
    # Keep label out of training classification by setting NaN
    bridge_df['label'] = np.nan

    # Preserve any non-numeric columns present in inputs by filling with NaN or defaults
    for c in features_b4.columns:
        if c in {'time', 'segment', 'soft_label', 'label'} or c in cols:
            continue
        if c not in bridge_df.columns:
            bridge_df[c] = np.nan

    # Order columns: keep original order where possible
    ordered_cols = [c for c in features_b4.columns if c in bridge_df.columns]
    # Ensure required columns are present
    for c in ['time', 'segment', 'soft_label', 'label']:
        if c not in ordered_cols:
            ordered_cols.append(c)
    # Add any remaining cols
    for c in bridge_df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)

    bridge_df = bridge_df[ordered_cols]
    return bridge_df
