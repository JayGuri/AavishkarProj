"""
Hunger index construction and polynomial analysis
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import cfg
from .feature_extraction import standardize_features


def build_hunger_index_weighted(feat_df: pd.DataFrame, 
                                weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Build hunger index using data-driven logistic regression coefficients
    
    This approach learns the optimal feature weights from the data itself,
    ensuring hunger is consistently high (or low) across different subjects.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Use only key features to prevent overfitting
    key_features = ['alpha_rel', 'beta_rel', 'emg_rms']
    
    # Separate B4 and A5 for training
    train_mask = feat_df['segment'].isin(['B4', 'A5'])
    train_df = feat_df[train_mask].copy()
    
    # Standardize features
    feat_std, _ = standardize_features(train_df)
    
    # Select only key features that exist
    available_features = [f for f in key_features if f in feat_std.columns]
    
    if len(available_features) == 0:
        # Fallback: use all available features
        available_features = [c for c in feat_std.columns if c not in ['time', 'segment', 'label']]
    
    X_train = feat_std[available_features].values
    y_train = train_df['label'].values
    
    # Fit logistic regression to learn optimal weights
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Use learned coefficients as weights
    learned_weights = dict(zip(available_features, lr.coef_[0]))
    
    # Apply weights to all data (including bridge)
    feat_std_all, _ = standardize_features(feat_df)
    idx = np.zeros(len(feat_std_all))
    
    for feat, w in learned_weights.items():
        if feat in feat_std_all.columns:
            idx += w * feat_std_all[feat].values
    
    # Ensure hunger (B4) is positive, satiety (A5) is negative
    b4_mean = idx[feat_df['segment'] == 'B4'].mean()
    a5_mean = idx[feat_df['segment'] == 'A5'].mean()
    
    if b4_mean < a5_mean:
        idx = -idx
    
    return idx


def build_hunger_index_pca(feat_df: pd.DataFrame) -> np.ndarray:
    """Build hunger index from PCA (PC1)"""
    feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(feat_df[feat_cols]).flatten()
    
    corr = np.corrcoef(pc1, feat_df['label'].values)[0, 1]
    if corr < 0:
        pc1 = -pc1
    
    return pc1


def plot_hunger_indices(feat_df: pd.DataFrame, idx_weighted: np.ndarray, 
                        idx_pca: np.ndarray, show: bool = False):
    """Compare weighted and PCA-based hunger indices"""
    t = feat_df['time'].values  # Already in seconds
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi, sharex=True)
    
    colors = feat_df['segment'].map({'B4': cfg.color_b4, 
                                     'bridge': cfg.color_bridge, 
                                     'A5': cfg.color_a5})
    
    axes[0].scatter(t, idx_weighted, c=colors, s=15, alpha=0.6)
    axes[0].set_ylabel('Hunger Index')
    axes[0].set_title('Weighted Hunger Index (Higher = More Hungry)')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='k', ls='--', alpha=0.3)
    
    axes[1].scatter(t, idx_pca, c=colors, s=15, alpha=0.6)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Hunger Index (PC1)')
    axes[1].set_title('PCA-based Hunger Index')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='k', ls='--', alpha=0.3)
    
    handles = [mpatches.Patch(color=cfg.color_b4, label='B4 (Hunger)'),
               mpatches.Patch(color=cfg.color_bridge, label='Bridge'),
               mpatches.Patch(color=cfg.color_a5, label='A5 (Satiety)')]
    axes[0].legend(handles=handles, loc='upper right', ncol=3)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def fit_polynomial(t: np.ndarray, idx: np.ndarray, deg: int = 5) -> Tuple:
    """Fit polynomial to hunger index"""
    coeffs = np.polyfit(t, idx, deg)
    poly = np.poly1d(coeffs)
    fitted = poly(t)
    
    return coeffs, fitted, poly


def compute_derivatives(t: np.ndarray, poly: np.poly1d) -> Tuple[np.ndarray, np.ndarray]:
    """Compute first and second derivatives of polynomial"""
    poly_deriv1 = np.polyder(poly, 1)
    poly_deriv2 = np.polyder(poly, 2)
    
    deriv1 = poly_deriv1(t)
    deriv2 = poly_deriv2(t)
    
    return deriv1, deriv2


def plot_polynomial_fit(t: np.ndarray, idx: np.ndarray, fitted: np.ndarray,
                       deriv1: np.ndarray, segments: np.ndarray, show: bool = False):
    """Plot polynomial fit and derivative"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi, sharex=True)
    
    seg_map = {'B4': cfg.color_b4, 'bridge': cfg.color_bridge, 'A5': cfg.color_a5}
    colors = [seg_map.get(s, 'gray') for s in segments]
    
    axes[0].scatter(t, idx, c=colors, s=15, alpha=0.4, label='Raw index')
    axes[0].plot(t, fitted, 'k', lw=2.5, label='Polynomial fit (deg 5)')
    axes[0].set_ylabel('Hunger Index')
    axes[0].set_title('Hunger Index with Polynomial Fit')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='gray', ls='--', alpha=0.3)
    
    axes[1].plot(t, deriv1, cfg.color_bridge, lw=2, label='First derivative')
    axes[1].fill_between(t, 0, deriv1, alpha=0.3, color=cfg.color_bridge)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('dIndex/dt')
    axes[1].set_title('Rate of Change (First Derivative)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='k', ls='-', alpha=0.5, lw=1)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def detect_breakpoints_derivative(t: np.ndarray, deriv1: np.ndarray, 
                                  deriv2: np.ndarray, 
                                  threshold_factor: float = 1.5,
                                  smooth_window: int = 30,
                                  edge_margin: float = 0.1) -> List[float]:
    """Detect breakpoints from derivative magnitude with smoothing and edge constraints
    
    Args:
        t: Time array
        deriv1: First derivative
        deriv2: Second derivative
        threshold_factor: Multiplier for std threshold
        smooth_window: Window size for moving average smoothing
        edge_margin: Ignore first/last X% of timeline (0.1 = 10%)
    """
    # Apply moving average smoothing
    if smooth_window > 1 and len(deriv1) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        deriv1_smooth = np.convolve(deriv1, kernel, mode='same')
    else:
        deriv1_smooth = deriv1.copy()
    
    deriv_abs = np.abs(deriv1_smooth)
    
    # Define valid region (ignore edges)
    n = len(t)
    edge_n = int(n * edge_margin)
    valid_start = max(0, edge_n)
    valid_end = min(n, n - edge_n)
    
    # Only consider interior region
    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[valid_start:valid_end] = True
    
    # Threshold only on valid region
    threshold = threshold_factor * np.std(deriv_abs[valid_mask])
    
    high_deriv_idx = (deriv_abs > threshold) & valid_mask
    
    if not np.any(high_deriv_idx):
        return []
    
    regions = []
    in_region = False
    start = 0
    
    for i, is_high in enumerate(high_deriv_idx):
        if is_high and not in_region:
            start = i
            in_region = True
        elif not is_high and in_region:
            regions.append((start, i))
            in_region = False
    
    if in_region:
        regions.append((start, len(high_deriv_idx)))
    
    breakpoints = []
    for start, end in regions:
        mid_idx = (start + end) // 2
        breakpoints.append(t[mid_idx])
    
    return breakpoints


def plot_breakpoints(t: np.ndarray, idx: np.ndarray, fitted: np.ndarray,
                    deriv1: np.ndarray, bkpts_deriv: List[float], segments: np.ndarray, show: bool = False):
    """Visualize detected breakpoints"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi, sharex=True)
    
    seg_map = {'B4': cfg.color_b4, 'bridge': cfg.color_bridge, 'A5': cfg.color_a5}
    colors = [seg_map.get(s, 'gray') for s in segments]
    
    axes[0].scatter(t, idx, c=colors, s=15, alpha=0.4)
    axes[0].plot(t, fitted, 'k', lw=2.5, label='Polynomial fit')
    
    for bp in bkpts_deriv:
        axes[0].axvline(bp, color='red', ls='--', lw=2, alpha=0.7)
    
    axes[0].set_ylabel('Hunger Index')
    axes[0].set_title('Detected Breakpoints')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(t, deriv1, cfg.color_bridge, lw=2)
    axes[1].fill_between(t, 0, deriv1, alpha=0.3, color=cfg.color_bridge)
    
    for bp in bkpts_deriv:
        axes[1].axvline(bp, color='red', ls='--', lw=2, alpha=0.7)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('dIndex/dt')
    axes[1].set_title('First Derivative with Detected Breakpoints')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='k', ls='-', alpha=0.5, lw=1)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
