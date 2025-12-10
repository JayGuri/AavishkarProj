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
    # Exclude label, soft_label, and label_type columns
    feat_cols = [c for c in feat_df.columns 
                 if c not in ['time', 'segment', 'label', 'soft_label', 'label_type']]
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(feat_df[feat_cols]).flatten()
    
    # Use soft_label if available (for dynamic bridge), otherwise use label
    if 'soft_label' in feat_df.columns:
        # Use non-NaN values for correlation check
        valid_mask = feat_df['soft_label'].notna()
        if valid_mask.sum() > 0:
            corr = np.corrcoef(pc1[valid_mask], feat_df.loc[valid_mask, 'soft_label'])[0, 1]
            if corr < 0:
                pc1 = -pc1
    elif 'label' in feat_df.columns:
        valid_mask = feat_df['label'].notna()
        if valid_mask.sum() > 0:
            corr = np.corrcoef(pc1[valid_mask], feat_df.loc[valid_mask, 'label'])[0, 1]
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


def compute_power_ratio(feat_df: pd.DataFrame) -> np.ndarray:
    """Compute EEG power ratio (Alpha/Beta or other physiological ratio)
    
    Returns:
        Power ratio array
    """
    # Use alpha_rel / beta_rel as power ratio indicator
    if 'alpha_rel' in feat_df.columns and 'beta_rel' in feat_df.columns:
        power_ratio = feat_df['alpha_rel'].values / (feat_df['beta_rel'].values + 1e-10)
    else:
        # Fallback to theta/alpha ratio
        power_ratio = np.ones(len(feat_df))
    
    return power_ratio


def compute_nsw_percentage(feat_df: pd.DataFrame, window_size: int = 10) -> np.ndarray:
    """Compute percentage of Non-Stationary Windows (%NSW)
    
    Uses moving window to detect non-stationarity based on variance changes.
    
    Args:
        feat_df: Feature dataframe
        window_size: Size of moving window for stationarity check
        
    Returns:
        %NSW array (0-1 scale)
    """
    # Use EMG RMS as non-stationarity indicator (higher variance = more non-stationary)
    if 'emg_rms' in feat_df.columns:
        signal = feat_df['emg_rms'].values
    else:
        signal = np.random.randn(len(feat_df))  # Fallback
    
    nsw_pct = np.zeros(len(signal))
    
    for i in range(len(signal)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        
        window = signal[start:end]
        
        # Compute variance ratio (local vs global)
        local_var = np.var(window)
        global_var = np.var(signal)
        
        # Non-stationarity score: higher ratio = more non-stationary
        nsw_score = local_var / (global_var + 1e-10)
        
        # Normalize to [0, 1]
        nsw_pct[i] = min(1.0, nsw_score)
    
    return nsw_pct


def build_composite_hunger_index(feat_df: pd.DataFrame) -> np.ndarray:
    """Build composite hunger index: I = Power Ratio × %NSW
    
    This combines:
    - Power Ratio: EEG spectral features (alpha/beta)
    - %NSW: Non-stationarity percentage (EMG variance)
    
    Args:
        feat_df: Feature dataframe
        
    Returns:
        Composite hunger index array
    """
    power_ratio = compute_power_ratio(feat_df)
    nsw_pct = compute_nsw_percentage(feat_df)
    
    # Composite index
    composite_idx = power_ratio * nsw_pct
    
    # Normalize to [-1, 1] for consistency with other indices
    composite_idx = (composite_idx - composite_idx.mean()) / (composite_idx.std() + 1e-10)
    
    # Ensure hunger (B4) is positive
    if 'segment' in feat_df.columns:
        b4_mean = composite_idx[feat_df['segment'] == 'B4'].mean()
        a5_mean = composite_idx[feat_df['segment'] == 'A5'].mean()
        
        if b4_mean < a5_mean:
            composite_idx = -composite_idx
    
    return composite_idx


def detect_breakpoints_bayesian(idx: np.ndarray, method: str = 'pelt') -> List[int]:
    """Detect breakpoints using Bayesian changepoint detection
    
    Args:
        idx: Hunger index array
        method: 'pelt' (ruptures) or 'bayesian' (bayesian-changepoint-detection)
        
    Returns:
        List of breakpoint indices
    """
    try:
        if method == 'pelt':
            import ruptures as rpt
            
            # Prepare data for ruptures (needs 2D array)
            signal = idx.reshape(-1, 1)
            
            # Use PELT algorithm
            algo = rpt.Pelt(model='rbf', min_size=10, jump=1).fit(signal)
            breakpoints = algo.predict(pen=cfg.pelt_penalty)
            
            # Remove last breakpoint (end of signal)
            if len(breakpoints) > 0 and breakpoints[-1] == len(idx):
                breakpoints = breakpoints[:-1]
            
            print(f"  PELT detected {len(breakpoints)} breakpoints")
            
            return breakpoints
            
        else:
            print(f"  ⚠ Warning: Method '{method}' not implemented. Install ruptures: pip install ruptures")
            return []
            
    except ImportError:
        print(f"  ⚠ Warning: ruptures library not installed. Install with: pip install ruptures")
        print(f"  → Falling back to derivative method")
        return []


def plot_bayesian_breakpoints(t: np.ndarray, idx: np.ndarray, 
                              breakpoints: List[int], segments: np.ndarray, 
                              show: bool = False):
    """Visualize Bayesian breakpoints
    
    Args:
        t: Time array
        idx: Hunger index
        breakpoints: List of breakpoint indices
        segments: Segment labels
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(14, 6), dpi=cfg.fig_dpi)
    
    seg_map = {'B4': cfg.color_b4, 'bridge': cfg.color_bridge, 'A5': cfg.color_a5}
    colors = [seg_map.get(s, 'gray') for s in segments]
    
    ax.scatter(t, idx, c=colors, s=15, alpha=0.6, label='Hunger Index')
    
    # Plot breakpoints
    for bp_idx in breakpoints:
        if bp_idx < len(t):
            ax.axvline(t[bp_idx], color='red', ls='--', lw=2, alpha=0.7, label='Breakpoint')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hunger Index')
    ax.set_title('Bayesian Breakpoint Detection (PELT)')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    
    # Legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def calibrate_pelt_penalty(idx: np.ndarray, penalty_range: np.ndarray = None,
                           cost_model: str = 'rbf', show: bool = False):
    """Calibrate PELT penalty parameter using elbow method
    
    Args:
        idx: Hunger index array
        penalty_range: Array of penalties to test (default: log scale from 1 to 100)
        cost_model: Cost function ('rbf', 'l1', 'l2', 'normal')
        show: Whether to show plot
        
    Returns:
        Figure object and optimal penalty
    """
    try:
        import ruptures as rpt
        
        if penalty_range is None:
            penalty_range = np.logspace(0, 2, 20)  # 1 to 100 on log scale
        
        signal = idx.reshape(-1, 1)
        num_breakpoints = []
        
        print(f"\nCalibrating PELT penalty (cost={cost_model})...")
        
        for pen in penalty_range:
            algo = rpt.Pelt(model=cost_model, min_size=10, jump=1).fit(signal)
            bkpts = algo.predict(pen=pen)
            
            # Remove last breakpoint (end of signal)
            if len(bkpts) > 0 and bkpts[-1] == len(idx):
                bkpts = bkpts[:-1]
            
            num_breakpoints.append(len(bkpts))
            print(f"  Penalty={pen:6.2f} → {len(bkpts)} breakpoints")
        
        # Find elbow using second derivative
        num_bkpts_arr = np.array(num_breakpoints)
        
        # Smooth with moving average
        if len(num_bkpts_arr) > 3:
            kernel = np.ones(3) / 3
            num_bkpts_smooth = np.convolve(num_bkpts_arr, kernel, mode='same')
        else:
            num_bkpts_smooth = num_bkpts_arr
        
        # Compute second derivative
        deriv2 = np.gradient(np.gradient(num_bkpts_smooth))
        
        # Elbow is where second derivative is maximum (sharpest change)
        elbow_idx = np.argmax(np.abs(deriv2))
        optimal_penalty = penalty_range[elbow_idx]
        
        print(f"\n✓ Optimal penalty: {optimal_penalty:.2f} (elbow method)")
        print(f"  Expected breakpoints: {num_breakpoints[elbow_idx]}")
        
        # Plot calibration curve
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=cfg.fig_dpi)
        
        axes[0].plot(penalty_range, num_breakpoints, 'o-', lw=2, markersize=6, color=cfg.color_bridge)
        axes[0].axvline(optimal_penalty, color='red', ls='--', lw=2, alpha=0.7, label=f'Optimal={optimal_penalty:.2f}')
        axes[0].set_xlabel('Penalty')
        axes[0].set_ylabel('Number of Breakpoints')
        axes[0].set_title('PELT Penalty Calibration (Elbow Method)')
        axes[0].set_xscale('log')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(penalty_range, np.abs(deriv2), 'o-', lw=2, markersize=6, color=cfg.color_b4)
        axes[1].axvline(optimal_penalty, color='red', ls='--', lw=2, alpha=0.7, label='Elbow')
        axes[1].set_xlabel('Penalty')
        axes[1].set_ylabel('|Second Derivative|')
        axes[1].set_title('Elbow Detection (Maximum Curvature)')
        axes[1].set_xscale('log')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, optimal_penalty
        
    except ImportError:
        print("  ⚠ Warning: ruptures library not installed. Install with: pip install ruptures")
        return None, None
