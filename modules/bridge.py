"""
Synthetic bridge generation between B4 and A5 segments
Implements robust sigmoidal interpolation with variance injection
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from .config import cfg


def make_bridge_segment_sigmoidal(b4_tail: np.ndarray, a5_head: np.ndarray, 
                                  fs: int, bridge_sec: float, 
                                  noise_scale: float = 0.15) -> np.ndarray:
    """
    Create robust bridge using Sigmoidal interpolation with variance injection.
    
    This creates a biologically plausible transition:
    1. Sigmoid curve (S-shaped) mimics natural saturation behavior
    2. Variance injection adds realistic "roughness"
    3. Savitzky-Golay smoothing ensures continuous derivatives
    
    Args:
        b4_tail: Last portion of B4 signal
        a5_head: First portion of A5 signal
        fs: Sampling rate (Hz)
        bridge_sec: Bridge duration (seconds)
        noise_scale: Scaling factor for variance injection
        
    Returns:
        Bridge signal with smooth sigmoidal transition
    """
    n_bridge = int(bridge_sec * fs)
    
    # Step 1: Compute boundary statistics from tail/head windows
    b4_mean = b4_tail.mean()
    b4_std = b4_tail.std()
    a5_mean = a5_head.mean()
    a5_std = a5_head.std()
    
    # Step 2: Generate Sigmoid weights (S-curve from 0 to 1)
    # Range -6 to 6 captures the full sigmoid transition
    x = np.linspace(-6, 6, n_bridge)
    sigmoid_weights = 1 / (1 + np.exp(-x))
    
    # Step 3: Interpolate means (the "path")
    bridge_mean = (1 - sigmoid_weights) * b4_mean + sigmoid_weights * a5_mean
    
    # Step 4: Interpolate variance (the "texture")
    # Noise level also transitions smoothly
    bridge_std = (1 - sigmoid_weights) * b4_std + sigmoid_weights * a5_std
    
    # Step 5: Generate synthetic bridge with variance injection
    # Add Brownian-like noise scaled by local variance
    noise = np.random.normal(0, 1, n_bridge) * bridge_std * noise_scale
    bridge_raw = bridge_mean + noise
    
    # Step 6: Apply Savitzky-Golay filter to ensure smooth derivatives
    # This prevents "step function" artifacts at connection points
    if n_bridge >= 7:  # Need at least window_length points
        window_length = min(7, n_bridge if n_bridge % 2 == 1 else n_bridge - 1)
        bridge_smooth = savgol_filter(bridge_raw, window_length=window_length, polyorder=2)
    else:
        bridge_smooth = bridge_raw
    
    return bridge_smooth


def make_bridge_segment(b4_tail: np.ndarray, a5_head: np.ndarray, 
                        fs: int, bridge_sec: float, 
                        noise_scale: float = 0.08) -> np.ndarray:
    """
    Legacy cubic interpolation bridge (kept for backward compatibility)
    Consider using make_bridge_segment_sigmoidal() instead for more robust results.
    """
    n_bridge = int(bridge_sec * fs)
    
    b4_mean, b4_std = b4_tail.mean(), b4_tail.std()
    a5_mean, a5_std = a5_head.mean(), a5_head.std()
    
    k = len(b4_tail)
    x_ctrl = np.array([0, k//3, 2*k//3, n_bridge])
    y_ctrl = np.array([b4_mean, 
                       0.7*b4_mean + 0.3*a5_mean,
                       0.3*b4_mean + 0.7*a5_mean,
                       a5_mean])
    
    cs = CubicSpline(x_ctrl, y_ctrl, bc_type='clamped')
    x_bridge = np.arange(n_bridge)
    bridge_smooth = cs(x_bridge)
    
    white_noise = np.random.randn(n_bridge)
    
    f = np.fft.rfftfreq(n_bridge, 1/fs)
    f[0] = 1
    filt = 1 / np.sqrt(f)
    filt /= filt.max()
    
    noise_fft = np.fft.rfft(white_noise) * filt
    colored_noise = np.fft.irfft(noise_fft, n=n_bridge)
    
    local_std = np.linspace(b4_std, a5_std, n_bridge)
    scaled_noise = colored_noise * local_std * noise_scale
    
    bridge = bridge_smooth + scaled_noise
    
    return bridge


def combine_with_bridge(b4: pd.DataFrame, a5: pd.DataFrame, 
                        fs: int, bridge_sec: float,
                        noise_scale: float = 0.15, 
                        use_sigmoidal: bool = True) -> pd.DataFrame:
    """
    Concatenate B4, bridge, and A5 with continuous Counter
    
    Args:
        b4: B4 segment dataframe
        a5: A5 segment dataframe
        fs: Sampling rate
        bridge_sec: Bridge duration in seconds
        noise_scale: Noise scaling factor
        use_sigmoidal: If True, use robust sigmoidal bridge; if False, use legacy cubic
    """
    k = min(500, len(b4)//4, len(a5)//4)
    
    # Select bridge generation method
    bridge_func = make_bridge_segment_sigmoidal if use_sigmoidal else make_bridge_segment
    
    bridge_ch1 = bridge_func(b4['Channel1'].iloc[-k:].values,
                             a5['Channel1'].iloc[:k].values,
                             fs, bridge_sec, noise_scale)
    
    bridge_ch3 = bridge_func(b4['Channel3'].iloc[-k:].values,
                             a5['Channel3'].iloc[:k].values,
                             fs, bridge_sec, noise_scale)
    
    n_bridge = len(bridge_ch1)
    
    bridge_counter = np.arange(len(b4), len(b4) + n_bridge)
    bridge_df = pd.DataFrame({
        'Counter': bridge_counter,
        'Channel1': bridge_ch1,
        'Channel3': bridge_ch3,
        'segment': 'bridge'
    })
    
    b4_labeled = b4.copy()
    b4_labeled['segment'] = 'B4'
    
    a5_labeled = a5.copy()
    a5_labeled['Counter'] = a5_labeled['Counter'] + len(b4) + n_bridge
    a5_labeled['segment'] = 'A5'
    
    combined = pd.concat([b4_labeled, bridge_df, a5_labeled], ignore_index=True)
    
    return combined


def plot_bridge_visualization(combined: pd.DataFrame, fs: int, show: bool = False):
    """Visualize B4 tail, bridge, and A5 head"""
    b4_end = combined[combined['segment'] == 'B4'].index[-1]
    bridge_end = combined[combined['segment'] == 'bridge'].index[-1]
    
    margin = int(3 * fs)
    start_idx = max(0, b4_end - margin)
    end_idx = min(len(combined), bridge_end + margin)
    
    plot_df = combined.iloc[start_idx:end_idx]
    t = plot_df['Counter'].values / fs
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi)
    
    for ax, ch in zip(axes, ['Channel1', 'Channel3']):
        for seg, color in [('B4', cfg.color_b4), 
                          ('bridge', cfg.color_bridge), 
                          ('A5', cfg.color_a5)]:
            mask = plot_df['segment'] == seg
            ax.plot(t[mask], plot_df[ch].values[mask], 
                   color=color, lw=1.5, label=seg)
        
        bridge_start_t = combined.iloc[b4_end]['Counter'] / fs
        bridge_end_t = combined.iloc[bridge_end]['Counter'] / fs
        ax.axvline(bridge_start_t, color='k', ls='--', alpha=0.5, lw=1)
        ax.axvline(bridge_end_t, color='k', ls='--', alpha=0.5, lw=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ch_name = 'EMG (Ch1)' if ch == 'Channel1' else 'EEG (Ch3)'
        ax.set_title(f'{ch_name}: B4 -> Bridge -> A5 Transition')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
