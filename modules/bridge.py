"""
Synthetic bridge generation between B4 and A5 segments
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from .config import cfg


def make_bridge_segment(b4_tail: np.ndarray, a5_head: np.ndarray, 
                        fs: int, bridge_sec: float, 
                        noise_scale: float = 0.08) -> np.ndarray:
    """
    Create smooth bridge between B4 tail and A5 head using cubic interpolation + colored noise
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
                        noise_scale: float = 0.08) -> pd.DataFrame:
    """
    Concatenate B4, bridge, and A5 with continuous Counter
    """
    k = min(500, len(b4)//4, len(a5)//4)
    
    bridge_ch1 = make_bridge_segment(b4['Channel1'].iloc[-k:].values,
                                     a5['Channel1'].iloc[:k].values,
                                     fs, bridge_sec, noise_scale)
    
    bridge_ch3 = make_bridge_segment(b4['Channel3'].iloc[-k:].values,
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
