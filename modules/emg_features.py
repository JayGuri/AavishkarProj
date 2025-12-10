"""
EMG feature extraction from Channel 1
"""
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from .config import cfg
from .eeg_features import sliding_windows


def compute_rms(x: np.ndarray) -> float:
    """Root mean square"""
    return np.sqrt(np.mean(x**2))


def compute_envelope(x: np.ndarray, fs: int, cutoff: float = 2.0) -> float:
    """Rectify and lowpass filter to get envelope, return mean"""
    rectified = np.abs(x)
    
    sos = signal.butter(2, cutoff, btype='low', fs=fs, output='sos')
    envelope = signal.sosfiltfilt(sos, rectified)
    
    return np.mean(envelope)


def compute_burst_fraction(x: np.ndarray, threshold_factor: float = 1.5) -> float:
    """Fraction of samples exceeding threshold * baseline RMS"""
    baseline_rms = compute_rms(x)
    threshold = threshold_factor * baseline_rms
    
    burst_samples = np.sum(np.abs(x) > threshold)
    
    return burst_samples / len(x)


def extract_emg_features(emg: np.ndarray, counter: np.ndarray,
                         fs: int, win_sec: float, step_sec: float) -> pd.DataFrame:
    """Extract EMG time-domain features"""
    features = []
    
    for start, end, win in sliding_windows(emg, fs, win_sec, step_sec):
        t = (counter[start] + counter[end-1]) / 2
        
        feat = {
            'time': t,
            'emg_rms': compute_rms(win),
            'emg_mrv': np.mean(np.abs(win)),
            'emg_env': compute_envelope(win, fs),
            'emg_burst': compute_burst_fraction(win)
        }
        
        features.append(feat)
    
    return pd.DataFrame(features)


def plot_emg_comparison(b4_emg: np.ndarray, a5_emg: np.ndarray, 
                        b4_counter: np.ndarray, a5_counter: np.ndarray, fs: int):
    """Plot EMG signals with RMS envelope"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi, sharex=True)
    
    n_samp = min(int(10 * fs), len(b4_emg), len(a5_emg))
    
    # B4
    t_b4 = b4_counter[:n_samp] / fs
    axes[0].plot(t_b4, b4_emg[:n_samp], cfg.color_b4, alpha=0.6, lw=0.8)
    
    win_samp = fs
    rms_b4 = []
    t_rms_b4 = []
    for i in range(0, n_samp - win_samp, win_samp//2):
        rms_b4.append(compute_rms(b4_emg[i:i+win_samp]))
        t_rms_b4.append((b4_counter[i] + b4_counter[i+win_samp]) / (2*fs))
    
    axes[0].plot(t_rms_b4, rms_b4, 'k', lw=2, label='RMS Envelope')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('B4 (Hunger) - EMG with RMS Envelope')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # A5
    t_a5 = a5_counter[:n_samp] / fs
    axes[1].plot(t_a5, a5_emg[:n_samp], cfg.color_a5, alpha=0.6, lw=0.8)
    
    rms_a5 = []
    t_rms_a5 = []
    for i in range(0, n_samp - win_samp, win_samp//2):
        rms_a5.append(compute_rms(a5_emg[i:i+win_samp]))
        t_rms_a5.append((a5_counter[i] + a5_counter[i+win_samp]) / (2*fs))
    
    axes[1].plot(t_rms_a5, rms_a5, 'k', lw=2, label='RMS Envelope')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('A5 (Satiety) - EMG with RMS Envelope')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
