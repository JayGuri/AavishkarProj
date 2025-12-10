"""
EEG feature extraction from Channel 3
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict
import matplotlib.pyplot as plt

from .config import cfg


def sliding_windows(x: np.ndarray, fs: int, win_sec: float, step_sec: float):
    """Generate sliding windows with overlap"""
    win_samp = int(win_sec * fs)
    step_samp = int(step_sec * fs)
    
    for start in range(0, len(x) - win_samp + 1, step_samp):
        end = start + win_samp
        yield start, end, x[start:end]


def compute_band_powers(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """Compute absolute and relative band powers"""
    theta_idx = (freqs >= cfg.theta[0]) & (freqs <= cfg.theta[1])
    alpha_idx = (freqs >= cfg.alpha[0]) & (freqs <= cfg.alpha[1])
    beta_idx = (freqs >= cfg.beta[0]) & (freqs <= cfg.beta[1])
    gamma_idx = (freqs >= cfg.gamma[0]) & (freqs <= cfg.gamma[1])
    
    theta_pow = np.trapz(psd[theta_idx], freqs[theta_idx])
    alpha_pow = np.trapz(psd[alpha_idx], freqs[alpha_idx])
    beta_pow = np.trapz(psd[beta_idx], freqs[beta_idx])
    gamma_pow = np.trapz(psd[gamma_idx], freqs[gamma_idx])
    
    total_pow = theta_pow + alpha_pow + beta_pow + gamma_pow
    
    return {
        'theta_pow': theta_pow,
        'alpha_pow': alpha_pow,
        'beta_pow': beta_pow,
        'gamma_pow': gamma_pow,
        'theta_rel': theta_pow / total_pow if total_pow > 0 else 0,
        'alpha_rel': alpha_pow / total_pow if total_pow > 0 else 0,
        'beta_rel': beta_pow / total_pow if total_pow > 0 else 0,
        'gamma_rel': gamma_pow / total_pow if total_pow > 0 else 0,
    }


def compute_peak_alpha(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """Find peak alpha frequency and power"""
    alpha_idx = (freqs >= cfg.alpha[0]) & (freqs <= cfg.alpha[1])
    alpha_freqs = freqs[alpha_idx]
    alpha_psd = psd[alpha_idx]
    
    if len(alpha_psd) == 0:
        return {'alpha_pk_freq': 10.0, 'alpha_pk_pow': 0.0}
    
    peak_idx = np.argmax(alpha_psd)
    
    return {
        'alpha_pk_freq': alpha_freqs[peak_idx],
        'alpha_pk_pow': alpha_psd[peak_idx]
    }


def compute_spectral_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Compute 1/f spectral slope in log-log space (4-30 Hz)"""
    idx = (freqs >= 4) & (freqs <= 30)
    log_f = np.log10(freqs[idx])
    log_p = np.log10(psd[idx] + 1e-10)
    
    if len(log_f) < 2:
        return 0.0
    
    slope = np.polyfit(log_f, log_p, 1)[0]
    
    return slope


def extract_eeg_features(eeg: np.ndarray, counter: np.ndarray,
                         fs: int, win_sec: float, step_sec: float) -> pd.DataFrame:
    """Extract comprehensive EEG spectral features"""
    features = []
    
    for start, end, win in sliding_windows(eeg, fs, win_sec, step_sec):
        t = (counter[start] + counter[end-1]) / 2
        
        freqs, psd = signal.welch(win, fs=fs, 
                                  nperseg=min(len(win), fs*2),
                                  noverlap=fs*1,
                                  scaling='density')
        
        powers = compute_band_powers(freqs, psd)
        peak_alpha = compute_peak_alpha(freqs, psd)
        spec_slope = compute_spectral_slope(freqs, psd)
        
        alpha_pow = powers['alpha_pow']
        beta_pow = powers['beta_pow']
        theta_pow = powers['theta_pow']
        
        a_b_ratio = alpha_pow / (beta_pow + 1e-10)
        a_b_frac = alpha_pow / (alpha_pow + beta_pow + 1e-10)
        beta_frac = beta_pow / (theta_pow + alpha_pow + beta_pow + 1e-10)
        
        feat = {
            'time': t,
            **powers,
            **peak_alpha,
            'spec_slope': spec_slope,
            'a_b_ratio': a_b_ratio,
            'a_b_frac': a_b_frac,
            'beta_frac': beta_frac
        }
        
        features.append(feat)
    
    return pd.DataFrame(features)


def plot_psd_comparison(b4_eeg: np.ndarray, a5_eeg: np.ndarray, fs: int):
    """Plot averaged PSD for B4 vs A5 with band highlights"""
    f_b4, p_b4 = signal.welch(b4_eeg, fs=fs, nperseg=fs*2, noverlap=fs*1)
    f_a5, p_a5 = signal.welch(a5_eeg, fs=fs, nperseg=fs*2, noverlap=fs*1)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=cfg.fig_dpi)
    
    ax.semilogy(f_b4, p_b4, cfg.color_b4, lw=2, label='B4 (Hunger)', alpha=0.8)
    ax.semilogy(f_a5, p_a5, cfg.color_a5, lw=2, label='A5 (Satiety)', alpha=0.8)
    
    bands = [
        ('Theta', cfg.theta, '#F39C12'),
        ('Alpha', cfg.alpha, '#27AE60'),
        ('Beta', cfg.beta, '#8E44AD'),
        ('Gamma', cfg.gamma, '#E67E22')
    ]
    
    for name, (low, high), color in bands:
        ax.axvspan(low, high, alpha=0.1, color=color, label=f'{name} ({low}-{high} Hz)')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('EEG Power Spectral Density: Hunger (B4) vs Satiety (A5)')
    ax.set_xlim([0, 40])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
