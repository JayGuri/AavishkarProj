"""
Data loading and signal preprocessing functions
"""
import pandas as pd
import numpy as np
from scipy import signal
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .config import cfg


def load_signal(path: str) -> pd.DataFrame:
    """Load CSV file and keep Counter, Channel1 (EMG), Channel3 (EEG)"""
    df = pd.read_csv(path)
    if 'Channel4' in df.columns:
        df = df[['Counter', 'Channel1', 'Channel3']]
    return df


def design_filters(fs: int) -> Tuple:
    """Design Butterworth bandpass filters for EEG and EMG"""
    sos_eeg = signal.butter(cfg.filt_order, [cfg.eeg_filt_low, cfg.eeg_filt_high], 
                            btype='bandpass', fs=fs, output='sos')
    
    sos_emg = signal.butter(cfg.filt_order, [cfg.emg_filt_low, cfg.emg_filt_high], 
                            btype='bandpass', fs=fs, output='sos')
    
    return sos_eeg, sos_emg


def preprocess_signals(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """Apply EEG and EMG filters"""
    sos_eeg, sos_emg = design_filters(fs)
    
    df_filt = df.copy()
    df_filt['Channel1'] = signal.sosfiltfilt(sos_emg, df['Channel1'].values)
    df_filt['Channel3'] = signal.sosfiltfilt(sos_eeg, df['Channel3'].values)
    
    return df_filt


def plot_filter_verification(df_raw: pd.DataFrame, df_filt: pd.DataFrame, 
                             fs: int, n_sec: float = 5.0, show: bool = False):
    """Plot raw vs filtered signals and filter frequency response"""
    n_samp = int(n_sec * fs)
    t = np.arange(n_samp) / fs
    
    fig = plt.figure(figsize=(14, 8), dpi=cfg.fig_dpi)
    gs = GridSpec(2, 2, figure=fig)
    
    # EMG comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df_raw['Channel1'].iloc[:n_samp], 'k', alpha=0.5, label='Raw EMG', lw=0.8)
    ax1.plot(t, df_filt['Channel1'].iloc[:n_samp], cfg.color_b4, label='Filtered EMG', lw=1.2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('EMG: Raw vs Filtered (20-250 Hz)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # EEG comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df_raw['Channel3'].iloc[:n_samp], 'k', alpha=0.5, label='Raw EEG', lw=0.8)
    ax2.plot(t, df_filt['Channel3'].iloc[:n_samp], cfg.color_a5, label='Filtered EEG', lw=1.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('EEG: Raw vs Filtered (1-40 Hz)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Filter frequency responses
    sos_eeg, sos_emg = design_filters(fs)
    
    ax3 = fig.add_subplot(gs[1, 0])
    w_emg, h_emg = signal.sosfreqz(sos_emg, worN=2048, fs=fs)
    ax3.plot(w_emg, 20 * np.log10(abs(h_emg)), cfg.color_b4, lw=2)
    ax3.axvline(cfg.emg_filt_low, color='gray', ls='--', alpha=0.5)
    ax3.axvline(cfg.emg_filt_high, color='gray', ls='--', alpha=0.5)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Gain (dB)')
    ax3.set_title(f'EMG Filter Response ({cfg.emg_filt_low}-{cfg.emg_filt_high} Hz)')
    ax3.grid(alpha=0.3)
    ax3.set_xlim([0, 300])
    
    ax4 = fig.add_subplot(gs[1, 1])
    w_eeg, h_eeg = signal.sosfreqz(sos_eeg, worN=2048, fs=fs)
    ax4.plot(w_eeg, 20 * np.log10(abs(h_eeg)), cfg.color_a5, lw=2)
    ax4.axvline(cfg.eeg_filt_low, color='gray', ls='--', alpha=0.5)
    ax4.axvline(cfg.eeg_filt_high, color='gray', ls='--', alpha=0.5)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Gain (dB)')
    ax4.set_title(f'EEG Filter Response ({cfg.eeg_filt_low}-{cfg.eeg_filt_high} Hz)')
    ax4.grid(alpha=0.3)
    ax4.set_xlim([0, 50])
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
