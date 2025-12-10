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


def load_signal(path: str, apply_exclusion: bool = False) -> pd.DataFrame:
    """Load CSV file and keep Counter, Channel1 (EMG), Channel3 (EEG)
    
    Args:
        path: Path to CSV file
        apply_exclusion: If True and file is A5, exclude first 30 minutes (lag phase)
        
    Returns:
        DataFrame with Counter, Channel1 (EMG), Channel3 (EEG)
    """
    df = pd.read_csv(path)
    if 'Channel4' in df.columns:
        df = df[['Counter', 'Channel1', 'Channel3']]
    
    # Apply exclusion zone for A5 (satiety) files to remove post-meal lag
    if apply_exclusion and cfg.exclusion_zone_min > 0:
        # Check if this is an A5 file
        if 'A5' in path or 'a5' in path.lower():
            # Calculate samples to exclude (30 minutes by default)
            samples_to_exclude = int(cfg.exclusion_zone_min * 60 * cfg.fs)
            
            if len(df) > samples_to_exclude:
                df = df.iloc[samples_to_exclude:].reset_index(drop=True)
                print(f"  ✓ Excluded first {cfg.exclusion_zone_min} min ({samples_to_exclude} samples) from A5 data")
            else:
                print(f"  ⚠ Warning: A5 file too short ({len(df)} samples), skipping exclusion")
    
    return df


def design_filters(fs: int) -> Tuple:
    """Design Butterworth bandpass filters for EEG and EMG"""
    sos_eeg = signal.butter(cfg.filt_order, [cfg.eeg_filt_low, cfg.eeg_filt_high], 
                            btype='bandpass', fs=fs, output='sos')
    
    sos_emg = signal.butter(cfg.filt_order, [cfg.emg_filt_low, cfg.emg_filt_high], 
                            btype='bandpass', fs=fs, output='sos')
    
    return sos_eeg, sos_emg


def apply_vmd(signal_data: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Variational Mode Decomposition
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        
    Returns:
        modes: VMD modes (K x N)
        selected_mode: Mode closest to target frequency (0.05 Hz)
    """
    try:
        from vmdpy import VMD
        
        # VMD Parameters
        alpha = cfg.vmd_alpha  # Bandwidth constraint (2000)
        tau = 0  # Noise tolerance (no strict fidelity)
        K = cfg.vmd_K  # Number of modes (5)
        DC = 0  # No DC component
        init = 1  # Initialize omegas uniformly
        tol = 1e-7  # Convergence tolerance
        
        # Run VMD
        modes, u_hat, omega = VMD(signal_data, alpha, tau, K, DC, init, tol)
        
        # Find mode closest to target frequency (0.05 Hz)
        target_freq = cfg.vmd_target_freq
        mode_freqs = omega[-1] * fs / (2 * np.pi)  # Convert to Hz
        
        freq_diffs = np.abs(mode_freqs - target_freq)
        selected_idx = np.argmin(freq_diffs)
        selected_mode = modes[selected_idx, :]
        
        print(f"    VMD: Selected mode {selected_idx} (freq={mode_freqs[selected_idx]:.3f} Hz, target={target_freq} Hz)")
        
        return modes, selected_mode
        
    except ImportError:
        print("  ⚠ Warning: vmdpy not installed. Install with: pip install vmdpy")
        print("  → Falling back to bandpass filtering")
        return None, None


def compute_artifact_energy(modes: np.ndarray, fs: int) -> float:
    """Compute high-frequency artifact energy from VMD modes
    
    Args:
        modes: VMD modes (K x N)
        fs: Sampling frequency
        
    Returns:
        Artifact energy (normalized)
    """
    if modes is None:
        return 0.0
    
    # High-frequency modes (last 2-3 modes typically contain EMG/motion artifacts)
    hf_modes = modes[-2:, :]  # Last 2 modes
    
    # Compute RMS energy
    hf_energy = np.sqrt(np.mean(hf_modes**2))
    
    return hf_energy


def preprocess_signals(df: pd.DataFrame, fs: int, use_vmd: bool = None) -> pd.DataFrame:
    """Apply EEG and EMG filters or VMD decomposition
    
    Args:
        df: Input dataframe
        fs: Sampling frequency
        use_vmd: Override config.use_vmd if provided
        
    Returns:
        Filtered dataframe with optional VMD metadata
    """
    if use_vmd is None:
        use_vmd = cfg.use_vmd
    
    df_filt = df.copy()
    
    if use_vmd:
        print("  Using VMD decomposition...")
        
        # Apply VMD to EEG (Channel3)
        eeg_modes, eeg_selected = apply_vmd(df['Channel3'].values, fs)
        
        if eeg_selected is not None:
            df_filt['Channel3'] = eeg_selected
            
            # Store artifact energy for squelch
            if cfg.squelch_enabled:
                artifact_energy = compute_artifact_energy(eeg_modes, fs)
                df_filt['artifact_energy'] = artifact_energy
        else:
            # Fallback to bandpass
            sos_eeg, sos_emg = design_filters(fs)
            df_filt['Channel3'] = signal.sosfiltfilt(sos_eeg, df['Channel3'].values)
        
        # Apply bandpass to EMG (Channel1) - VMD not needed for EMG
        sos_eeg, sos_emg = design_filters(fs)
        df_filt['Channel1'] = signal.sosfiltfilt(sos_emg, df['Channel1'].values)
    else:
        # Standard bandpass filtering
        sos_eeg, sos_emg = design_filters(fs)
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


def plot_vmd_modes(signal_data: np.ndarray, modes: np.ndarray, fs: int, 
                   title: str = "VMD Decomposition", show: bool = False):
    """Visualize VMD modes and their frequency content
    
    Args:
        signal_data: Original signal
        modes: VMD modes (K x N)
        fs: Sampling frequency
        title: Plot title
        show: Whether to show plot
        
    Returns:
        Figure object
    """
    K = modes.shape[0]
    n_samples = min(5000, modes.shape[1])  # Show first 5 seconds
    t = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(K + 1, 2, figsize=(16, 3*(K+1)), dpi=cfg.fig_dpi)
    
    # Original signal
    axes[0, 0].plot(t, signal_data[:n_samples], 'k', lw=0.8)
    axes[0, 0].set_ylabel('Original')
    axes[0, 0].set_title(f'{title} - Time Domain')
    axes[0, 0].grid(alpha=0.3)
    
    # Original signal spectrum
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    spectrum = np.abs(np.fft.rfft(signal_data[:n_samples]))
    axes[0, 1].plot(freqs, spectrum, 'k', lw=0.8)
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('Frequency Domain')
    axes[0, 1].set_xlim([0, 50])
    axes[0, 1].grid(alpha=0.3)
    
    # Plot each mode
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    
    for k in range(K):
        # Time domain
        axes[k+1, 0].plot(t, modes[k, :n_samples], color=colors[k], lw=0.8)
        axes[k+1, 0].set_ylabel(f'Mode {k+1}')
        axes[k+1, 0].grid(alpha=0.3)
        
        # Frequency domain
        mode_spectrum = np.abs(np.fft.rfft(modes[k, :n_samples]))
        axes[k+1, 1].plot(freqs, mode_spectrum, color=colors[k], lw=0.8)
        axes[k+1, 1].set_xlim([0, 50])
        axes[k+1, 1].grid(alpha=0.3)
        
        # Find dominant frequency
        peak_idx = np.argmax(mode_spectrum[:len(freqs)//2])  # Only first half
        peak_freq = freqs[peak_idx]
        axes[k+1, 1].set_ylabel(f'{peak_freq:.2f} Hz')
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def analyze_vmd_subject(subject_name: str, signal_data: np.ndarray, fs: int, 
                        signal_type: str = 'EEG') -> Tuple[np.ndarray, np.ndarray, dict]:
    """Comprehensive VMD analysis for a subject
    
    Args:
        subject_name: Name of subject
        signal_data: Input signal (EEG or EMG)
        fs: Sampling frequency
        signal_type: 'EEG' or 'EMG'
        
    Returns:
        modes: VMD modes (K x N)
        selected_mode: Physiological mode
        stats: Dictionary of analysis statistics
    """
    print(f"\n{'='*70}")
    print(f"VMD ANALYSIS: {subject_name} - {signal_type}")
    print('='*70)
    
    # Apply VMD
    modes, selected_mode = apply_vmd(signal_data, fs)
    
    if modes is None:
        return None, None, {}
    
    # Compute mode statistics
    stats = {
        'num_modes': modes.shape[0],
        'signal_length': modes.shape[1],
        'mode_energies': [],
        'mode_frequencies': [],
        'selected_mode_idx': None
    }
    
    # Analyze each mode
    for k in range(modes.shape[0]):
        # Energy
        energy = np.sqrt(np.mean(modes[k, :]**2))
        stats['mode_energies'].append(energy)
        
        # Dominant frequency
        spectrum = np.abs(np.fft.rfft(modes[k, :]))
        freqs = np.fft.rfftfreq(modes.shape[1], 1/fs)
        peak_idx = np.argmax(spectrum[:len(freqs)//2])
        peak_freq = freqs[peak_idx]
        stats['mode_frequencies'].append(peak_freq)
        
        print(f"  Mode {k+1}: Energy={energy:.4f}, Peak Freq={peak_freq:.3f} Hz")
        
        # Check if this is the selected mode
        if np.allclose(modes[k, :], selected_mode):
            stats['selected_mode_idx'] = k
    
    # Artifact energy
    artifact_energy = compute_artifact_energy(modes, fs)
    stats['artifact_energy'] = artifact_energy
    
    print(f"\n  Selected Mode: {stats['selected_mode_idx'] + 1}")
    print(f"  Artifact Energy: {artifact_energy:.4f}")
    
    return modes, selected_mode, stats
