"""
Configuration parameters for EEG/EMG signal analysis
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """Global configuration parameters"""
    # Sampling
    fs: int = 500  # Sampling rate (Hz)
    
    # Frequency bands (Hz)
    theta: Tuple[float, float] = (4, 8)
    alpha: Tuple[float, float] = (8, 13)
    beta: Tuple[float, float] = (13, 30)
    gamma: Tuple[float, float] = (30, 40)
    
    # Bridge parameters (Robust Sigmoidal Bridge)
    bridge_sec: float = 10.0  # Bridge duration (seconds) - increased for visibility in PCA
    bridge_noise_scale: float = 0.15  # Noise amplitude (fraction of local std)
    bridge_use_sigmoidal: bool = True  # Use sigmoidal interpolation (recommended)
    
    # Feature extraction
    win_sec: float = 2.0  # Window length (seconds)
    step_sec: float = 1.0  # Window step (50% overlap)
    
    # Filters
    eeg_filt_low: float = 1.0  # EEG highpass cutoff (Hz)
    eeg_filt_high: float = 40.0  # EEG lowpass cutoff (Hz)
    emg_filt_low: float = 20.0  # EMG highpass cutoff (Hz)
    emg_filt_high: float = 245.0  # EMG lowpass cutoff (Hz)
    filt_order: int = 4  # Filter order
    
    # Visualization
    fig_dpi: int = 100
    color_b4: str = '#E74C3C'  # Red for hunger (B4)
    color_a5: str = '#3498DB'  # Blue for satiety (A5)
    color_bridge: str = '#95A5A6'  # Gray for bridge
    
    # Model
    test_size: float = 0.3
    random_state: int = 42

# Global configuration instance
cfg = Config()
