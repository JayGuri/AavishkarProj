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
    bridge_sec: float = 10.0  # Default bridge duration (seconds) for signal-level bridge
    bridge_noise_scale: float = 0.15  # Noise amplitude (fraction of local std)
    bridge_use_sigmoidal: bool = True  # Use sigmoidal interpolation (recommended)

    # Dynamic feature-bridge parameters (Mahalanobis + Brownian Bridge)
    dyn_base_sec: float = 5.0   # Base bridge duration (seconds)
    dyn_alpha: float = 1.0      # Scaling factor for Mahalanobis distance
    dyn_min_sec: float = 5.0    # Min bridge duration (seconds)
    dyn_max_sec: float = 20.0   # Max bridge duration (seconds)
    dyn_tail_windows: int = 20  # N windows from tails/heads to estimate centroids and variance
    dyn_noise_jitter: float = 0.15  # Jitter scale for Brownian Bridge variance injection
    
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
    
    # Phase 1: Medical Validation
    use_loso: bool = True  # Use Leave-One-Subject-Out cross-validation
    exclusion_zone_min: float = 30.0  # Exclude first 30 minutes of A5 (post-meal lag)
    
    # Phase 2: VMD Signal Processing
    use_vmd: bool = False  # Use Variational Mode Decomposition (set True when vmdpy installed)
    vmd_K: int = 5  # Number of VMD modes
    vmd_alpha: int = 2000  # VMD bandwidth constraint
    vmd_target_freq: float = 0.05  # Target frequency for physiological mode (Hz)
    
    # Phase 2: Artifact Gating (Squelch)
    squelch_enabled: bool = False  # Enable artifact suppression
    squelch_threshold: float = 2.5  # Energy threshold (multiples of median)
    
    # Phase 3: Bayesian Breakpoints
    use_bayesian_breakpoints: bool = False  # Use ruptures PELT instead of polynomial
    pelt_penalty: float = 10.0  # PELT penalty parameter

# Global configuration instance
cfg = Config()
