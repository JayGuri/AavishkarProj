"""
Unified feature extraction and visualization
"""
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .config import cfg
from .eeg_features import extract_eeg_features
from .emg_features import extract_emg_features


def apply_squelch(feat_df: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """Apply artifact gating (squelch) to suppress contaminated windows
    
    Args:
        feat_df: Feature dataframe
        combined: Combined signal dataframe with artifact_energy column
        
    Returns:
        Feature dataframe with squelch_flag and confidence columns
    """
    if 'artifact_energy' not in combined.columns or not cfg.squelch_enabled:
        # No squelch needed
        feat_df['squelch_flag'] = False
        feat_df['confidence'] = 1.0
        return feat_df
    
    # Compute median artifact energy
    median_energy = np.median(combined['artifact_energy'].values)
    threshold = cfg.squelch_threshold * median_energy
    
    # Flag windows with high artifact energy
    def check_artifact(t_sec):
        # Convert time to sample index
        t_samples = int(t_sec * cfg.fs)
        idx = np.argmin(np.abs(combined['Counter'].values - t_samples))
        
        # Get artifact energy at this time point
        energy = combined.iloc[idx]['artifact_energy']
        
        return energy > threshold, energy
    
    feat_df['squelch_flag'] = False
    feat_df['artifact_energy'] = 0.0
    feat_df['confidence'] = 1.0
    
    for i, row in feat_df.iterrows():
        is_artifact, energy = check_artifact(row['time'])
        feat_df.at[i, 'squelch_flag'] = is_artifact
        feat_df.at[i, 'artifact_energy'] = energy
        
        # Confidence inversely proportional to artifact energy
        if is_artifact:
            feat_df.at[i, 'confidence'] = max(0.1, threshold / (energy + 1e-10))
        else:
            feat_df.at[i, 'confidence'] = 1.0
    
    n_flagged = feat_df['squelch_flag'].sum()
    print(f"  Squelch: Flagged {n_flagged}/{len(feat_df)} windows ({n_flagged/len(feat_df)*100:.1f}%)")
    
    return feat_df


def extract_all_features(combined: pd.DataFrame, fs: int,
                         win_sec: float, step_sec: float) -> pd.DataFrame:
    """Extract and merge EEG + EMG features with segment labels
    
    Args:
        combined: Combined signal dataframe
        fs: Sampling frequency
        win_sec: Window length in seconds
        step_sec: Step size in seconds
        
    Returns:
        Feature dataframe with squelch flags if enabled
    """
    eeg_feat = extract_eeg_features(combined['Channel3'].values,
                                    combined['Counter'].values,
                                    fs, win_sec, step_sec)
    
    emg_feat = extract_emg_features(combined['Channel1'].values,
                                    combined['Counter'].values,
                                    fs, win_sec, step_sec)
    
    feat_df = pd.merge(eeg_feat, emg_feat, on='time', how='inner')
    
    def get_segment(t):
        idx = np.argmin(np.abs(combined['Counter'].values - t))
        return combined.iloc[idx]['segment']
    
    feat_df['segment'] = feat_df['time'].apply(get_segment)
    
    label_map = {'B4': 1, 'A5': 0, 'bridge': 0.5}
    # Normalize time to seconds; merged 'time' is in samples
    feat_df['time'] = feat_df['time'] / fs
    feat_df['label'] = feat_df['segment'].map(label_map)
    
    # Apply squelch if enabled
    if cfg.squelch_enabled:
        feat_df = apply_squelch(feat_df, combined)
    
    return feat_df


def standardize_features(feat_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features (excluding time, segment, label, soft_label)"""
    feat_cols = [c for c in feat_df.columns 
                 if c not in ['time', 'segment', 'label', 'soft_label', 'label_type']]
    
    scaler = StandardScaler()
    feat_df_std = feat_df.copy()
    feat_df_std[feat_cols] = scaler.fit_transform(feat_df[feat_cols])
    
    return feat_df_std, scaler


def plot_feature_heatmap(feat_df: pd.DataFrame, show: bool = False):
    """Heatmap of averaged features per segment"""
    feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    
    avg_by_seg = feat_df.groupby('segment')[feat_cols].mean()
    avg_by_seg = avg_by_seg.reindex(['B4', 'bridge', 'A5'])
    
    avg_by_seg_z = (avg_by_seg - avg_by_seg.mean()) / (avg_by_seg.std() + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 12), dpi=cfg.fig_dpi)
    
    sns.heatmap(avg_by_seg_z.T, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Z-score'},
                linewidths=0.5, ax=ax, vmin=-2, vmax=2)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Heatmap: Averaged per Segment (Z-scored)')
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_features_timeline(feat_df: pd.DataFrame, show: bool = False):
    """Multi-panel plot of key features over time"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), dpi=cfg.fig_dpi, sharex=True)
    
    # 'time' is already in seconds
    t = feat_df['time'].values
    
    colors = feat_df['segment'].map({'B4': cfg.color_b4, 
                                     'bridge': cfg.color_bridge, 
                                     'A5': cfg.color_a5})
    
    axes[0].scatter(t, feat_df['alpha_rel'], c=colors, s=10, alpha=0.6)
    axes[0].set_ylabel('Alpha Rel.')
    axes[0].set_title('Alpha Relative Power')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(t, feat_df['beta_rel'], c=colors, s=10, alpha=0.6)
    axes[1].set_ylabel('Beta Rel.')
    axes[1].set_title('Beta Relative Power')
    axes[1].grid(alpha=0.3)
    
    axes[2].scatter(t, feat_df['alpha_pk_freq'], c=colors, s=10, alpha=0.6)
    axes[2].set_ylabel('Freq (Hz)')
    axes[2].set_title('Peak Alpha Frequency')
    axes[2].grid(alpha=0.3)
    
    axes[3].scatter(t, feat_df['emg_rms'], c=colors, s=10, alpha=0.6)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('RMS')
    axes[3].set_title('EMG RMS')
    axes[3].grid(alpha=0.3)
    
    handles = [mpatches.Patch(color=cfg.color_b4, label='B4 (Hunger)'),
               mpatches.Patch(color=cfg.color_bridge, label='Bridge'),
               mpatches.Patch(color=cfg.color_a5, label='A5 (Satiety)')]
    axes[0].legend(handles=handles, loc='upper right', ncol=3)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
