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


def extract_all_features(combined: pd.DataFrame, fs: int,
                         win_sec: float, step_sec: float) -> pd.DataFrame:
    """Extract and merge EEG + EMG features with segment labels"""
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
    
    return feat_df


def standardize_features(feat_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features (excluding time, segment, label)"""
    feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    
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
