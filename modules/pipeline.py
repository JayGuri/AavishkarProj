"""
Complete processing pipeline
"""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split

from .config import cfg
from .preprocessing import load_signal, preprocess_signals, plot_filter_verification
from .bridge import combine_with_bridge, plot_bridge_visualization
from .feature_extraction import extract_all_features, plot_feature_heatmap, plot_features_timeline
from .eeg_features import plot_psd_comparison
from .emg_features import plot_emg_comparison
from .hunger_index import (build_hunger_index_weighted, build_hunger_index_pca,
                           plot_hunger_indices, fit_polynomial, compute_derivatives,
                           plot_polynomial_fit, detect_breakpoints_derivative, plot_breakpoints)
from .classification import (build_feature_matrix_labels, train_models,
                            plot_model_performance, plot_feature_importance,
                            predict_hunger_proba, plot_prediction_timeline)


def process_subject(name: str, b4_path: str, a5_path: str) -> Dict:
    """
    Complete processing pipeline for one subject
    
    Returns dict with all results
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING SUBJECT: {name}")
    print(f"{'='*70}\n")
    
    results = {'name': name}
    
    print("1. Loading data...")
    b4_raw = load_signal(b4_path)
    a5_raw = load_signal(a5_path)
    
    # FIX: Overwrite the broken Counter with a continuous index
    # The raw Counter is a hardware counter that loops 0-255, causing time calculation issues
    b4_raw['Counter'] = np.arange(len(b4_raw))
    a5_raw['Counter'] = np.arange(len(a5_raw))
    
    print(f"   B4: {len(b4_raw)} samples ({len(b4_raw)/cfg.fs:.1f} sec)")
    print(f"   A5: {len(a5_raw)} samples ({len(a5_raw)/cfg.fs:.1f} sec)")
    
    print("2. Filtering signals...")
    b4_filt = preprocess_signals(b4_raw, cfg.fs)
    a5_filt = preprocess_signals(a5_raw, cfg.fs)
    
    plot_filter_verification(b4_raw, b4_filt, cfg.fs)
    
    print("3. Generating synthetic bridge...")
    combined = combine_with_bridge(b4_filt, a5_filt, cfg.fs, 
                                   cfg.bridge_sec, cfg.bridge_noise_scale)
    print(f"   Combined: {len(combined)} samples ({len(combined)/cfg.fs:.1f} sec)")
    
    plot_bridge_visualization(combined, cfg.fs)
    
    print("4. Extracting EEG and EMG features...")
    feat_df = extract_all_features(combined, cfg.fs, cfg.win_sec, cfg.step_sec)
    print(f"   Features: {len(feat_df)} windows")
    
    b4_eeg = b4_filt['Channel3'].values
    a5_eeg = a5_filt['Channel3'].values
    plot_psd_comparison(b4_eeg, a5_eeg, cfg.fs)
    
    b4_emg = b4_filt['Channel1'].values
    a5_emg = a5_filt['Channel1'].values
    plot_emg_comparison(b4_emg, a5_emg, 
                       b4_filt['Counter'].values, 
                       a5_filt['Counter'].values, cfg.fs)
    
    plot_feature_heatmap(feat_df)
    plot_features_timeline(feat_df)
    
    results['feat_df'] = feat_df
    
    print("5. Building hunger indices...")
    idx_weighted = build_hunger_index_weighted(feat_df)
    idx_pca = build_hunger_index_pca(feat_df)
    
    plot_hunger_indices(feat_df, idx_weighted, idx_pca)
    
    results['idx_weighted'] = idx_weighted
    results['idx_pca'] = idx_pca
    
    print("6. Fitting polynomial...")
    t_idx = feat_df['time'].values / cfg.fs
    coeffs, fitted, poly = fit_polynomial(t_idx, idx_weighted, deg=5)
    deriv1, deriv2 = compute_derivatives(t_idx, poly)
    
    plot_polynomial_fit(t_idx, idx_weighted, fitted, deriv1, feat_df['segment'].values)
    
    results['poly_coeffs'] = coeffs
    results['poly_fitted'] = fitted
    results['deriv1'] = deriv1
    
    print("7. Detecting breakpoints...")
    bkpts_deriv = detect_breakpoints_derivative(t_idx, deriv1, deriv2, 
                                                threshold_factor=1.5, 
                                                smooth_window=30, 
                                                edge_margin=0.1)
    
    print(f"   Derivative method: {len(bkpts_deriv)} breakpoints at {bkpts_deriv}")
    
    plot_breakpoints(t_idx, idx_weighted, fitted, deriv1, 
                    bkpts_deriv, feat_df['segment'].values)
    
    results['bkpts_deriv'] = bkpts_deriv
    
    print("8. Training classification models (time-series split)...")
    from .classification import time_series_split
    
    # Build X, y with reduced features
    X, y = build_feature_matrix_labels(feat_df, use_reduced_features=True)
    
    # Use time-series split instead of random split
    feat_df_filtered = feat_df[feat_df['segment'].isin(['B4', 'A5'])].copy()
    X_train, X_test, y_train, y_test = time_series_split(X, y, feat_df_filtered, test_ratio=cfg.test_size)
    
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    model_results = train_models(X_train, y_train, X_test, y_test)
    
    plot_model_performance(y_test, model_results)
    
    feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    plot_feature_importance(model_results['rf']['model'], feat_cols, top_n=15)
    
    results['models'] = model_results
    results['X_test'] = X_test
    results['y_test'] = y_test
    
    print("9. Generating prediction timeline...")
    proba_lr = predict_hunger_proba(model_results['lr']['model'], feat_df)
    proba_rf = predict_hunger_proba(model_results['rf']['model'], feat_df)
    
    plot_prediction_timeline(feat_df, proba_lr, idx_weighted, "Logistic Regression")
    plot_prediction_timeline(feat_df, proba_rf, idx_weighted, "Random Forest")
    
    results['proba_lr'] = proba_lr
    results['proba_rf'] = proba_rf
    
    print(f"\nProcessing complete for {name}\n")
    
    return results


def generate_summary_report(results: Dict):
    """Generate summary statistics and interpretation"""
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    name = results['name']
    print(f"\nSubject: {name}")
    
    feat_df = results['feat_df']
    n_b4 = len(feat_df[feat_df['segment'] == 'B4'])
    n_bridge = len(feat_df[feat_df['segment'] == 'bridge'])
    n_a5 = len(feat_df[feat_df['segment'] == 'A5'])
    
    print(f"\nDataset:")
    print(f"  B4 windows:     {n_b4}")
    print(f"  Bridge windows: {n_bridge}")
    print(f"  A5 windows:     {n_a5}")
    print(f"  Total windows:  {len(feat_df)}")
    
    idx = results['idx_weighted']
    print(f"\nHunger Index (Weighted):")
    print(f"  Mean: {np.mean(idx):.3f}")
    print(f"  Std:  {np.std(idx):.3f}")
    print(f"  B4 mean:  {np.mean(idx[feat_df['segment'] == 'B4']):.3f}")
    print(f"  A5 mean:  {np.mean(idx[feat_df['segment'] == 'A5']):.3f}")
    
    bkpts = results['bkpts_deriv']
    print(f"\nDetected Breakpoints (Derivative Method):")
    if len(bkpts) > 0:
        for i, bp in enumerate(bkpts):
            print(f"  Breakpoint {i+1}: {bp:.2f} sec")
    else:
        print("  No breakpoints detected")
    
    print(f"\nClassification Performance:")
    for model_name, model_res in results['models'].items():
        print(f"  {model_name.upper()}:")
        print(f"    Accuracy:  {model_res['accuracy']:.3f}")
        print(f"    ROC-AUC:   {model_res['roc_auc']:.3f}")
        print(f"    F1-Score:  {model_res['f1']:.3f}")
    
    print("\n" + "="*70 + "\n")


def export_results(results: Dict, output_dir: str = "output"):
    """Export features and predictions to CSV"""
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    name = results['name']
    
    feat_df = results['feat_df'].copy()
    feat_df['idx_weighted'] = results['idx_weighted']
    feat_df['idx_pca'] = results['idx_pca']
    feat_df['poly_fitted'] = results['poly_fitted']
    feat_df['proba_lr'] = results['proba_lr']
    feat_df['proba_rf'] = results['proba_rf']
    
    csv_path = os.path.join(output_dir, f"{name}_features.csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"Features exported to: {csv_path}")
    
    summary = {
        'subject': name,
        'sampling_rate': cfg.fs,
        'bridge_duration': cfg.bridge_sec,
        'window_size': cfg.win_sec,
        'step_size': cfg.step_sec,
        'n_windows': len(feat_df),
        'breakpoints': results['bkpts_deriv'],
        'model_performance': {
            name: {k: v for k, v in res.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
            for name, res in results['models'].items()
        }
    }
    
    json_path = os.path.join(output_dir, f"{name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary exported to: {json_path}")
