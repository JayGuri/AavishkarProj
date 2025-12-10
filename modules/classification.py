"""
Classification models and prediction
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .config import cfg


def time_series_split(X: np.ndarray, y: np.ndarray, feat_df: pd.DataFrame, 
                     test_ratio: float = 0.2) -> Tuple:
    """Split data by time rather than randomly to prevent overfitting
    
    Train on first (1-test_ratio) of B4 and A5, test on the rest.
    This forces the model to predict future state, not fill gaps.
    
    Args:
        X: Feature matrix
        y: Labels
        feat_df: Feature dataframe (filtered to B4 and A5 only)
        test_ratio: Ratio of test data
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Split B4 and A5 separately by time
    b4_mask = feat_df['segment'] == 'B4'
    a5_mask = feat_df['segment'] == 'A5'
    
    b4_indices = np.where(b4_mask)[0]
    a5_indices = np.where(a5_mask)[0]
    
    # Get split points
    b4_split = int(len(b4_indices) * (1 - test_ratio))
    a5_split = int(len(a5_indices) * (1 - test_ratio))
    
    # Create train/test masks
    train_indices = np.concatenate([b4_indices[:b4_split], a5_indices[:a5_split]])
    test_indices = np.concatenate([b4_indices[b4_split:], a5_indices[a5_split:]])
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def build_feature_matrix_labels(feat_df: pd.DataFrame, 
                               use_reduced_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Build X, y for classification (exclude bridge samples)
    
    Args:
        feat_df: Feature dataframe
        use_reduced_features: If True, use only key features to prevent overfitting
    """
    train_df = feat_df[feat_df['segment'].isin(['B4', 'A5'])].copy()
    
    if use_reduced_features:
        # Use only key features that reliably distinguish hunger/satiety
        key_features = ['alpha_rel', 'beta_rel', 'emg_rms']
        feat_cols = [c for c in key_features if c in train_df.columns]
        if len(feat_cols) == 0:
            # Fallback to all features if key features not available
            feat_cols = [c for c in train_df.columns if c not in ['time', 'segment', 'label']]
    else:
        feat_cols = [c for c in train_df.columns if c not in ['time', 'segment', 'label']]
    
    X = train_df[feat_cols].values
    y = train_df['label'].values
    
    return X, y


def train_models(X_train: np.ndarray, y_train: np.ndarray, 
                X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Train Logistic Regression and Random Forest"""
    results = {}
    
    lr = LogisticRegression(C=1.0, class_weight='balanced', 
                           max_iter=1000, random_state=cfg.random_state)
    lr.fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    results['lr'] = {
        'model': lr,
        'y_pred': y_pred_lr,
        'y_proba': y_proba_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_lr)
    }
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                class_weight='balanced',
                                random_state=cfg.random_state)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['rf'] = {
        'model': rf,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_rf)
    }
    
    return results


def train_models_loso(features_by_subject: Dict[str, pd.DataFrame],
                     use_reduced_features: bool = True) -> Dict:
    """Train models using Leave-One-Subject-Out (LOSO) cross-validation
    
    This provides unbiased accuracy estimates by testing on entirely unseen subjects.
    Expected baseline: 60-75% (realistic for generalization across subjects).
    
    Args:
        features_by_subject: Dict mapping subject names to their feature dataframes
        use_reduced_features: Use only key features to prevent overfitting
        
    Returns:
        Dict with LOSO results for LR and RF models
    """
    subject_names = list(features_by_subject.keys())
    n_subjects = len(subject_names)
    
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION ({n_subjects} subjects)")
    print('='*70)
    
    # Storage for predictions across all folds
    all_y_true_lr = []
    all_y_pred_lr = []
    all_y_proba_lr = []
    
    all_y_true_rf = []
    all_y_pred_rf = []
    all_y_proba_rf = []
    
    fold_results = []
    
    for test_subject in subject_names:
        print(f"\nFold: Testing on {test_subject}")
        
        # Prepare training data (all subjects except test_subject)
        train_subjects = [s for s in subject_names if s != test_subject]
        
        train_dfs = []
        for subj in train_subjects:
            df = features_by_subject[subj]
            # Filter to B4 and A5 only (exclude bridge)
            df_filtered = df[df['segment'].isin(['B4', 'A5'])].copy()
            train_dfs.append(df_filtered)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Prepare test data
        test_df = features_by_subject[test_subject]
        test_df = test_df[test_df['segment'].isin(['B4', 'A5'])].copy()
        
        # Build feature matrices
        if use_reduced_features:
            key_features = ['alpha_rel', 'beta_rel', 'emg_rms']
            feat_cols = [c for c in key_features if c in train_df.columns]
            if len(feat_cols) == 0:
                feat_cols = [c for c in train_df.columns 
                           if c not in ['time', 'segment', 'label', 'soft_label', 'label_type', 'subject_id']]
        else:
            feat_cols = [c for c in train_df.columns 
                        if c not in ['time', 'segment', 'label', 'soft_label', 'label_type', 'subject_id']]
        
        X_train = train_df[feat_cols].values
        y_train = train_df['label'].values
        
        X_test = test_df[feat_cols].values
        y_test = test_df['label'].values
        
        print(f"  Train: {len(X_train)} samples from {len(train_subjects)} subjects")
        print(f"  Test: {len(X_test)} samples from {test_subject}")
        
        # Train Logistic Regression
        lr = LogisticRegression(C=1.0, class_weight='balanced', 
                               max_iter=1000, random_state=cfg.random_state)
        lr.fit(X_train, y_train)
        
        y_pred_lr = lr.predict(X_test)
        y_proba_lr = lr.predict_proba(X_test)[:, 1]
        
        acc_lr = accuracy_score(y_test, y_pred_lr)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced',
                                   random_state=cfg.random_state)
        rf.fit(X_train, y_train)
        
        y_pred_rf = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)[:, 1]
        
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        print(f"  LR Accuracy: {acc_lr:.3f}")
        print(f"  RF Accuracy: {acc_rf:.3f}")
        
        # Store predictions
        all_y_true_lr.extend(y_test)
        all_y_pred_lr.extend(y_pred_lr)
        all_y_proba_lr.extend(y_proba_lr)
        
        all_y_true_rf.extend(y_test)
        all_y_pred_rf.extend(y_pred_rf)
        all_y_proba_rf.extend(y_proba_rf)
        
        fold_results.append({
            'test_subject': test_subject,
            'lr_accuracy': acc_lr,
            'rf_accuracy': acc_rf,
            'n_test': len(y_test)
        })
    
    # Convert to arrays
    all_y_true_lr = np.array(all_y_true_lr)
    all_y_pred_lr = np.array(all_y_pred_lr)
    all_y_proba_lr = np.array(all_y_proba_lr)
    
    all_y_true_rf = np.array(all_y_true_rf)
    all_y_pred_rf = np.array(all_y_pred_rf)
    all_y_proba_rf = np.array(all_y_proba_rf)
    
    # Compute overall metrics
    results = {
        'lr': {
            'y_pred': all_y_pred_lr,
            'y_proba': all_y_proba_lr,
            'accuracy': accuracy_score(all_y_true_lr, all_y_pred_lr),
            'precision': precision_score(all_y_true_lr, all_y_pred_lr, zero_division=0),
            'recall': recall_score(all_y_true_lr, all_y_pred_lr, zero_division=0),
            'f1': f1_score(all_y_true_lr, all_y_pred_lr, zero_division=0),
            'roc_auc': roc_auc_score(all_y_true_lr, all_y_proba_lr),
            'fold_results': fold_results
        },
        'rf': {
            'y_pred': all_y_pred_rf,
            'y_proba': all_y_proba_rf,
            'accuracy': accuracy_score(all_y_true_rf, all_y_pred_rf),
            'precision': precision_score(all_y_true_rf, all_y_pred_rf, zero_division=0),
            'recall': recall_score(all_y_true_rf, all_y_pred_rf, zero_division=0),
            'f1': f1_score(all_y_true_rf, all_y_pred_rf, zero_division=0),
            'roc_auc': roc_auc_score(all_y_true_rf, all_y_proba_rf),
            'fold_results': fold_results
        },
        'y_test': all_y_true_lr  # Same for both models
    }
    
    print(f"\n{'='*70}")
    print("LOSO CROSS-VALIDATION RESULTS (Unbiased Generalization)")
    print('='*70)
    print(f"\nLogistic Regression:")
    print(f"  Accuracy:  {results['lr']['accuracy']:.3f}")
    print(f"  Precision: {results['lr']['precision']:.3f}")
    print(f"  Recall:    {results['lr']['recall']:.3f}")
    print(f"  F1-Score:  {results['lr']['f1']:.3f}")
    print(f"  ROC-AUC:   {results['lr']['roc_auc']:.3f}")
    
    print(f"\nRandom Forest:")
    print(f"  Accuracy:  {results['rf']['accuracy']:.3f}")
    print(f"  Precision: {results['rf']['precision']:.3f}")
    print(f"  Recall:    {results['rf']['recall']:.3f}")
    print(f"  F1-Score:  {results['rf']['f1']:.3f}")
    print(f"  ROC-AUC:   {results['rf']['roc_auc']:.3f}")
    
    print(f"\nPer-Subject Results:")
    for fold in fold_results:
        print(f"  {fold['test_subject']}: LR={fold['lr_accuracy']:.3f}, RF={fold['rf_accuracy']:.3f}")
    
    print('='*70)
    
    return results


def plot_model_performance(y_test: np.ndarray, results: Dict, show: bool = False):
    """Plot ROC curves and confusion matrices"""
    fig = plt.figure(figsize=(14, 6), dpi=cfg.fig_dpi)
    gs = GridSpec(1, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        auc = res['roc_auc']
        label = f"{name.upper()} (AUC={auc:.3f})"
        ax1.plot(fpr, tpr, lw=2, label=label)
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i+1])
        cm = confusion_matrix(y_test, res['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['A5 (Sat)', 'B4 (Hun)'],
                   yticklabels=['A5 (Sat)', 'B4 (Hun)'])
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name.upper()} Confusion Matrix')
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_feature_importance(model, feat_cols: List[str], top_n: int = 15, show: bool = False):
    """Plot feature importance for Random Forest"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=cfg.fig_dpi)
    
    ax.barh(range(top_n), importances[indices], color=cfg.color_a5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    if show:
        plt.show()


def predict_hunger_proba(model, feat_df: pd.DataFrame, use_reduced_features: bool = True) -> np.ndarray:
    """Predict hunger probability for all windows
    
    Args:
        model: Trained classifier
        feat_df: Feature dataframe
        use_reduced_features: Must match the setting used during training
    """
    if use_reduced_features:
        # Use only key features that were used during training
        key_features = ['alpha_rel', 'beta_rel', 'emg_rms']
        feat_cols = [c for c in key_features if c in feat_df.columns]
        if len(feat_cols) == 0:
            # Fallback to all features if key features not available
            feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    else:
        feat_cols = [c for c in feat_df.columns if c not in ['time', 'segment', 'label']]
    
    X = feat_df[feat_cols].values
    
    proba = model.predict_proba(X)[:, 1]
    
    return proba


def plot_prediction_timeline(feat_df: pd.DataFrame, proba: np.ndarray, 
                             idx: np.ndarray, model_name: str = "Model", show: bool = False):
    """Plot predicted probability over time with true segments
    Avoid double-scaling time: if `feat_df['time']` is already in seconds,
    use it directly; otherwise convert samples to seconds.
    """
    t_raw = feat_df['time'].values
    # Heuristic: if values look like raw sample indices, convert to seconds
    t = (t_raw / cfg.fs) if (np.nanmax(t_raw) > 1000) else t_raw
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.fig_dpi, sharex=True)
    
    segments = feat_df['segment'].values
    seg_changes = np.where(segments[:-1] != segments[1:])[0] + 1
    seg_bounds = [0] + list(seg_changes) + [len(segments)]
    
    seg_map = {'B4': cfg.color_b4, 'bridge': cfg.color_bridge, 'A5': cfg.color_a5}
    
    for i in range(len(seg_bounds) - 1):
        start, end = seg_bounds[i], seg_bounds[i+1]
        seg = segments[start]
        color = seg_map.get(seg, 'gray')
        
        for ax in axes:
            ax.axvspan(t[start], t[end-1], alpha=0.15, color=color)
    
    axes[0].plot(t, proba, 'k', lw=2, label='P(Hunger)')
    axes[0].axhline(0.5, color='gray', ls='--', alpha=0.5)
    axes[0].set_ylabel('P(B4 | features)')
    axes[0].set_title(f'{model_name}: Predicted Hunger Probability')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].plot(t, idx, cfg.color_bridge, lw=2, label='Hunger index')
    axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Hunger Index')
    axes[1].set_title('Hunger Index (for reference)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    handles = [mpatches.Patch(color=cfg.color_b4, alpha=0.3, label='B4 (Hunger)'),
               mpatches.Patch(color=cfg.color_bridge, alpha=0.3, label='Bridge'),
               mpatches.Patch(color=cfg.color_a5, alpha=0.3, label='A5 (Satiety)')]
    axes[0].legend(handles=handles, loc='upper right', ncol=3)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
