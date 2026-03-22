import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Scientific Batch Correction (Z-Score per Cohort)
def batch_correction(X, cohorts):
    X_corr = X.copy()
    for cohort in np.unique(cohorts):
        mask = (cohorts == cohort)
        # Scales each dataset to a common mean and variance
        X_corr.loc[mask] = StandardScaler().fit_transform(X_corr.loc[mask])
    return X_corr

# 2. LOGO Cross-Validation (The Generalizability Proof)
def logo_cross_validation(X, y, groups):
    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    scores = []
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        rf.fit(X.iloc[train_idx], y[train_idx])
        preds = rf.predict(X.iloc[test_idx])
        # Balanced accuracy is the gold standard for medical datasets
        scores.append(balanced_accuracy_score(y[test_idx], preds))
    
    return np.mean(scores), scores

# 3. Brain Tissue Transfer Test (The 'Mirror' Test)
def brain_transfer_test(X_blood, y_blood, X_brain, y_brain):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_blood, y_blood) # Train on Blood
    
    preds = rf.predict(X_brain) # Predict on Brain
    probs = rf.predict_proba(X_brain)[:, 1]
    
    return balanced_accuracy_score(y_brain, preds), roc_auc_score(y_brain, probs)

# 4. Random Genes Control (The p-value Generator)
def random_genes_experiment(X_full, y, groups, n_iterations=1000):
    null_scores = []
    all_genes = [c for c in X_full.columns if c not in ['Status', 'Cohort']]
    
    for _ in range(n_iterations):
        # Pick random genes to see if they perform as well as Autophagy genes
        random_subset = np.random.choice(all_genes, size=len(included_genes), replace=False)
        score, _ = logo_cross_validation(X_full[random_subset], y, groups)
        null_scores.append(score)
    return null_scores

# 5. Statistical Summary
def generate_summary(actual_acc, null_dist, brain_acc):
    # Standard scientific p-value calculation
    p_val = (np.sum(np.array(null_dist) >= actual_acc) + 1) / (len(null_dist) + 1)
    print(f"Mean Validation Accuracy: {actual_acc:.2%}")
    print(f"Permutation p-value: {p_val:.4f}")
    print(f"Brain Transfer Accuracy: {brain_acc:.2%}")

# 6. Top 10 Genes Analysis
def top_genes_analysis(X, y):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(10)

# 7. Fixed Visualizations (Removed 'Epochs' for RF)
def plot_results(cohort_names, accuracy_list):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cohort_names, y=accuracy_list, palette="viridis")
    plt.axhline(0.5, color='red', linestyle='--', label="Random Chance")
    plt.title('Performance Across Independent Lab Cohorts')
    plt.ylabel('Balanced Accuracy')
    plt.ylim(0, 1)
    plt.show()