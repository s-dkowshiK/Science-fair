!pip install GEOparse pandas numpy matplotlib seaborn scikit-learn scipy

import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from scipy.stats import binomtest

# Autophagynet

AUTOPHAGY_NET_GENES = [
    # Core Machinery
    'ATG5', 'ATG7', 'ATG12', 'ATG16L1', 'ATG10', 'ATG3', 'ATG4A', 'ATG4B',
    'ATG4C', 'ATG4D', 'MAP1LC3B', 'MAP1LC3A', 'MAP1LC3C', 'GABARAP', 'GABARAPL1',
    'GABARAPL2', 'BECN1', 'PIK3C3', 'PIK3R4', 'UVRAG', 'AMBRA1', 'KIAA0226', 'BCL2',

    # Initiation & Regulation
    'ULK1', 'ULK2', 'RB1CC1', 'ATG13', 'ATG101', 'MTOR', 'RPS6KB1', 'RICTOR',
    'RPTOR', 'AKT1', 'MAPK1', 'TP53', 'FOXO3', 'TFEB', 'ZKSCAN3',

    # Cargo Recognition
    'SQSTM1', 'NBR1', 'CALCOCO2', 'OPTN', 'TAX1BP1', 'WDFY3', 'ALFY',

    # Lysosome & Fusion
    'LAMP1', 'LAMP2', 'LAMP3', 'CTSD', 'CTSB', 'CTSL', 'HEXA', 'GBA',
    'STX17', 'SNAP29', 'VAMP8', 'RAB7A', 'RAB5A', 'VPS34', 'VPS11', 'VPS16',
    'VPS18', 'VPS33A', 'VPS39', 'VPS41',

    # Mitophagy
    'PINK1', 'PRKN', 'BNIP3', 'BNIP3L', 'FUNDC1', 'DNM1L', 'MFN1', 'MFN2',

    # Stress Sensors
    'HIF1A', 'EIF2AK4', 'DDIT3', 'HSPA5'
]

# Patient Datasets

PHENO_CONFIG = {
    "GSE63060": {"status_col": "characteristics_ch1.0.status",
                 "sick_values": ["Alzheimer", "AD", "MCI"]},
    "GSE63061": {"status_col": "characteristics_ch1.0.status",
                 "sick_values": ["Alzheimer", "AD", "MCI"]},
    "GSE65391": {"status_col": "characteristics_ch1.10.disease state",
                 "sick_values": ["Sepsis", "Septic", "Pneumonia", "SLE"]},
    "GSE11909": {"status_col": "characteristics_ch1.4.Illness",
                 "sick_values": ["SLE", "Lupus"]},
    "GSE109887": {"status_col": "characteristics_ch1.3.disease state",
                  "sick_values": ["AD", "Alzheimer"]}
}

datasets_config = [
    {"id": "GSE63060", "label": "Alz_Blood_1"},
    {"id": "GSE63061", "label": "Alz_Blood_2"},
    {"id": "GSE65391", "label": "Sepsis"},
    {"id": "GSE11909", "label": "Lupus"},
    {"id": "GSE109887", "label": "Brain_Tissue"}
]

# functions
#Gene mapping

def get_gene_mapping(gse):
    """Maps probe IDs to Gene Symbols for the specific platform."""
    platform_id = gse.metadata['platform_id'][0]
    if platform_id not in gse.gpls:
        gpl = GEOparse.get_GEO(geo=platform_id, destdir="./", silent=True)
    else:
        gpl = gse.gpls[platform_id]

    possible_cols = ['Symbol', 'ORF', 'Gene Symbol', 'GENE_SYMBOL', 'ILMN_Gene', 'gene_assignment', 'Gene']
    gene_col = next((c for c in possible_cols if c in gpl.table.columns), None)
    if not gene_col:
        return None

    id_col = 'ID_REF' if 'ID_REF' in gpl.table.columns else 'ID'
    mapping = gpl.table.set_index(id_col)[gene_col].to_dict()

    clean_map = {}
    for k, v in mapping.items():
        if pd.isna(v):
            continue
        clean_map[k] = str(v).split('///')[0].split('//')[0].strip()
    return clean_map

def load_dataset(gse_id, label):
    """Downloads and processes a single GEO dataset."""
    print(f"\n  Processing {gse_id} ({label})...")
    if gse_id not in PHENO_CONFIG:
        return None

    cfg = PHENO_CONFIG[gse_id]
    gse = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=True)
    meta = gse.phenotype_data

    if cfg["status_col"] not in meta.columns:
        print(f"   Phenotype column not found.")
        return None

    # Create Labels (1=Sick, 0=Healthy)
    sick_vals = [s.lower() for s in cfg["sick_values"]]
    status = meta[cfg["status_col"].astype(str).apply(
        lambda x: 1 if any(s in x.lower() for s in sick_vals) else 0
    ].values

    # Process Expression Data
    expr = gse.pivot_samples("VALUE")
    mapping = get_gene_mapping(gse)
    if not mapping:
        return None

    expr.index = expr.index.map(lambda x: mapping.get(x, np.nan))
    expr = expr.loc[expr.index.notnull()].groupby(level=0).mean().T

    # Align Samples
    common = sorted(set(expr.index).intersection(meta.index))
    if not common:
        return None

    df = expr.loc[common].copy()
    df['Status'] = status[meta.index.get_indexer(common)]
    df['Cohort'] = label

    # Filter for AutophagyNet Genes
    keep_cols = [g for g in AUTOPHAGY_NET_GENES if g in df.columns] + ['Status', 'Cohort']
    return df[keep_cols]

dfs = []
for ds in datasets_config:
    d = load_dataset(ds["id"], ds["label"])
    if d is not None:
        dfs.append(d)

if not dfs:
    raise RuntimeError("No data loaded.")

full_df = pd.concat(dfs, ignore_index=True).fillna(0)
print(f"\n SUCCESS: Loaded {full_df.shape[0]} patients. "
      f"Scanning {len([c for c in full_df.columns if c in AUTOPHAGY_NET_GENES])} AutophagyNet genes.")

#Training and validation

gene_cols = [c for c in full_df.columns if c in AUTOPHAGY_NET_GENES]
X = full_df[gene_cols].copy()
y = full_df['Status'].values
cohorts = full_df['Cohort'].values

# Batch Correction (per-cohort standardization)
print("\nApplying Batch Correction...")
X_corr = X.copy()
for c in np.unique(cohorts):
    idx = (cohorts == c)
    X_corr.loc[idx, :] = StandardScaler().fit_transform(X_corr.loc[idx, :])

# Classifier Setup
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
logo = LeaveOneGroupOut()

# Blood cohorts only for LOGO
train_mask = (cohorts != "Brain_Tissue")
X_blood = X_corr.loc[train_mask]
y_blood = y[train_mask]
groups_blood = cohorts[train_mask]

results = []
aucs = []
pvals = []
names = []
conf_mats = []

print("\n" + "=" * 50)
print("    AUTOPHAGYNET UNIVERSAL VALIDATION (LOGO)   ")
print("=" * 50)

for train_idx, test_idx in logo.split(X_blood, y_blood, groups=groups_blood):
    test_disease = np.unique(groups_blood[test_idx])[0]

    rf.fit(X_blood.iloc[train_idx], y_blood[train_idx])
    preds = rf.predict(X_blood.iloc[test_idx])
    probs = rf.predict_proba(X_blood.iloc[test_idx])[:, 1]

    acc = accuracy_score(y_blood[test_idx], preds)
    auc = roc_auc_score(y_blood[test_idx], probs)
    correct = np.sum(preds == y_blood[test_idx])
    total = len(test_idx)
    pval = binomtest(correct, total, p=0.5, alternative='greater').pvalue

    cm = confusion_matrix(y_blood[test_idx], preds)
    tn, fp, fn, tp = cm.ravel()

    results.append(acc)
    aucs.append(auc)
    pvals.append(pval)
    names.append(test_disease)
    conf_mats.append(cm)

    print(f"\nTesting on: {test_disease}")
    print(f" -> Accuracy:    {acc:.2%}")
    print(f" -> AUC:         {auc:.3f}")
    print(f" -> p-value:     {pval:.4f}{'   Significant' if pval < 0.05 else '    Not significant'}")
    print(f" -> Found Sick:  {tp} | Missed Sick: {fn}")

print("\n" + "-" * 50)
print(f" Average Blood Accuracy : {np.mean(results):.2%}")
print(f" Average Blood AUC      : {np.mean(aucs):.3f}")
print(f" Average Blood p-value  : {np.mean(pvals):.4f}")
print("-" * 50)

# Mirror brain tests

brain_acc = brain_auc = brain_p = None

if "Brain_Tissue" in cohorts:
    print("\n MIRROR TEST (Brain Tissue)")
    rf.fit(X_blood, y_blood)  # Train on ALL blood cohorts

    brain_mask = (cohorts == "Brain_Tissue")
    brain_preds = rf.predict(X_corr.loc[brain_mask])
    brain_probs = rf.predict_proba(X_corr.loc[brain_mask])[:, 1]

    brain_acc = accuracy_score(y[brain_mask], brain_preds)
    brain_auc = roc_auc_score(y[brain_mask], brain_probs)
    correct = np.sum(brain_preds == y[brain_mask])
    total = len(brain_preds)
    brain_p = binomtest(correct, total, p=0.5, alternative='greater').pvalue

    print(f" -> Brain Accuracy : {brain_acc:.2%}")
    print(f" -> Brain AUC      : {brain_auc:.3f}")
    print(f" -> Brain p-value  : {brain_p:.4f}{'   Significant' if brain_p < 0.05 else '    Not significant'}")

    if brain_acc > 0.60:
        print("\n Hypothesis Confirmed: Blood AutophagyNet signature predicts Brain pathology.")

#Visualization

# --- Plot 1: Accuracy Bar Chart (colored by threshold) ---
plt.figure(figsize=(9, 5))
colors = ['#ff9999' if s < 0.6 else '#66b3ff' for s in results]
bars = plt.bar(names, results, color=colors, edgecolor='black')
for bar, pv in zip(bars, pvals):
    label = f"{bar.get_height():.1%}\n(p={pv:.3f})"
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             label, ha='center', fontsize=9)
plt.title("AutophagyNet Universal Validation — Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1.15)
plt.axhline(0.5, color='red', linestyle='--', label="Random Chance")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: AUC Bar Chart ---
plt.figure(figsize=(9, 5))
auc_colors = ['#ff9999' if a < 0.6 else '#66b3ff' for a in aucs]
bars = plt.bar(names, aucs, color=auc_colors, edgecolor='black')
for bar, pv in zip(bars, pvals):
    label = f"{bar.get_height():.3f}\n(p={pv:.3f})"
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             label, ha='center', fontsize=9)
plt.title("AutophagyNet Universal Validation — AUC")
plt.ylabel("AUC")
plt.ylim(0, 1.15)
plt.axhline(0.5, color='red', linestyle='--', label="Random Chance")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 3: Brain Tissue Mirror Test (Accuracy + AUC side by side) ---
if brain_acc is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].bar(["Brain_Tissue"], [brain_acc],
                color='#66b3ff' if brain_acc >= 0.6 else '#ff9999', edgecolor='black')
    axes[0].text(0, brain_acc + 0.02, f"{brain_acc:.1%}\n(p={brain_p:.4f})", ha='center', fontsize=10)
    axes[0].set_title("Brain Tissue Mirror Test — Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.15)
    axes[0].axhline(0.5, color='red', linestyle='--', label="Random Chance")
    axes[0].legend()

    axes[1].bar(["Brain_Tissue"], [brain_auc],
                color='#66b3ff' if brain_auc >= 0.6 else '#ff9999', edgecolor='black')
    axes[1].text(0, brain_auc + 0.02, f"{brain_auc:.3f}\n(p={brain_p:.4f})", ha='center', fontsize=10)
    axes[1].set_title("Brain Tissue Mirror Test — AUC")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1.15)
    axes[1].axhline(0.5, color='red', linestyle='--', label="Random Chance")
    axes[1].legend()

    plt.suptitle("Mirror Test: Blood Signature → Brain Pathology", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

# --- Plot 4: Feature Importance (Top 15 Genes) ---
rf.fit(X_blood, y_blood)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 6))
plt.title("Top 15 Predictive Genes (AutophagyNet Features)")
plt.bar(range(len(indices)), importances[indices], align="center", color="teal", edgecolor='black')
plt.xticks(range(len(indices)), [gene_cols[i] for i in indices], rotation=45, ha='right')
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()