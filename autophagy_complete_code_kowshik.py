# =============================================================================
#  AUTOPHAGYNET — COMPLETE GOOGLE COLAB PIPELINE
#  Science Fair Project: Autophagy Gene Expression × Disease Classification
#
#  USAGE: Run each cell in order. In Google Colab, paste this entire file
#         into a notebook (one section per cell), or run as a script.
#
#  SECTIONS:
#    0.  Install & imports
#    1.  Gene list & dataset config          (your original code)
#    2.  Helper functions                    (your original code)
#    3.  Load & process all 5 datasets       (your original code)
#    4.  Batch correction                    (your original code)
#    5.  Random Forest + LOGO validation     (your original code)
#    6.  Plots: accuracy, feature importance (your original code)
#    7.  Macro-flux radar charts             (your original code)
#    8.  METADATA DISCOVERY — print all GEO columns (NEW)
#    9.  METADATA EXTRACTION — pull card-ready demographics (NEW)
#    10. CLEANUP METER — real gene expression per sample (NEW)
#    11. PATIENT CARD EXPORT — final CSV for booth cards (NEW)
# =============================================================================


# =============================================================================
# SECTION 0 — Install & imports
# =============================================================================
# Run once per Colab session.

# !pip install GEOparse pandas numpy matplotlib seaborn scikit-learn

import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# =============================================================================
# SECTION 1 — Gene list & dataset config  (ORIGINAL)
# =============================================================================

AUTOPHAGY_NET_GENES = [
    'ATG5','ATG7','ATG12','ATG16L1','ATG10','ATG3',
    'ATG4A','ATG4B','ATG4C','ATG4D',
    'MAP1LC3B','MAP1LC3A','MAP1LC3C',
    'GABARAP','GABARAPL1','GABARAPL2',
    'BECN1','PIK3C3','PIK3R4','UVRAG','AMBRA1','KIAA0226','BCL2',
    'ULK1','ULK2','RB1CC1','ATG13','ATG101',
    'MTOR','RPS6KB1','RICTOR','RPTOR',
    'AKT1','MAPK1','TP53','FOXO3','TFEB','ZKSCAN3',
    'SQSTM1','NBR1','CALCOCO2','OPTN','TAX1BP1','WDFY3','ALFY',
    'LAMP1','LAMP2','LAMP3','CTSD','CTSB','CTSL','HEXA','GBA',
    'STX17','SNAP29','VAMP8','RAB7A','RAB5A',
    'VPS34','VPS11','VPS16','VPS18','VPS33A','VPS39','VPS41',
    'PINK1','PRKN','BNIP3','BNIP3L','FUNDC1','DNM1L','MFN1','MFN2',
    'HIF1A','EIF2AK4','DDIT3','HSPA5',
]

# Status column & sick-value config (used in Sections 2–5 AND Section 9)
PHENO_CONFIG = {
    "GSE63060":  {
        "status_col": "characteristics_ch1.0.status",
        "sick_values": ["Alzheimer", "AD", "MCI"],
    },
    "GSE63061":  {
        "status_col": "characteristics_ch1.0.status",
        "sick_values": ["Alzheimer", "AD", "MCI"],
    },
    "GSE65391":  {
        "status_col": "characteristics_ch1.10.disease state",
        "sick_values": ["Sepsis", "Septic", "Pneumonia", "SLE"],
    },
    "GSE11909":  {
        "status_col": "characteristics_ch1.4.Illness",
        "sick_values": ["SLE", "Lupus"],
    },
    "GSE109887": {
        "status_col": "characteristics_ch1.3.disease state",
        "sick_values": ["AD", "Alzheimer"],
    },
}

datasets_config = [
    {"id": "GSE63060",  "label": "Alz_Blood_1"},
    {"id": "GSE63061",  "label": "Alz_Blood_2"},
    {"id": "GSE65391",  "label": "Sepsis"},
    {"id": "GSE11909",  "label": "Lupus"},
    {"id": "GSE109887", "label": "Brain_Tissue"},
]

MACRO_PHASES = {
    "Initiation": [
        'ULK1','ULK2','RB1CC1','ATG13','ATG101',
        'MTOR','RICTOR','RPTOR','AKT1','PIK3C3',
    ],
    "Loading": [
        'MAP1LC3B','MAP1LC3A','ATG5','ATG7','ATG12',
        'ATG16L1','SQSTM1','NBR1','OPTN','TAX1BP1',
    ],
    "Shredding": [
        'LAMP1','LAMP2','CTSD','CTSB','CTSL',
        'STX17','SNAP29','VAMP8','RAB7A','GBA',
    ],
}

# Cleanup meter gene groups (Sections 10–11)
CLEANUP_GENES = {
    "alarm_system":  ["ULK1", "ULK2", "RB1CC1"],
    "garbage_bag":   ["ATG5", "ATG7", "MAP1LC3B"],
    "cargo_handler": ["SQSTM1", "NBR1", "OPTN"],
    "shredder":      ["LAMP2", "CTSD", "RAB7A"],
    "off_switch":    ["MTOR"],   # inverted: high mTOR = bad cleanup
}

CLEANUP_PLAIN_NAMES = {
    "alarm_system":  "Alarm system (ULK1/ULK2/RB1CC1)",
    "garbage_bag":   "Garbage bag crew (ATG5/ATG7/LC3B)",
    "cargo_handler": "Cargo handler (SQSTM1/NBR1/OPTN)",
    "shredder":      "Shredder (LAMP2/CTSD/RAB7A)",
    "off_switch":    "Off-switch pressure (MTOR) — high = bad",
}


# =============================================================================
# SECTION 2 — Helper functions  (ORIGINAL)
# =============================================================================

def find_gene_symbol_column(gpl_table):
    possible = [
        'Symbol','ORF','Gene Symbol','GENE_SYMBOL',
        'ILMN_Gene','gene_assignment','Gene',
    ]
    for col in possible:
        if col in gpl_table.columns:
            return col
    return None


def clean_gene_symbol(text):
    t = str(text)
    t = t.split('///')[0]
    t = t.split('//')[0]
    return t.strip()


def get_gene_mapping(gse):
    platform_id = gse.metadata['platform_id'][0]
    if platform_id in gse.gpls:
        gpl = gse.gpls[platform_id]
    else:
        gpl = GEOparse.get_GEO(geo=platform_id, destdir="./", silent=True)

    gene_col = find_gene_symbol_column(gpl.table)
    if gene_col is None:
        return None

    id_col = 'ID_REF' if 'ID_REF' in gpl.table.columns else 'ID'
    raw_map = gpl.table.set_index(id_col)[gene_col].to_dict()

    clean_map = {}
    for probe_id, val in raw_map.items():
        if pd.isna(val):
            continue
        clean_map[probe_id] = clean_gene_symbol(val)
    return clean_map


def make_status_labels(meta_df, status_col, sick_values):
    sick_list = [s.lower() for s in sick_values]
    labels = []
    for sample_id in meta_df.index:
        text  = str(meta_df.loc[sample_id, status_col]).lower()
        label = int(any(s in text for s in sick_list))
        labels.append(label)
    return np.array(labels)


def convert_probes_to_genes(expr_df, mapping):
    expr_copy = expr_df.copy()
    new_names = [mapping.get(p, np.nan) for p in expr_copy.index]
    expr_copy.index = new_names
    expr_copy = expr_copy.loc[pd.notnull(expr_copy.index)]
    expr_copy = expr_copy.groupby(expr_copy.index).mean()
    return expr_copy


def keep_only_autophagy_genes(df_samples_by_genes):
    keep = [g for g in AUTOPHAGY_NET_GENES if g in df_samples_by_genes.columns]
    keep += ["Status", "Cohort"]
    return df_samples_by_genes[keep]


def load_one_dataset(gse_id, cohort_label):
    print(f"\nProcessing {gse_id} ({cohort_label})")
    if gse_id not in PHENO_CONFIG:
        return None

    cfg = PHENO_CONFIG[gse_id]
    gse  = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=True)
    meta = gse.phenotype_data

    if cfg["status_col"] not in meta.columns:
        print(f"  Phenotype column not found: {cfg['status_col']}")
        return None

    status_labels = make_status_labels(meta, cfg["status_col"], cfg["sick_values"])
    expr          = gse.pivot_samples("VALUE")
    mapping       = get_gene_mapping(gse)
    if mapping is None:
        return None

    gene_expr      = convert_probes_to_genes(expr, mapping).T
    common_samples = [s for s in gene_expr.index if s in meta.index]
    if not common_samples:
        return None

    df            = gene_expr.loc[common_samples].copy()
    idxs          = meta.index.get_indexer(common_samples)
    df["Status"]  = status_labels[idxs]
    df["Cohort"]  = cohort_label
    df            = keep_only_autophagy_genes(df)
    return df


def batch_correct_within_each_cohort(X_df, cohorts_array):
    X_corr  = X_df.copy()
    for cohort_name in np.unique(cohorts_array):
        mask    = (cohorts_array == cohort_name)
        scaler  = StandardScaler()
        X_corr.loc[mask, :] = scaler.fit_transform(X_corr.loc[mask, :])
    return X_corr


def get_gene_columns(full_df):
    return [c for c in full_df.columns if c in AUTOPHAGY_NET_GENES]


# =============================================================================
# SECTION 3 — Load & process all 5 datasets  (ORIGINAL)
# =============================================================================

print("=" * 50)
print("STEP 1: Download + clean each dataset")
print("=" * 50)

dfs = []
for ds in datasets_config:
    one = load_one_dataset(ds["id"], ds["label"])
    if one is not None:
        dfs.append(one)

if not dfs:
    raise RuntimeError("No datasets loaded. Check your internet connection.")

full_df = pd.concat(dfs, ignore_index=False).fillna(0)

print(f"\nSUCCESS: Loaded {full_df.shape[0]} patients.")
print(f"Scanning {len(full_df.columns) - 2} AutophagyNet genes.")
print(f"Cohorts: {full_df['Cohort'].value_counts().to_dict()}")
print(f"Status:  {full_df['Status'].value_counts().to_dict()}")


# =============================================================================
# SECTION 4 — Batch correction  (ORIGINAL)
# =============================================================================

print("\n" + "=" * 50)
print("STEP 2: Batch correction")
print("=" * 50)

gene_cols = get_gene_columns(full_df)
X         = full_df[gene_cols].copy()
y         = full_df["Status"].values
cohorts   = full_df["Cohort"].values

print("Applying within-cohort StandardScaler batch correction...")
X_corr = batch_correct_within_each_cohort(X, cohorts)
print("Done.")


# =============================================================================
# SECTION 5 — Random Forest + LOGO validation  (ORIGINAL)
# =============================================================================

def run_logo_validation(rf, X_blood, y_blood, groups_blood):
    logo    = LeaveOneGroupOut()
    results = []
    names   = []

    for train_idx, test_idx in logo.split(X_blood, y_blood, groups=groups_blood):
        test_group = np.unique(groups_blood[test_idx])[0]
        rf.fit(X_blood.iloc[train_idx], y_blood[train_idx])
        preds = rf.predict(X_blood.iloc[test_idx])
        acc   = accuracy_score(y_blood[test_idx], preds)
        results.append(acc)
        names.append(test_group)

        tn, fp, fn, tp = confusion_matrix(
            y_blood[test_idx], preds, labels=[0, 1]
        ).ravel()

        print(f"\n  Testing on: {test_group}")
        print(f"    Accuracy   : {acc:.2%}")
        print(f"    Found sick : {tp}  |  Missed sick: {fn}")
        print(f"    False alarm: {fp}  |  True healthy: {tn}")

    return results, names


print("\n" + "=" * 50)
print(" AUTOPHAGYNET UNIVERSAL VALIDATION")
print("=" * 50)

rf          = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
train_mask  = (cohorts != "Brain_Tissue")
X_blood     = X_corr.loc[train_mask]
y_blood     = y[train_mask]
groups_blood = cohorts[train_mask]

results, names = run_logo_validation(rf, X_blood, y_blood, groups_blood)
print(f"\nAverage blood accuracy: {np.mean(results):.2%}")

# Mirror test
brain_acc = None
if "Brain_Tissue" in cohorts:
    print("\n--- MIRROR TEST (Brain Tissue) ---")
    rf.fit(X_blood, y_blood)
    brain_mask  = (cohorts == "Brain_Tissue")
    brain_preds = rf.predict(X_corr.loc[brain_mask])
    brain_acc   = accuracy_score(y[brain_mask], brain_preds)
    print(f"  Brain tissue accuracy: {brain_acc:.2%}")
    if brain_acc > 0.60:
        print("  Hypothesis confirmed: blood AutophagyNet signature "
              "predicts brain pathology.")


# =============================================================================
# SECTION 6 — Plots: accuracy bars + feature importance  (ORIGINAL)
# =============================================================================

def plot_accuracy_bars(names, results, brain_acc=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#66b3ff" if a >= 0.6 else "#ff9999" for a in results]
    bars    = ax.bar(names, results, color=colors, edgecolor="black")

    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{b.get_height():.1%}",
            ha="center", fontsize=10,
        )

    if brain_acc is not None:
        ax.axhline(brain_acc, color="purple", linestyle=":",
                   linewidth=1.5, label=f"Brain mirror: {brain_acc:.1%}")

    ax.axhline(0.5, color="red", linestyle="--", label="Random chance (50%)")
    ax.set_title("AutophagyNet LOGO Validation Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig("accuracy_bars.png", dpi=150)
    plt.show()
    print("Saved: accuracy_bars.png")


def plot_top_feature_importance(rf, gene_cols, top_n=15):
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [gene_cols[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), top_vals, color="teal", align="center")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_names, rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Predictive Autophagy Genes (Feature Importance)")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Saved: feature_importance.png")


print("\n" + "=" * 50)
print("STEP 3: Accuracy & feature importance plots")
print("=" * 50)

plot_accuracy_bars(names, results, brain_acc)
plot_top_feature_importance(rf, gene_cols, top_n=15)


# =============================================================================
# SECTION 7 — Macro-flux radar charts  (ORIGINAL)
# =============================================================================

def compute_flux_table(full_df, macro_phases):
    flux_results = []
    for cohort in full_df["Cohort"].unique():
        cohort_data = full_df[full_df["Cohort"] == cohort]
        for status in [0, 1]:
            subset = cohort_data[cohort_data["Status"] == status]
            scores = {}
            for phase, genes in macro_phases.items():
                valid = [g for g in genes if g in subset.columns]
                scores[phase] = subset[valid].mean(axis=1).mean() if valid else 0
            row = {
                "Cohort": cohort,
                "Status": "Sick" if status == 1 else "Healthy",
                **scores,
            }
            flux_results.append(row)
    return pd.DataFrame(flux_results)


def plot_flux_radar(flux_df, macro_phases, target_cohort):
    labels   = list(macro_phases.keys())
    num_vars = len(labels)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for status, color in zip(["Sick", "Healthy"], ["#e74c3c", "#2ecc71"]):
        row    = flux_df[
            (flux_df["Cohort"] == target_cohort) & (flux_df["Status"] == status)
        ]
        if row.empty:
            continue
        values  = row[labels].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2,
                label=f"{target_cohort} — {status}")
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(
        f"Autophagy Flux Fingerprint: {target_cohort}",
        size=16, y=1.1,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fname = f"radar_{target_cohort}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


print("\n" + "=" * 50)
print("STEP 4: Macro-flux radar charts")
print("=" * 50)

flux_df = compute_flux_table(full_df, MACRO_PHASES)

for ds in datasets_config:
    plot_flux_radar(flux_df, MACRO_PHASES, ds["label"])

print("\nFlux matrix summary:")
print(
    flux_df.pivot(
        index="Cohort", columns="Status",
        values=list(MACRO_PHASES.keys()),
    )
)


# =============================================================================
# SECTION 8 — METADATA DISCOVERY  (NEW)
# Run this to see every available GEO metadata column for each dataset.
# Read the output carefully — column names vary by dataset.
# Update CARD_META_CONFIG in Section 9 to match what you see here.
# =============================================================================

print("\n" + "=" * 55)
print("STEP 5: Discover GEO metadata columns")
print("=" * 55)

gse_cache = {}   # cache loaded GSE objects — reused in Section 9

for ds in datasets_config:
    gse_id = ds["id"]
    print(f"\n{'─'*55}")
    print(f"  {gse_id}  ({ds['label']})")
    print(f"{'─'*55}")

    gse  = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=True)
    gse_cache[gse_id] = gse
    meta = gse.phenotype_data

    print(f"  Total samples: {len(meta)}")

    char_cols = [c for c in meta.columns if "characteristics" in c.lower()]
    if not char_cols:
        print("  No 'characteristics' columns found — printing all columns:")
        char_cols = list(meta.columns)

    for col in char_cols:
        counts = meta[col].value_counts(dropna=False)
        print(f"\n  Column: {col}")
        for val, n in counts.head(6).items():
            print(f"    {str(val)[:65]:<67}  n={n}")
        if len(counts) > 6:
            print(f"    ... {len(counts) - 6} more unique values")

print("\n>>> ACTION: Read the output above carefully.")
print(">>> Update the column names in CARD_META_CONFIG (Section 9)")
print(">>> to match exactly what this output shows for each dataset.")


# =============================================================================
# SECTION 9 — METADATA EXTRACTION  (NEW)
#
# IMPORTANT: After running Section 8, update the column name strings below
# to match the exact output from your Section 8 run.
# The values below are best-guess defaults based on published documentation.
# If a column does not exist in your data, set its value to None and the
# code will skip it gracefully.
# =============================================================================

# ------------------------------------------------------------------
# UPDATE THESE COLUMN NAMES after reading Section 8 output
# ------------------------------------------------------------------
# NOTE: Column names verified from actual GEO metadata output (Section 8).
# GSE65391 is a paediatric SLE cohort — NOT Sepsis. Disease state = SLE/Healthy.
# GSE11909 age is stored as a sentence e.g. "16 yrs, when sample taken." — parsed below.

CARD_META_CONFIG = {
    "GSE63060": {
        "label":      "Alz_Blood_1",
        "status_col": "characteristics_ch1.0.status",
        "sick_vals":  ["AD", "MCI"],
        "card_cols": {
            "age":       "characteristics_ch1.2.age",
            "gender":    "characteristics_ch1.3.gender",
            "ethnicity": "characteristics_ch1.1.ethnicity",
        },
    },
    "GSE63061": {
        "label":      "Alz_Blood_2",
        "status_col": "characteristics_ch1.0.status",
        "sick_vals":  ["AD", "MCI"],
        "card_cols": {
            "age":       "characteristics_ch1.2.age",
            "gender":    "characteristics_ch1.3.gender",
            "ethnicity": "characteristics_ch1.1.ethnicity",
        },
    },
    # GSE65391 is a paediatric SLE cohort — disease state = SLE / Healthy
    # Rich clinical data: SLEDAI, WBC, neutrophil count, treatment, complement
    "GSE65391": {
        "label":      "SLE_Paediatric",
        "status_col": "characteristics_ch1.10.disease state",
        "sick_vals":  ["SLE"],
        "card_cols": {
            "age":              "characteristics_ch1.13.age",
            "gender":           "characteristics_ch1.11.gender",
            "race":             "characteristics_ch1.12.race",
            "sledai":           "characteristics_ch1.53.sledai",
            "disease_activity": "characteristics_ch1.54.disease_activity",
            "treatment":        "characteristics_ch1.51.treatment",
            "wbc":              "characteristics_ch1.18.wbc",
            "complement_c3":    "characteristics_ch1.37.c3",
            "complement_c4":    "characteristics_ch1.38.c4",
            "esr":              "characteristics_ch1.26.esr",
        },
    },
    "GSE11909": {
        "label":      "Lupus",
        "status_col": "characteristics_ch1.4.Illness",
        "sick_vals":  ["SLE"],
        "card_cols": {
            "age":       "characteristics_ch1.0.Age",       # parsed from "16 yrs, when sample taken."
            "gender":    "characteristics_ch1.1.Gender",
            "ethnicity": "characteristics_ch1.2.Ethnicity",
            "race":      "characteristics_ch1.3.Race",
        },
    },
    "GSE109887": {
        "label":      "Brain_Tissue",
        "status_col": "characteristics_ch1.3.disease state",
        "sick_vals":  ["AD"],
        "card_cols": {
            "age":          "characteristics_ch1.1.age",
            "gender":       "characteristics_ch1.0.gender",
            "brain_region": "characteristics_ch1.2.tissue",
        },
    },
}


def safe_get_col(meta, col):
    """Return column series or None if column missing."""
    if col is None or col not in meta.columns:
        return None
    return meta[col]


def classify_status(text, sick_vals):
    t = str(text).lower()
    return "Sick" if any(v.lower() in t for v in sick_vals) else "Healthy"


def parse_age(val):
    """
    Handles multiple age formats across GEO datasets:
      - Plain numeric:  "76"  → "76 yrs"
      - Sentence form:  "16 yrs, when sample taken."  → "16 yrs"
      - Already clean:  "16 yrs"  → "16 yrs"
    Returns a clean string or None.
    """
    import re
    s = str(val).strip()
    if s.lower() in ('', 'nan', 'none', 'n/a', 'not applicable'):
        return None
    # extract leading integer
    m = re.match(r'(\d+)', s)
    if m:
        return f"{m.group(1)} yrs"
    return s


def clean_gender(val):
    """Normalise gender values to Female / Male / Unknown."""
    s = str(val).strip().lower()
    if s in ('f', 'female'):
        return 'Female'
    if s in ('m', 'male'):
        return 'Male'
    if s in ('u', 'unknown', 'not reported'):
        return 'Unknown'
    return str(val).strip()


def clean_race(val):
    """Expand race abbreviations used in GSE65391."""
    mapping = {
        'H':  'Hispanic',
        'AA': 'African American',
        'C':  'Caucasian',
        'AS': 'Asian',
    }
    s = str(val).strip()
    return mapping.get(s, s)


def clean_field(friendly, val):
    """Route each friendly field name through appropriate cleaner."""
    if pd.isna(val) or str(val).strip().lower() in ('nan', 'none', '',
                                                      'not applicable',
                                                      'data not available'):
        return None
    if friendly == 'age':
        return parse_age(val)
    if friendly == 'gender':
        return clean_gender(val)
    if friendly == 'race':
        return clean_race(val)
    return str(val).strip()


def extract_card_metadata(gse_id, cfg, gse_obj, n_per_class=3):
    """
    Selects n_per_class samples per status (Sick / Healthy),
    preferring samples with the most metadata fields populated.
    Cleans age, gender, and race fields automatically.
    Returns a DataFrame ready to join with gene expression.
    """
    meta = gse_obj.phenotype_data.copy()

    if cfg["status_col"] not in meta.columns:
        print(f"  WARNING {gse_id}: status col '{cfg['status_col']}' not found. "
              f"Available: {[c for c in meta.columns if 'char' in c][:5]}")
        return None

    meta["_status"] = meta[cfg["status_col"]].apply(
        lambda x: classify_status(x, cfg["sick_vals"])
    )
    meta["_dataset"] = gse_id
    meta["_cohort"]  = cfg["label"]

    # pull and clean card fields
    for friendly, col in cfg["card_cols"].items():
        col_data = safe_get_col(meta, col)
        if col_data is not None:
            meta[f"_card_{friendly}"] = col_data.apply(
                lambda v: clean_field(friendly, v)
            )
        else:
            meta[f"_card_{friendly}"] = None

    card_field_cols       = [f"_card_{k}" for k in cfg["card_cols"]]
    meta["_completeness"] = meta[card_field_cols].notna().sum(axis=1)

    # pick most-complete samples per class
    selected = (
        meta.groupby("_status", group_keys=False)
            .apply(lambda g: g.sort_values("_completeness", ascending=False)
                              .head(n_per_class))
    )

    out = selected[["_dataset","_cohort","_status","_completeness"]
                   + card_field_cols].copy()
    out.columns = (
        ["dataset", "cohort", "status", "completeness"]
        + list(cfg["card_cols"].keys())
    )
    out.index.name = "sample_id"
    return out


print("\n" + "=" * 55)
print("STEP 6: Extract card-ready metadata from all datasets")
print("=" * 55)

all_card_meta = []

for gse_id, cfg in CARD_META_CONFIG.items():
    print(f"\n  {gse_id} ({cfg['label']})")
    gse_obj = gse_cache.get(gse_id)
    if gse_obj is None:
        gse_obj = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=True)
        gse_cache[gse_id] = gse_obj

    df = extract_card_metadata(gse_id, cfg, gse_obj, n_per_class=3)
    if df is not None:
        all_card_meta.append(df)
        print(df.to_string())

card_meta_df = pd.concat(all_card_meta)
print(f"\nTotal card-candidate samples: {len(card_meta_df)}")


# =============================================================================
# SECTION 10 — CLEANUP METER  (NEW)
# Computes real gene expression scores per sample from full_df.
# Each cleanup phase = mean z-scored expression of its genes.
# off_switch (mTOR) is inverted: high mTOR → low cleanup score.
# =============================================================================

def compute_cleanup_meter_raw(sample_id, X_corr_df):
    """
    Returns raw mean z-score for each cleanup phase for one sample.
    Uses X_corr (batch-corrected, z-scored expression matrix).
    """
    if sample_id not in X_corr_df.index:
        return None

    row    = X_corr_df.loc[sample_id]
    meter  = {}
    for phase, genes in CLEANUP_GENES.items():
        available = [g for g in genes if g in X_corr_df.columns]
        if not available:
            meter[phase] = None
        else:
            meter[phase] = float(row[available].mean())
    return meter


def normalise_meter_to_100(meter_raw, X_corr_df):
    """
    Maps raw z-scores to 0–100 display values.
    z = -2  →  0  (very low activity)
    z =  0  →  50 (average)
    z = +2  → 100 (very high activity)
    Clamp to [0, 100].
    off_switch is inverted.
    """
    normed = {}
    for phase, raw in meter_raw.items():
        if raw is None:
            normed[phase] = None
            continue
        score = int(min(100, max(0, (raw + 2) / 4 * 100)))
        if phase == "off_switch":
            score = 100 - score   # high mTOR = high suppression = low cleanup
        normed[phase] = score
    return normed


def meter_to_label(score):
    """Convert 0–100 score to display label and colour category."""
    if score is None:
        return "N/A", "gray"
    if score >= 65:
        return "Active",    "green"
    if score >= 45:
        return "Borderline","orange"
    if score >= 25:
        return "Low",       "red"
    return "Very low", "red"


print("\n" + "=" * 55)
print("STEP 7: Compute cleanup meters from real expression data")
print("=" * 55)

meter_rows = []

for sample_id in card_meta_df.index:
    raw    = compute_cleanup_meter_raw(sample_id, X_corr)
    if raw is None:
        print(f"  Sample {sample_id} not in X_corr — skipping.")
        continue

    normed = normalise_meter_to_100(raw, X_corr)
    row    = {"sample_id": sample_id}

    for phase, score in normed.items():
        label, colour = meter_to_label(score)
        row[f"{phase}_score"]  = score
        row[f"{phase}_label"]  = label
        row[f"{phase}_colour"] = colour

    meter_rows.append(row)

meter_df = pd.DataFrame(meter_rows).set_index("sample_id")

print(f"\nCleanup meters computed for {len(meter_df)} samples.")
print("\nSample output (first 5 rows):")
score_cols = [f"{p}_score" for p in CLEANUP_GENES]
print(meter_df[score_cols].head().to_string())


# =============================================================================
# SECTION 11 — PATIENT CARD EXPORT  (NEW)
# Joins metadata + cleanup meter + RF prediction confidence.
# Saves patient_card_data.csv — one row per candidate patient card.
# =============================================================================

print("\n" + "=" * 55)
print("STEP 8: Build final patient card data & export")
print("=" * 55)

# -- RF prediction confidence for each sample --------------------------------
def get_rf_confidence(sample_id, X_corr_df, rf_model, gene_cols):
    """
    Returns (predicted_label, confidence_pct) for one sample.
    0 = Healthy, 1 = Sick.
    """
    if sample_id not in X_corr_df.index:
        return None, None
    row   = X_corr_df.loc[[sample_id], gene_cols]
    proba = rf_model.predict_proba(row)[0]
    pred  = int(rf_model.predict(row)[0])
    conf  = round(float(max(proba)) * 100, 1)
    return pred, conf


# make sure rf is trained on full blood set
rf.fit(X_blood, y_blood)

confidence_rows = []
for sample_id in card_meta_df.index:
    pred, conf = get_rf_confidence(sample_id, X_corr, rf, gene_cols)
    confidence_rows.append({
        "sample_id":        sample_id,
        "rf_prediction":    "Sick" if pred == 1 else ("Healthy" if pred == 0 else None),
        "rf_confidence_pct": conf,
    })

conf_df = pd.DataFrame(confidence_rows).set_index("sample_id")

# -- Also pull the top-5 feature gene values for each sample -----------------
top5_idx   = np.argsort(rf.feature_importances_)[::-1][:5]
top5_genes = [gene_cols[i] for i in top5_idx]

top5_rows = []
for sample_id in card_meta_df.index:
    if sample_id not in X_corr.index:
        top5_rows.append({"sample_id": sample_id})
        continue
    row = {"sample_id": sample_id}
    for g in top5_genes:
        if g in X_corr.columns:
            z = float(X_corr.loc[sample_id, g])
            row[f"gene_{g}_zscore"] = round(z, 3)
            if   z >  0.5: row[f"gene_{g}_level"] = "High"
            elif z < -0.5: row[f"gene_{g}_level"] = "Low"
            else:          row[f"gene_{g}_level"] = "Medium"
    top5_rows.append(row)

top5_df = pd.DataFrame(top5_rows).set_index("sample_id")

# -- Join everything ---------------------------------------------------------
final_card_df = (
    card_meta_df
    .join(meter_df,  how="left")
    .join(conf_df,   how="left")
    .join(top5_df,   how="left")
)

# -- Print summary -----------------------------------------------------------
print("\nFINAL PATIENT CARD DATA SUMMARY")
print("─" * 55)

for sample_id, row in final_card_df.iterrows():
    print(f"\n{'─'*50}")
    print(f"  Sample   : {sample_id}")
    print(f"  Dataset  : {row.get('dataset','')}  ({row.get('cohort','')})")
    print(f"  Status   : {row.get('status','')}")

    # demographics — field names match CARD_META_CONFIG keys
    for field in ["age","gender","ethnicity","race","sledai",
                  "disease_activity","treatment","wbc","complement_c3",
                  "complement_c4","esr","brain_region"]:
        val = row.get(field)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            print(f"  {field:<14}: {val}")

    # cleanup meter
    print(f"  Cleanup meter (0=very low, 100=active):")
    for phase in CLEANUP_GENES:
        score  = row.get(f"{phase}_score")
        label  = row.get(f"{phase}_label", "")
        pname  = CLEANUP_PLAIN_NAMES.get(phase, phase)
        bar    = ("█" * (score // 10) + "░" * (10 - score // 10)
                  if score is not None else "N/A")
        print(f"    {pname:<45}  {bar}  {score}  ({label})")

    # RF confidence
    pred = row.get("rf_prediction")
    conf = row.get("rf_confidence_pct")
    if pred is not None:
        print(f"  RF prediction: {pred}  (confidence: {conf}%)")

    # top gene levels
    print(f"  Top-5 gene activity levels for card front:")
    for g in top5_genes:
        level  = row.get(f"gene_{g}_level")
        zscore = row.get(f"gene_{g}_zscore")
        if level is not None:
            print(f"    {g:<12}: {level:<8}  (z={zscore})")

# -- Save CSV ----------------------------------------------------------------
output_path = "patient_card_data.csv"
final_card_df.to_csv(output_path)
print(f"\n{'='*55}")
print(f"  Saved: {output_path}")
print(f"  Rows : {len(final_card_df)}")
print(f"  Cols : {len(final_card_df.columns)}")
print(f"{'='*55}")

# -- Quick bar chart of cleanup scores per cohort ----------------------------
score_cols = [f"{p}_score" for p in CLEANUP_GENES]
plot_df    = final_card_df[score_cols + ["cohort","status"]].copy()
plot_df    = plot_df.dropna(subset=score_cols)

if not plot_df.empty:
    fig, axes = plt.subplots(1, len(score_cols), figsize=(16, 5), sharey=True)
    palette   = {"Sick": "#e74c3c", "Healthy": "#2ecc71"}

    for ax, col in zip(axes, score_cols):
        phase_name = col.replace("_score", "").replace("_", " ").title()
        for (cohort, status), grp in plot_df.groupby(["cohort", "status"]):
            ax.scatter(
                [cohort] * len(grp),
                grp[col],
                c=palette.get(status, "gray"),
                label=status, alpha=0.8, s=60,
                edgecolors="white", linewidths=0.5,
            )
        ax.set_title(phase_name, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)

    # deduplicate legend
    handles, labels_ = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), loc="upper right")

    fig.suptitle(
        "Cleanup meter scores — card-candidate samples\n"
        "Red = sick  ·  Green = healthy  ·  Dashed = cohort average",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("cleanup_meter_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: cleanup_meter_overview.png")

print("\nAll done. Files saved:")
print("  accuracy_bars.png")
print("  feature_importance.png")
print("  radar_<cohort>.png  (one per cohort)")
print("  cleanup_meter_overview.png")
print("  patient_card_data.csv  <-- use this for your booth cards")
