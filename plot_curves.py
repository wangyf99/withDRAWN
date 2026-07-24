import glob, os, re, sys
import numpy as np
import matplotlib.pyplot as plt

# Pass the folder containing the extracted CSVs as a command-line arg, e.g.:
#   python3 plot_prroc.py /Users/alexwang/Desktop/withDRAWN/prroc
# Otherwise it defaults to a folder named "prroc" next to this script.
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prroc")

# Files look like: {group}_{fold}nonensembleall-{model}-curves.csv
# e.g. 1_0nonensembleall-tpotdefault-curves.csv, 2_7nonensembleall-tpotsk-curves.csv
FNAME_RE = re.compile(r"^(\d+)_(\d+)nonensembleall-(.+)-curves\.csv$")

def load_fold(path):
    with open(path) as f:
        rows = [list(map(float, line.strip().split(","))) for line in f if line.strip()]
    fpr, tpr, precision, recall = rows
    return np.array(fpr), np.array(tpr), np.array(precision), np.array(recall)

def auc(x, y):
    order = np.argsort(x)
    xs = np.array(x)[order]
    ys = np.array(y)[order]
    return np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2)

# Discover categories: group + model, e.g. "1-tpotdefault", "2-tpotsk"
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*-curves.csv")))
if not all_files:
    raise FileNotFoundError(f"No '*-curves.csv' files found in {DATA_DIR}")

groups = {}  # category -> list of file paths
for path in all_files:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        print(f"Skipping unrecognized filename: {os.path.basename(path)}")
        continue
    group, fold, model = m.groups()
    cat = f"group{group}-{model}"
    groups.setdefault(cat, []).append(path)

categories = sorted(groups.keys())
palette = plt.cm.tab10.colors
colors = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

mean_fpr_grid = np.linspace(0, 1, 200)
mean_recall_grid = np.linspace(0, 1, 200)

results = {}

for cat in categories:
    files = sorted(groups[cat])
    tprs, precisions, roc_aucs, pr_aucs = [], [], [], []
    for path in files:
        fpr, tpr, precision, recall = load_fold(path)

        order = np.argsort(fpr)
        fpr_s, tpr_s = fpr[order], tpr[order]
        tpr_interp = np.interp(mean_fpr_grid, fpr_s, tpr_s)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_aucs.append(auc(fpr, tpr))

        order2 = np.argsort(recall)
        recall_s, prec_s = recall[order2], precision[order2]
        prec_interp = np.interp(mean_recall_grid, recall_s, prec_s)
        precisions.append(prec_interp)
        pr_aucs.append(auc(recall, precision))

    results[cat] = {
        "mean_tpr": np.mean(tprs, axis=0),
        "std_tpr": np.std(tprs, axis=0),
        "mean_prec": np.mean(precisions, axis=0),
        "std_prec": np.std(precisions, axis=0),
        "mean_roc_auc": np.mean(roc_aucs),
        "std_roc_auc": np.std(roc_aucs),
        "mean_pr_auc": np.mean(pr_aucs),
        "std_pr_auc": np.std(pr_aucs),
        "n_folds": len(files),
    }

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for cat in categories:
    r = results[cat]
    ax.plot(mean_fpr_grid, r["mean_tpr"], color=colors[cat], lw=2,
             label=f"{cat} (AUC={r['mean_roc_auc']:.3f}±{r['std_roc_auc']:.3f})")
    ax.fill_between(mean_fpr_grid, r["mean_tpr"]-r["std_tpr"], r["mean_tpr"]+r["std_tpr"],
                     color=colors[cat], alpha=0.15)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (mean ± std across folds)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

ax = axes[1]
for cat in categories:
    r = results[cat]
    ax.plot(mean_recall_grid, r["mean_prec"], color=colors[cat], lw=2,
             label=f"{cat} (AUC={r['mean_pr_auc']:.3f}±{r['std_pr_auc']:.3f})")
    ax.fill_between(mean_recall_grid, r["mean_prec"]-r["std_prec"], r["mean_prec"]+r["std_prec"],
                     color=colors[cat], alpha=0.15)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves (mean ± std across folds)")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.0, 1.05)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr_roc_curves.png")
plt.savefig(out_path, dpi=150)
print("Saved to", out_path)

for cat in categories:
    r = results[cat]
    print(f"{cat}: n_folds={r['n_folds']}, ROC-AUC={r['mean_roc_auc']:.4f}±{r['std_roc_auc']:.4f}, PR-AUC={r['mean_pr_auc']:.4f}±{r['std_pr_auc']:.4f}")