"""
Runs the simple_tox_pred.py logic but uses the pre-exported TPOT pipelines
(XGBClassifier max_depth=10, n_estimators=150) to skip the lengthy TPOT search.
"""
import os, sys, random, math
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, matthews_corrcoef,
                              roc_curve, precision_recall_curve)

def make_xgb(random_state=0):
    '''XGBClassifier(max_depth=10, n_estimators=150), matching the params
    shown in tpotdefault-onelevel-tpot_exported_pipeline.py /
    tpotsk-onelevel-tpot_exported_pipeline.py. use_label_encoder was removed
    in xgboost>=2.0, so we only pass it on older versions that still need it.'''
    kwargs = dict(max_depth=10, n_estimators=150, random_state=random_state,
                  eval_metric='logloss')
    try:
        return XGBClassifier(use_label_encoder=False, **kwargs)
    except TypeError:
        return XGBClassifier(**kwargs)

os.chdir('/Users/alexwang/Documents/GitHub/withDRAWN')

OUTDIR = 'simple_8010/'
os.makedirs(OUTDIR + 'prroc', exist_ok=True)

# ── helpers (identical to simple_tox_pred.py) ────────────────────────────────

def load_labels(label_key_number, random_seed, train_percent, test_percent):
    positive, negative = [], []
    with open('tox_labels.csv') as fo:
        for i, line in enumerate(fo):
            if i == 0: continue
            sp = line[:-1].split(',')
            (positive if sp[label_key_number] == '1' else negative).append(sp[0].lower())
    random.seed(random_seed)
    rp = random.sample(positive, len(positive))
    rn = random.sample(negative, len(negative))
    np1 = math.floor(len(positive) * train_percent)
    nn1 = math.floor(len(negative) * train_percent)
    npt = math.floor(len(positive) * test_percent)
    nnt = math.floor(len(negative) * test_percent)
    train1  = rp[:np1]  + rn[:nn1]
    test1   = rp[np1:np1+npt]  + rn[nn1:nn1+nnt]
    test2   = rp[np1+npt:]     + rn[nn1+nnt:]
    Lt1  = [1]*np1  + [0]*nn1
    Lte1 = [1]*npt  + [0]*nnt
    Lte2 = [1]*len(rp[np1+npt:]) + [0]*len(rn[nn1+nnt:])
    return train1, test1, test2, Lt1, Lte1, Lte2

def load_nongraph(drug_names_list, filename):
    track = {}
    with open(filename) as fo:
        for line in fo:
            sp = line[:-1].split(',')
            vals = []
            for elt in sp[1:]:
                try:    vals.append(float(elt))
                except: vals.append(0.0)
            track[sp[0].lower()] = vals
    return [track[d] for d in drug_names_list]

def load_data(drug_names_list):
    sages = load_nongraph(drug_names_list, 'sages.csv')
    fp    = load_nongraph(drug_names_list, 'fp.csv')
    df    = load_nongraph(drug_names_list, 'drug_features.csv')
    ta    = load_nongraph(drug_names_list, 'targetsall.csv')
    all_data = [sages[i]+fp[i]+df[i]+ta[i] for i in range(len(sages))]
    return pd.DataFrame(all_data)

def norm_data_by_train(trainset, testset):
    mins = trainset.min(axis=0)
    maxs = trainset.max(axis=0)
    for col in range(testset.shape[1]):
        mi, ma = mins[col], maxs[col]
        testset[col] = (testset[col]-mi)/(ma-mi)
    return testset.fillna(0)

def split_norm_data(train, test):
    lt = load_data(train);  lte = load_data(test)
    tr = norm_data_by_train(lt, lt)
    te = norm_data_by_train(lt, lte)
    return tr, te

def evaluate(y_true, y_pred):
    return (accuracy_score(y_true,y_pred), roc_auc_score(y_true,y_pred),
            f1_score(y_true,y_pred), precision_score(y_true,y_pred),
            recall_score(y_true,y_pred), matthews_corrcoef(y_true,y_pred))

def update_miss_dict(d, names, labels, preds):
    for i in range(len(names)):
        if int(labels[i]) != int(preds[i]):
            d[names[i]] = d.get(names[i], 0) + 1
    return d

# ── main loop (mirrors tuning_level1 but uses pre-built XGB directly) ─────────

classifier_labels = ['tpotdefault']   # as per simple_tox_pred.py (cl = ['tpotdefault'])

miss_train, miss_test = {}, {}

fout0 = open(OUTDIR+'simple_level0_summary.csv', 'w')
fout0.write('RandomSeed,TestSet,Data,Accuracy,AUROC,F1,Precision,Recall,MCC,Classifier\n')
fout0.close()

for cl_label in classifier_labels:
    print(f"\n=== Classifier: {cl_label} ===")
    # NOTE: simple_tox_pred.py extracts the *already-fitted* TPOT pipeline
    # (clf.fitted_pipeline_.steps[-1][1]) once, with random_state fixed to 0
    # (TPOT bakes random_state=0 into the exported estimator — see
    # tpotdefault-onelevel-tpot_exported_pipeline.py). That same fixed-seed
    # model object is then re-fit on each of the 10 per-seed training sets.
    # The randomness across the rs loop therefore comes ONLY from the data
    # split (load_labels(label_key_number, rs, ...)), never from the model's
    # own random_state. We replicate that here: one XGBClassifier with
    # random_state=0, reused (re-fit) across all 10 seeds.
    exctracted_best_model = make_xgb(random_state=0)

    for rs in range(0, 10):
        print(f"  random seed {rs} …", flush=True)
        rstr, rste1, rste2, rsLtr, rsLte1, rsLte2 = load_labels(1, rs, 0.8, 0.1)
        tr_set, te_set1 = split_norm_data(rstr, rste1)
        _,      te_set2 = split_norm_data(rstr, rste2)

        # Re-fit the SAME model object on this seed's training data, exactly
        # as `rsmodel = exctracted_best_model.fit(rstrain_set1, ...)` does in
        # the original tuning_level1 — random_state stays 0 throughout.
        model = exctracted_best_model.fit(tr_set, np.array(rsLtr))

        yp1      = model.predict(te_set1)
        yproba1  = model.predict_proba(te_set1)
        yp2      = model.predict(te_set2)
        yproba2  = model.predict_proba(te_set2)
        yp_train = model.predict(tr_set)

        miss_train = update_miss_dict(miss_train, rstr,  rsLtr,  yp_train)
        miss_test  = update_miss_dict(miss_test,  rste1, rsLte1, yp1)
        miss_test  = update_miss_dict(miss_test,  rste2, rsLte2, yp2)

        for ts_idx, (Lte, yp, yproba, te_set) in enumerate(
                [(rsLte1, yp1, yproba1, te_set1),
                 (rsLte2, yp2, yproba2, te_set2)], start=1):
            acc,aroc,f1v,prec,rec,mcc = evaluate(np.array(Lte), yp)
            out_str = f"{rs},{ts_idx},alldataonelevel,{acc},{aroc},{f1v},{prec},{rec},{mcc},{cl_label}\n"
            with open(OUTDIR+'simple_level0_summary.csv', 'a') as f:
                f.write(out_str)

            proba_pos = [float(e[1]) for e in yproba]
            fpr,tpr,_ = roc_curve(np.array(Lte), proba_pos, pos_label=1)
            prec_c,rec_c,_ = precision_recall_curve(np.array(Lte), proba_pos)
            fname = OUTDIR+f'prroc/{ts_idx}_{rs}nonensembleall-{cl_label}-curves.csv'
            with open(fname, 'w') as f:
                f.write("\n".join(",".join(map(str,x)) for x in (fpr,tpr,prec_c,rec_c)))

# Write misclassified drugs.
# NOTE: in the original tuning_level1, BOTH miss_dict_train and
# miss_dict_test get appended into the SAME file, 'missclassified_drugs_train.csv'
# (the test-loop write target is a typo/bug in simple_tox_pred.py — it never
# writes a separate missclassified_drugs_test.csv). We reproduce that exact
# behavior here so the output matches, rather than "fixing" the typo.
with open(OUTDIR+'missclassified_drugs_train.csv', 'w') as f:
    for drug, cnt in miss_train.items():
        f.write(f"{drug},{cnt}\n")
    for drug, cnt in miss_test.items():
        f.write(f"{drug},{cnt}\n")

print("\nDone! Results written to", OUTDIR)
