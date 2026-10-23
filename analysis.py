"""Child ASD screening: fixed cutoff rule vs calibrated, CV-tuned LR+LightGBM average.
Stand-in experiment (UCI id 419); not the paper's data."""
import json, os, re, sys, time
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from ucimlrepo import fetch_ucirepo

SEEDS = [42, 43, 44]
CUTOFF = 6  # standard screening rule: flag when item sum >= 6


def load():
    """Download with retry/backoff; return item matrix, background features, label."""
    for k in range(5):
        try:
            ds = fetch_ucirepo(id=419)
            break
        except Exception as e:
            print("download failed:", e)
            if k == 4:
                sys.exit("could not download data")
            time.sleep(2 ** k * 2)
    X, y = ds.data.features.copy(), ds.data.targets.iloc[:, 0]
    X.columns = [str(c).strip() for c in X.columns]
    items = [c for c in X.columns if re.fullmatch(r"A\d+_Score", c, re.I)]
    if len(items) != 10:
        sys.exit(f"expected 10 item columns, found {items}")
    F = X[items].apply(pd.to_numeric, errors="coerce")
    low = {c.lower(): c for c in X.columns}  # 'result' (item sum) is excluded: it would duplicate the rule
    if "age" in low:
        F["age"] = pd.to_numeric(X[low["age"]], errors="coerce")
    flag = lambda s, ch: s.astype(str).str.strip().str.lower().str[0].map({ch: 1.0}).where(
        s.astype(str).str.strip().str.lower().str[0].isin([ch, "n" if ch == "y" else "f"]), np.nan).fillna(0.0).where(
        s.notna(), np.nan)
    for name, cands, ch in [("male", ["gender", "sex"], "m"), ("jaundice", ["jundice", "jaundice"], "y"),
                            ("family_hx", ["austim", "autism"], "y")]:
        for c in cands:
            if c in low:
                F[name] = flag(X[low[c]], ch)
                break
    yb = y.astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0, "1": 1, "0": 0})
    ok = yb.notna()
    return F[ok].reset_index(drop=True), yb[ok].astype(int).reset_index(drop=True), items


def evaluate(y, pred, score):
    """Shared evaluation: positive-class F1, sensitivity, ROC-AUC."""
    return {"f1": f1_score(y, pred, zero_division=0), "sensitivity": recall_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, score)}


def pipe(model, scale):
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return Pipeline(steps + [("clf", model)])


def improved(Xtr, ytr, Xte, seed):
    """Tune LR and LGBM by CV on train, calibrate (sigmoid), average, pick threshold from train OOF."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lr = pipe(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed), True)
    gb = pipe(LGBMClassifier(class_weight="balanced", min_child_samples=5, learning_rate=0.05,
                             n_jobs=1, verbose=-1, random_state=seed), False)
    grids = [(lr, {"clf__C": [0.03, 0.1, 0.3, 1, 3, 10]}),
             (gb, {"clf__num_leaves": [4, 8], "clf__n_estimators": [50, 150]})]
    base = [clone(GridSearchCV(m, g, scoring="f1", cv=cv).fit(Xtr, ytr).best_estimator_) for m, g in grids]
    cal = lambda m: CalibratedClassifierCV(clone(m), method="sigmoid", cv=3)
    oof = np.zeros(len(ytr))
    for a, b in cv.split(Xtr, ytr):  # out-of-fold ensemble probabilities for the threshold
        oof[b] = np.mean([cal(m).fit(Xtr.iloc[a], ytr.iloc[a]).predict_proba(Xtr.iloc[b])[:, 1] for m in base], axis=0)
    ths = np.linspace(0.1, 0.9, 33)
    th = ths[int(np.argmax([f1_score(ytr, oof >= t) for t in ths]))]
    p = np.mean([cal(m).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for m in base], axis=0)
    return p, th


def main():
    F, y, items = load()
    print("rows:", len(F), "positive rate:", round(float(y.mean()), 3), "features:", list(F.columns))
    if len(F) < 100 or y.nunique() < 2:
        sys.exit("dataset unusable")
    Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.3, stratify=y, random_state=42)
    # Baseline: deterministic fixed cutoff on the item sum
    s = Xte[items].sum(axis=1)
    rb = evaluate(yte, (s >= CUTOFF).astype(int), s)
    print("baseline", rb)
    runs, ths = [], []
    for sd in SEEDS:
        p, th = improved(Xtr, ytr, Xte, sd)
        runs.append(evaluate(yte, (p >= th).astype(int), p))
        ths.append(th)
        print("improved seed", sd, "threshold", round(float(th), 3), runs[-1])
    m = {}
    for k in ["f1", "sensitivity", "auc"]:
        v = [r[k] for r in runs]
        m[f"baseline_{k}"], m[f"improved_{k}"] = float(rb[k]), float(np.mean(v))
        m[f"improved_{k}_std"] = float(np.std(v))
    m["baseline_score"], m["improved_score"] = m["baseline_f1"], m["improved_f1"]
    m["baseline_score_std"], m["improved_score_std"] = 0.0, m["improved_f1_std"]
    m["improvement"] = m["improved_score"] - m["baseline_score"]
    m["mean_chosen_threshold"] = float(np.mean(ths))
    if not all(np.isfinite(v) for v in m.values()):
        sys.exit("non-finite metrics")
    os.makedirs("figures", exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks, w = ["f1", "sensitivity", "auc"], 0.38
    plt.bar(np.arange(3) - w / 2, [m[f"baseline_{k}"] for k in ks], w, label="cutoff rule")
    plt.bar(np.arange(3) + w / 2, [m[f"improved_{k}"] for k in ks], w,
            yerr=[m[f"improved_{k}_std"] for k in ks], label="calibrated LR+LGBM")
    plt.xticks(range(3), ks); plt.ylabel("test score"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/baseline_vs_improved.png", dpi=120)
    res = {
        "question": "On a public child questionnaire screening dataset, can a calibrated, CV-tuned model that learns item weights beat the standard fixed score-cutoff rule at matching labelled screening outcomes?",
        "method": "Stratified 70/30 split (seed 42) of UCI child ASD screening data (id 419). Baseline: item sum >= 6. Improved: class-balanced, CV-tuned, sigmoid-calibrated LR + LightGBM probability average with a threshold chosen on train out-of-fold predictions; 3 seeds. Test set used once.",
        "dataset": {"name": "UCI Autistic Spectrum Disorder Screening Data for Children", "source_url": "https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children",
                    "n_rows": int(len(F)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "positive-class F1 (test set)", "higher_is_better": True,
        "baseline": "Standard screening rule: sum of the 10 yes/no items, flag if >= 6 (deterministic).",
        "improved": "Learns item weights and uses background features via calibrated, CV-tuned, class-balanced LR + LightGBM average with a CV-chosen decision threshold.",
        "metrics": m,
        "notes": "Small dataset, so the test set is small and differences are noisy; single split. Labels are ASD (not ADHD) and may be closely tied to the item score, so the cutoff rule can be hard to beat. The 'result' column (item sum) was excluded from model features. Baseline is deterministic (std 0); improved std is over seeds 42-44 on the same split. Hyper-parameters are tuned on train only, and the threshold is chosen from train out-of-fold predictions after tuning on the same train data (mild optimism confined to train).",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("saved results.json; improvement =", m["improvement"])


if __name__ == "__main__":
    main()
