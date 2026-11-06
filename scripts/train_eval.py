"""Step 3: train on simulation, test on real events (zero-shot), compare with baselines.
  python train_eval.py [--boot 100]
Methods (all scored out-of-fold on the SAME real test folds, folds split by run/segment):
  heuristic        : -last_diff (signal currently falling)            [no learning]
  sim_drop         : GBDT trained on sim, label = >=10 dB drop         [zero-shot]
  sim_los2nlos     : GBDT trained on sim, label = LOS->NLOS (physical) [zero-shot]
  sim+real_k       : sim + k real training segments (sample-efficiency)
  real_only        : GBDT trained on real training folds only (upper bound)
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import argparse, os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from common import *
from sim_dataset import sim_samples
from real_dataset import real_samples

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "results")
TARGETS = ("drop", "tput", "nrloss", "hho")
K_LIST = (2, 5, 10)


def fit(X, y, w=None, seed=0):
    ok = np.isfinite(y); X, y = X[ok], y[ok].astype(int)
    w = np.ones(len(y)) if w is None else w[ok]
    pos = y.sum()
    cw = np.where(y == 1, 0.5 / max(pos, 1), 0.5 / max(len(y) - pos, 1)) * len(y)   # class balance
    return GB(max_iter=100, learning_rate=0.1, max_depth=6, random_state=seed).fit(X, y, sample_weight=cw * w)

def score(m, X): return m.predict_proba(X)[:, 1]

def metrics(y, s):
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    if y.sum() < 20 or (1 - y).sum() < 20: return dict(auroc=np.nan, ap=np.nan, prev=y.mean() if len(y) else np.nan, lift=np.nan, rec_fpr10=np.nan)
    fpr, tpr, _ = roc_curve(y, s)
    ap = average_precision_score(y, s)
    return dict(auroc=roc_auc_score(y, s), ap=ap, prev=y.mean(), lift=ap / y.mean(), rec_fpr10=float(np.interp(0.1, fpr, tpr)))

def auc_diff(y, s, baseline):
    ok = np.isfinite(y) & np.isfinite(s) & np.isfinite(baseline)
    y, s, baseline = y[ok], s[ok], baseline[ok]
    if y.sum() < 20 or (1 - y).sum() < 20: return np.nan
    return roc_auc_score(y, s) - roc_auc_score(y, baseline)

def boot_ci(y, s, g, n, rng):
    ok = np.isfinite(y) & np.isfinite(s); y, s, g = y[ok], s[ok], g[ok]
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[ii].sum() < len(ii): v.append(roc_auc_score(y[ii], s[ii]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) > 10 else (np.nan, np.nan)

def boot_diff_ci(y, s, baseline, g, n, rng):
    ok = np.isfinite(y) & np.isfinite(s) & np.isfinite(baseline)
    y, s, baseline, g = y[ok], s[ok], baseline[ok], g[ok]
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; v = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[ii].sum() >= 20 and (1 - y[ii]).sum() >= 20:
            v.append(roc_auc_score(y[ii], s[ii]) - roc_auc_score(y[ii], baseline[ii]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) > 10 else (np.nan, np.nan)

def subsample(X, Y, G, n, rng):
    if len(X) <= n: return X, Y, G
    i = rng.choice(len(X), n, replace=False); return X[i], {k: v[i] for k, v in Y.items()}, G[i]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS)); ap.add_argument("--no-speed", action="store_true", help="zero the speed features (diagnose domain shift)"); ap.add_argument("--k-list", type=int, nargs="+", default=list(K_LIST)); ap.add_argument("--targets", nargs="+", default=list(TARGETS)); ap.add_argument("--max-sim", type=int, default=100000); ap.add_argument("--real-weight", type=float, default=0.3)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    sim = pickle.load(open(os.path.join(DATA, "sim_trajs.pkl"), "rb")); real = pickle.load(open(os.path.join(DATA, "real_segs.pkl"), "rb"))
    rows = []
    for H in a.horizons:
        rng = np.random.default_rng(100 + H)
        K = tuple(a.k_list)
        Xtr, Ytr, Gtr = subsample(*sim_samples(sim["train"], H), a.max_sim, rng)
        Xte, Yte, _ = sim_samples(sim["test"], H); Xr, Yr, Gr = real_samples(real, H)
        if a.no_speed:
            for X_ in (Xtr, Xte, Xr): X_[:, -2:] = 0
        sim_models = {"sim_drop": fit(Xtr, Ytr["drop"]), "sim_dropgeo": fit(Xtr, Ytr["drop_geo"]), "sim_los2nlos": fit(Xtr, Ytr["los2nlos"])}
        # (a) sanity: does it work on held-out simulated map area?
        for name, lab, m in (("drop", "drop", sim_models["sim_drop"]), ("drop_geo", "drop_geo", sim_models["sim_dropgeo"]), ("los2nlos", "los2nlos", sim_models["sim_los2nlos"])):
            rows.append(dict(H=H, target=f"SIM-holdout:{name}", method="sim_in_domain", **metrics(Yte[lab], score(m, Xte))))
        # (b) domain shift: how separable are sim vs real features?
        n = min(len(Xtr), len(Xr), 20000)
        si = rng.choice(len(Xtr), n, replace=False); ri = rng.choice(len(Xr), n, replace=False)
        d = np.vstack([Xtr[si], Xr[ri]])
        dl = np.r_[np.zeros(n), np.ones(n)]
        dg = np.asarray([f"sim:{g}" for g in Gtr[si]] + [f"real:{g}" for g in Gr[ri]])
        dtr, dte = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(d, dl, groups=dg))
        dm = GB(max_iter=60).fit(d[dtr], dl[dtr])
        rows.append(dict(H=H, target="domain-classifier", method="sim_vs_real", auroc=roc_auc_score(dl[dte], score(dm, d[dte]))))
        fs = [roc_auc_score(dl, d[:, j]) for j in range(d.shape[1])]
        pd.DataFrame(dict(feature=FEATURE_NAMES, auroc_sim_vs_real=fs)).assign(H=H).to_csv(os.path.join(OUT, f"feature_shift_H{H}.csv"), index=False)
        print("most sim-vs-real separable features:", sorted(zip(np.round(np.abs(np.array(fs) - .5) + .5, 3), FEATURE_NAMES))[::-1][:4])
        # (c) real events
        folds = list(GroupKFold(n_splits=min(5, len(np.unique(Gr)))).split(Xr, groups=Gr))
        for tgt in a.targets:
            target_rng = np.random.default_rng(1000 * H + TARGETS.index(tgt))
            y = Yr[tgt]; sc = {"heuristic": -Xr[:, IDX_D_LAST], "sim_drop": score(sim_models["sim_drop"], Xr), "sim_dropgeo": score(sim_models["sim_dropgeo"], Xr),
                              "sim_los2nlos": score(sim_models["sim_los2nlos"], Xr)}
            for k in ("real_only",) + tuple(f"sim+real_{k}" for k in K) + tuple(f"realonly_{k}" for k in K): sc[k] = np.full(len(y), np.nan)
            for tr, te in folds:
                tg = np.unique(Gr[tr]); target_rng.shuffle(tg)
                sc["real_only"][te] = score(fit(Xr[tr], y[tr]), Xr[te])
                for k in K:
                    sel = np.isin(Gr[tr], tg[:k]); Xk, yk = Xr[tr][sel], y[tr][sel]
                    wr = a.real_weight * len(Xtr) / max(len(Xk), 1)
                    Xm, ym = np.vstack([Xtr, Xk]), np.r_[Ytr["drop"], yk]; wm = np.r_[np.ones(len(Xtr)), np.full(len(Xk), wr)]
                    sc[f"sim+real_{k}"][te] = score(fit(Xm, ym, wm), Xr[te])
                    sc[f"realonly_{k}"][te] = score(fit(Xk, yk), Xr[te]) if np.nansum(yk) > 5 else np.nan   # control: same k real runs, no sim
            for meth, s in sc.items():
                m = metrics(y, s); lo, hi = boot_ci(y, s, Gr, a.boot, target_rng) if a.boot and np.isfinite(m["auroc"]) else (np.nan, np.nan)
                if meth == "heuristic":
                    delta, delta_lo, delta_hi = 0.0, 0.0, 0.0
                else:
                    delta = auc_diff(y, s, sc["heuristic"])
                    delta_lo, delta_hi = (boot_diff_ci(y, s, sc["heuristic"], Gr, a.boot, target_rng)
                                          if a.boot and np.isfinite(delta) else (np.nan, np.nan))
                rows.append(dict(H=H, target=tgt, method=meth, ci_lo=lo, ci_hi=hi,
                                 delta_vs_heuristic=delta, delta_ci_lo=delta_lo, delta_ci_hi=delta_hi, **m))
            print(f"H={H}s {tgt}: " + ", ".join(f"{k}={metrics(y, s)['auroc']:.3f}" for k, s in sc.items()))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    for H in a.horizons:                                     # bar plots of zero-shot vs baselines
        d = df[(df.H == H) & df.target.isin(a.targets)].pivot(index="target", columns="method", values="auroc").reindex([x for x in a.targets])
        d.plot.bar(figsize=(10, 4)); plt.axhline(0.5, color="k", ls=":"); plt.ylabel("AUROC"); plt.title(f"Real-event prediction, horizon {H}s")
        plt.legend(fontsize=7, ncol=3); plt.tight_layout(); plt.savefig(os.path.join(OUT, f"auroc_H{H}.png"), dpi=150); plt.close()
    print("saved results/results.csv and results/auroc_H*.png")

if __name__ == "__main__":
    main()