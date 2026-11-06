"""Event-level evaluation at a FIXED false-alarm rate (zero-shot sim models vs. baselines on real data).
  python event_metrics.py [--horizons 5 10] [--targets drop nrloss hho tput] [--fprs 0.05 0.10] [--no-speed] [--real-only]
Definitions (all on real data, 1 Hz):
  * operating point: threshold = (1-FPR) quantile of the scores of real NEGATIVE windows. This is an *oracle*
    calibration (it uses real labels) and is used only to compare methods at equal false-alarm rate.
  * warning run: maximal run of consecutive samples with label=1 (an event follows within H s).
  * event recall: fraction of warning runs (not starting at the segment start) with >= 1 alarm inside the run.
  * lead time: H - (offset of first alarm inside the run). Exact time-to-event for onset-type events (nrloss, hho);
    for relative-threshold events (drop, tput) it is the remaining part of the warning window (approximation).
  * false-alarm episodes: rising edges of the alarm signal on negative windows, reported per hour.
"""
import argparse, os
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from common import *
from sim_dataset import sim_samples
from real_dataset import real_samples
from train_eval import fit, score, subsample

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "results")


def event_stats(y, s, G, thr, H):
    alarm = s >= thr; ev = det = fa = 0; leads = []
    for g in np.unique(G):
        i = np.where(G == g)[0]; yy, aa = y[i], alarm[i]
        prev = np.r_[False, aa[:-1]]
        fa += int(np.sum(aa & ~prev & (yy == 0)))
        pos = (yy == 1).astype(int); d = np.diff(np.r_[0, pos, 0])
        for a0, e0 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if a0 == 0: continue
            ev += 1; hit = np.where(aa[a0:e0])[0]
            if len(hit): det += 1; leads.append(H - hit[0])
    tp = int(np.sum(alarm & (y == 1))); fp = int(np.sum(alarm & (y == 0)))
    return dict(precision=tp / max(tp + fp, 1), prevalence=float(np.mean(y == 1)), n_events=ev,
                event_recall=det / max(ev, 1), median_lead_s=float(np.median(leads)) if leads else np.nan,
                false_alarms_per_hour=fa / (len(y) / 3600.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10]); ap.add_argument("--targets", nargs="+", default=["drop", "nrloss", "hho", "tput"])
    ap.add_argument("--fprs", type=float, nargs="+", default=[0.05, 0.10]); ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--real-only", action="store_true", help="also add out-of-fold real-trained reference"); ap.add_argument("--max-sim", type=int, default=100000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0)
    sim = pickle.load(open(os.path.join(DATA, "sim_trajs.pkl"), "rb")); real = pickle.load(open(os.path.join(DATA, "real_segs.pkl"), "rb"))
    rows = []
    for H in a.horizons:
        Xtr, Ytr, _ = subsample(*sim_samples(sim["train"], H), a.max_sim, rng); Xr, Yr, Gr = real_samples(real, H)
        if a.no_speed:
            Xtr[:, -2:] = 0; Xr[:, -2:] = 0
        models = {k: fit(Xtr, Ytr[k]) for k in ("drop", "drop_geo", "los2nlos")}
        for tgt in a.targets:
            y = Yr[tgt]; sc = {"random": rng.random(len(y)), "heuristic": -Xr[:, IDX_D_LAST],
                              "sim_drop": score(models["drop"], Xr), "sim_dropgeo": score(models["drop_geo"], Xr), "sim_los2nlos": score(models["los2nlos"], Xr)}
            if a.real_only:
                oof = np.full(len(y), np.nan)
                for tr, te in GroupKFold(n_splits=min(5, len(np.unique(Gr)))).split(Xr, groups=Gr): oof[te] = score(fit(Xr[tr], y[tr]), Xr[te])
                sc["real_only"] = oof
            for meth, s in sc.items():
                for fpr in a.fprs:
                    thr = np.quantile(s[(y == 0) & np.isfinite(s)], 1 - fpr)
                    rows.append(dict(H=H, target=tgt, method=meth, fpr=fpr, **event_stats(y, np.nan_to_num(s, nan=-1e9), Gr, thr, H)))
            print(f"H={H} {tgt} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "event_metrics.csv"), index=False)
    pd.set_option("display.width", 220); print(df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()