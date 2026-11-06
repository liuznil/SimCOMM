"""How much of the real 'signal drop' label is really a 5G loss (NR -> 4G, signal filled with the -140 dBm floor)?
  python decompose_drop.py [--horizons 5 10] [--no-speed]
Part 1: composition of real drop-positive windows (5G loss involved / horizontal HO only / neither = 'pure drop').
Part 2: zero-shot AUROC of the simulation models on the ORIGINAL drop label and on the 'pure drop' labels
        (drop with no 5G loss in the horizon; windows with 5G loss are excluded, not counted as negatives).
"""
import argparse, os, pickle
import numpy as np, pandas as pd
from common import *
from sim_dataset import sim_samples
from real_dataset import real_samples
from train_eval import fit, score, subsample, metrics, boot_ci

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "results")


def aux(segs, H):
    """per-sample auxiliary flags aligned with real_samples(): 5G loss in horizon, tower change in horizon, on-5G now."""
    nl, tc, on5 = [], [], []
    for sg in segs:
        n = len(sg["sig"])
        if n < W + H + 2: continue
        sl = slice(W - 1, n - H)
        nl.append(future_agg((~sg["conn"]).astype(float), H, np.max)[sl] == 1)
        chg = np.r_[0, (np.diff(sg["tower"].astype(float)) != 0).astype(float)]
        tc.append(future_agg(chg, H, np.max)[sl] == 1); on5.append(sg["conn"][sl])
    return np.concatenate(nl), np.concatenate(tc), np.concatenate(on5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10]); ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--boot", type=int, default=100); ap.add_argument("--max-sim", type=int, default=100000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0)
    sim = pickle.load(open(os.path.join(DATA, "sim_trajs.pkl"), "rb")); real = pickle.load(open(os.path.join(DATA, "real_segs.pkl"), "rb"))
    comp, rows = [], []
    for H in a.horizons:
        Xtr, Ytr, _ = subsample(*sim_samples(sim["train"], H), a.max_sim, rng); Xr, Yr, Gr = real_samples(real, H)
        nl, tc, on5 = aux(real, H); assert len(nl) == len(Xr)
        if a.no_speed: Xtr[:, -2:] = 0; Xr[:, -2:] = 0
        d = Yr["drop"] == 1
        comp.append(dict(H=H, n=len(Xr), frac_already_on_4G=1 - on5.mean(), drop_prevalence=d.mean(),
                         drop_pos_with_5G_loss=(d & nl).sum() / d.sum(), drop_pos_HHO_only=(d & ~nl & tc).sum() / d.sum(),
                         drop_pos_pure=(d & ~nl & ~tc).sum() / d.sum(), pure_drop_prevalence=(d & ~nl).mean()))
        models = {k: fit(Xtr, Ytr[k]) for k in ("drop", "drop_geo", "los2nlos")}
        sc = {"heuristic": -Xr[:, IDX_D_LAST], "sim_drop": score(models["drop"], Xr), "sim_dropgeo": score(models["drop_geo"], Xr), "sim_los2nlos": score(models["los2nlos"], Xr)}
        pure = np.where(d & nl, np.nan, d.astype(float))                       # exclude 5G-loss windows
        variants = {"drop_original": (Yr["drop"], np.ones(len(d), bool)), "drop_pure": (pure, np.ones(len(d), bool)),
                    "drop_pure_on5G_only": (pure, on5)}
        for vn, (y, m) in variants.items():
            for meth, s in sc.items():
                yy = np.where(m, y, np.nan); mt = metrics(yy, s); lo, hi = boot_ci(yy, s, Gr, a.boot, rng) if a.boot else (np.nan, np.nan)
                rows.append(dict(H=H, label=vn, method=meth, ci_lo=lo, ci_hi=hi, **mt))
    pd.set_option("display.width", 220); c = pd.DataFrame(comp); print(c.round(3).to_string(index=False))
    r = pd.DataFrame(rows); print(r.round(3).to_string(index=False))
    c.to_csv(os.path.join(OUT, "drop_composition.csv"), index=False); r.to_csv(os.path.join(OUT, "drop_decomposition.csv"), index=False)

if __name__ == "__main__":
    main()