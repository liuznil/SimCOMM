"""Does the zero-shot result depend on which cities are in the simulation training set?
  python single_city_robustness.py [--horizons 5 10] [--no-speed]
Needs data/sim_trajs.pkl produced by sim_dataset.py with several scenarios (trajectories carry a 'scen' tag).
Trains on (a) each single city, (b) leave-one-city-out, (c) all cities; evaluates zero-shot on real events.
"""
import argparse, os, pickle
import numpy as np, pandas as pd
from common import *
from sim_dataset import sim_samples
from real_dataset import real_samples
from train_eval import fit, score, subsample, metrics, boot_ci

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "results")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10]); ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--targets", nargs="+", default=["drop", "nrloss"]); ap.add_argument("--labels", nargs="+", default=["los2nlos", "drop"])
    ap.add_argument("--boot", type=int, default=50); ap.add_argument("--max-sim", type=int, default=100000)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0)
    sim = pickle.load(open(os.path.join(DATA, "sim_trajs.pkl"), "rb")); real = pickle.load(open(os.path.join(DATA, "real_segs.pkl"), "rb"))
    cities = sorted({t["scen"] for t in sim["train"]}); print("cities:", cities)
    sets = {"all": cities}; sets.update({f"only:{c}": [c] for c in cities}); sets.update({f"without:{c}": [x for x in cities if x != c] for c in cities if len(cities) > 1})
    rows = []
    for H in a.horizons:
        Xr, Yr, Gr = real_samples(real, H)
        if a.no_speed: Xr[:, -2:] = 0
        for sname, cs in sets.items():
            trs = [t for t in sim["train"] if t["scen"] in cs]
            Xtr, Ytr, _ = subsample(*sim_samples(trs, H), a.max_sim, rng)
            if a.no_speed: Xtr[:, -2:] = 0
            for lab in a.labels:
                m = fit(Xtr, Ytr[lab]); s = score(m, Xr)
                for tgt in a.targets:
                    mt = metrics(Yr[tgt], s); lo, hi = boot_ci(Yr[tgt], s, Gr, a.boot, rng) if a.boot else (np.nan, np.nan)
                    rows.append(dict(H=H, train_set=sname, label=lab, target=tgt, ci_lo=lo, ci_hi=hi, **mt))
            print(f"H={H} {sname} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "single_city.csv"), index=False)
    pd.set_option("display.width", 220)
    for (H, lab, tgt), g in df.groupby(["H", "label", "target"]):
        one = g[g.train_set.str.startswith("only:")].auroc
        print(f"H={H} label={lab} target={tgt}: all={g[g.train_set=='all'].auroc.iloc[0]:.3f}  single-city min/median/max = {one.min():.3f}/{one.median():.3f}/{one.max():.3f}")
    print(df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()