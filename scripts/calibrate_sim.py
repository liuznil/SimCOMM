"""Choose simulation realism parameters (smoothing, blocker rate/depth) WITHOUT using any real label:
only the real *signal statistics* are compared with the simulated ones.
  python calibrate_sim.py --scenario city_0_newyork_28 city_9_sanfrancisco_28 ... [--real data/real_segs.pkl]
  python calibrate_sim.py --synthetic
Reports, per setting: sim-vs-real domain-classifier AUROC (speed zeroed; lower = more similar) and the
single-feature AUROC of d_min / d_std / below_max (0.5 = identical distributions).
Requires data/real_segs.pkl (run real_dataset.py first).
"""
import argparse, itertools, os, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from common import *
from sim_dataset import load_grid, synthetic_grid, build, sim_samples
from real_dataset import real_samples

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", nargs="+"); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--tx", type=int, default=0); ap.add_argument("--n", type=int, default=60, help="trajectories per scenario")
ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.7, 0.5])
ap.add_argument("--blocker-rates", type=float, nargs="+", default=[0.0, 0.005, 0.01])
ap.add_argument("--dip-max", type=float, nargs="+", default=[25.0, 12.0])
a = ap.parse_args()
rng = np.random.default_rng(0)
real = pickle.load(open(os.path.join(DATA, "real_segs.pkl"), "rb"))
Xr, _, Gr = real_samples(real, 5)
grids = {}
for nm in (["synthetic"] if a.synthetic else a.scenario):
    grids[nm] = synthetic_grid() if a.synthetic else load_grid(nm, a.tx)
J = {n: FEATURE_NAMES.index(n) for n in ("d_min", "d_std", "below_max")}
rows = []
for alpha, br, dm in itertools.product(a.alphas, a.blocker_rates, a.dip_max):
    if br == 0 and dm != a.dip_max[0]: continue          # depth is irrelevant without blockers
    trajs = []
    for i, (nm, (P, L, res)) in enumerate(grids.items()):
        trajs += build(P, L, res, a.n, 0, 300, br, i, alpha, (min(8.0, dm), dm))["train"]
    Xs, _, Gs = sim_samples(trajs, 5)
    n = min(len(Xs), len(Xr), 20000)
    si = rng.choice(len(Xs), n, replace=False); ri = rng.choice(len(Xr), n, replace=False)
    d = np.vstack([Xs[si], Xr[ri]]); d[:, -2:] = 0
    dl = np.r_[np.zeros(n), np.ones(n)]
    dg = np.asarray([f"sim:{g}" for g in Gs[si]] + [f"real:{g}" for g in Gr[ri]])
    dtr, dte = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(d, dl, groups=dg))
    m = GB(max_iter=60).fit(d[dtr], dl[dtr])
    row = dict(alpha=alpha, blocker_rate=br, dip_max=dm, domain_auroc=roc_auc_score(dl[dte], m.predict_proba(d[dte])[:, 1]))
    for k, j in J.items(): row[k] = roc_auc_score(dl, d[:, j])
    rows.append(row); print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
df = pd.DataFrame(rows).sort_values("domain_auroc"); print("\nbest settings (lowest domain AUROC first):\n", df.round(3).to_string(index=False))
df.to_csv(os.path.join(DATA, "calibration.csv"), index=False)