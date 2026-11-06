"""Profile candidate DeepMIMO scenarios on the SIMULATION side only (no real data used),
so that city selection cannot be tuned to the real-data results.
  python profile_scenarios.py --scenario city_0_newyork_28 city_9_sanfrancisco_28 ...
  python profile_scenarios.py --synthetic
Output: table + data/scenario_profile.csv
"""
import argparse, os
import numpy as np, pandas as pd
from common import *
from sim_dataset import load_grid, synthetic_grid, build, sim_samples

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", nargs="+"); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--tx", type=int, default=0); ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()
rows = []
for nm in (["synthetic"] if a.synthetic else a.scenario):
    try:
        P, L, res = synthetic_grid() if a.synthetic else load_grid(nm, a.tx)
        valid = (L != -1) & np.isfinite(P)
        d = build(P, L, res, a.n, 0, 300, 0.0, 0)          # no injected blockers: geometry only
        tr = d["train"]
        if not tr: raise RuntimeError("no valid trajectories (map too fragmented?)")
        _, Y, _ = sim_samples(tr, 5)
        rows.append(dict(scenario=nm, grid=f"{P.shape[0]}x{P.shape[1]}", res_m=round(res, 2), valid_cells=int(valid.sum()),
                         valid_frac=round(valid.mean(), 3), los_share=round(float(np.mean(L[valid] == 1)), 3),
                         los2nlos_prev_H5=round(float(np.nanmean(Y["los2nlos"])), 4),
                         geo_drop_prev_H5=round(float(np.nanmean(Y["drop_geo"])), 4), n_traj=len(tr)))
    except Exception as e:
        rows.append(dict(scenario=nm, error=str(e)[:80]))
df = pd.DataFrame(rows); print(df.to_string(index=False))
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scenario_profile.csv"), index=False)