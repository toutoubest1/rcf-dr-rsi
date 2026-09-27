"""Record the fixed KuaiRec design (no estimator): kappa, population ESS, exact candidate-policy values, hashes.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from real_data.kuairec.kuairec_design import pop_ess  # noqa: E402
from real_data.kuairec.run_kuairec import kuairec_context, TAB  # noqa: E402


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.round(a, 12)).tobytes()).hexdigest()[:16]


def main():
    c = kuairec_context()
    d = c["d"]
    rec = dict(K=d.K, categories=d.cat_names, n_users=len(d.users), n_train=len(d.tr), n_eval=len(d.ev),
               cellsize_min=int(d.cellsize.min()), cellsize_median=float(np.median(d.cellsize)),
               target_mean_max_prob=float(d.target[d.ev].max(1).mean()), sbs=d.cat_names[d.sbs], sbs2=d.cat_names[d.sbs2],
               logging={ov: dict(kappa=c["kappa"][ov], pop_ess_train=pop_ess(d.target[d.tr], c["pb"][ov][d.tr]),
                                 pop_ess_eval=pop_ess(d.target[d.ev], c["pb"][ov][d.ev]),
                                 min_pb=float(c["pb"][ov][d.ev].min()),
                                 max_ratio=float((d.target[d.ev] / c["pb"][ov][d.ev]).max()), sha=h(c["pb"][ov]))
                        for ov in c["pb"]},
               truth=c["truth"], vbs={r: float(d.cell_mean[r][d.ev].max(1).mean()) for r in d.cell_mean},
               cand_pop_ess={p: {ov: pop_ess(P[d.ev], c["pb"][ov][d.ev]) for ov in c["pb"]} for p, P in d.cands.items()},
               sha_target=h(d.target), sha_pcs=h(d.pcs))
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(TAB / "design_kuairec.json", "w"), indent=1)
    print(json.dumps(rec, indent=1)[:3000])


if __name__ == "__main__":
    main()
