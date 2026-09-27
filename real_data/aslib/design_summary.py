"""Record the fixed ASlib design (no estimator is run)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from real_data.aslib.aslib_design import (Design, ESS_TARGETS_FINAL, calibrate_kappa, pop_ess, reward_matrix, zrow)  # noqa: E402

from real_data.aslib.paths import TAB as OUT  # noqa: E402



def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.round(a, 12)).tobytes()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for scen in ["SAT03-16_INDU", "ASP-POTASSCO"]:
        D = Design(scen)
        pe = D.target()
        rec = dict(scenario=scen, K=D.K, algs=D.d["algs"], n_instances=len(D.d["insts"]), n_train=len(D.tr),
                   n_eval=len(D.ev), primary_reward=D.primary, sbs=D.d["algs"][D.sbs], sbs2=D.d["algs"][D.sbs2],
                   n_allsolved_eval=int(D.d["ok"][D.ev].all(1).sum()),
                   target_mean_max_prob=float(pe[D.ev].max(1).mean()), logging={})
        for ov, tgt in ESS_TARGETS_FINAL.items():
            k = 0.0 if tgt is None else calibrate_kappa(pe[D.tr], D.g_hat[D.tr], tgt)[0]
            pb = D.logging(k)
            w = pe[D.ev] / pb[D.ev]
            rec["logging"][ov] = dict(kappa=k, ess_target=tgt, pop_ess_train=pop_ess(pe[D.tr], pb[D.tr]),
                                      pop_ess_eval=pop_ess(pe[D.ev], pb[D.ev]), min_pb_eval=float(pb[D.ev].min()),
                                      max_ratio_eval=float(w.max()), sha_pb=h(pb))
        rec["sha_target"] = h(pe)
        rec["sha_pcs"] = h(D.pcs)
        rec["train_idx_head"] = D.tr[:10].tolist()
        json.dump(rec, open(OUT / f"design_{scen}.json", "w"), indent=1)
        rows = []
        for pname, P in D.candidates().items():
            r = dict(policy=pname)
            for rw in ("logPAR10", "logPAR1", "speed"):
                Y = reward_matrix(D.d, rw)
                r[f"V_{rw}"] = float((P[D.ev] * Y[D.ev]).sum(1).mean())
            allsolved = D.ev[D.d["ok"][D.ev].all(1)]
            r["V_logPAR1_allsolved"] = float((P[allsolved] * reward_matrix(D.d, "logPAR1")[allsolved]).sum(1).mean())
            for ov, tgt in ESS_TARGETS_FINAL.items():
                k = rec["logging"][ov]["kappa"]
                r[f"popESS_{ov}"] = pop_ess(P[D.ev], D.logging(k)[D.ev])
            rows.append(r)
        Y = reward_matrix(D.d, D.primary)
        rows.append(dict(policy="(VBS oracle, reference only)", **{f"V_{rw}": float(reward_matrix(D.d, rw)[D.ev].max(1).mean())
                                                                   for rw in ("logPAR10", "logPAR1", "speed")}))
        pd.DataFrame(rows).to_csv(OUT / f"policy_truth_{scen}.csv", index=False)
        print(json.dumps(rec, indent=1)[:1500])
        print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
