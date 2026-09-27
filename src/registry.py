"""Named estimator configurations used in the paper (the only place where tuning constants are set).

Simulation study (experiments/final.py):
    DM, DR, DR-clip, RCF-DR (ablation: no one-step rescaling), RCF-DR-os, RCF-DR-VR, RCF-DR-VR-SN,
    RM-huber (ablation), RCF-DR-VR-tail (adaptive-logging supplement only),
    RCF-DR-VR-n13 (threshold-rate / contamination ablation only).
Real-data applications (real_data/aslib, real_data/kuairec):
    DM, IPW, SNIPW, DR, DR-clip, RCF-DR-os, RCF-DR-VR, RCF-DR-VR-SN.

Final tuning (frozen):
    RCF-DR-VR(-SN): tau_n = 0.5 * MAD(r) * n^(1/4), c_n = wbar * sqrt(n / ln n), one-step rescaling,
                    global self-normalisation for -SN.
    RCF-DR-os:      tau = 1.345 * MAD(r), c = 99% quantile of the weights, one-step rescaling.
    DR-clip:        c = wbar * sqrt(n) (Ionides 2008).
    RM-huber:       Huber location of the DR pseudo-outcome, tau_n = MAD * sqrt(n / ln n).
"""
from .estimators import dm, ips, snips, rm_dr, robust_dr
from .vr import vr_dr

K13 = 2.8 / 1000 ** (1 / 3)   # n^(1/3) threshold constant: tau = 2.8 MAD at n = 1000, as for kappa=0.5, n^(1/4)

SIM_ESTIMATORS = {
    "DM": (dm, {}),
    "DR": (robust_dr, dict(clip="none", tau="none")),
    "DR-clip": (robust_dr, dict(clip="ionides", tau="none")),
    "RCF-DR": (robust_dr, dict(clip="q99", tau="mad1.345")),
    "RCF-DR-os": (robust_dr, dict(clip="q99", tau="mad1.345", rescale=True)),
    "RCF-DR-VR": (vr_dr, dict(kappa=0.5, beta=0.25)),
    "RCF-DR-VR-SN": (vr_dr, dict(kappa=0.5, beta=0.25, sn="global")),
    "RM-huber": (rm_dr, dict(mean="huber", rate="sqrt", C=1.0)),
    "RCF-DR-VR-tail": (vr_dr, dict(kappa=0.5, beta=0.25, tail="exact")),
    "RCF-DR-VR-n13": (vr_dr, dict(kappa=K13, beta=1 / 3)),
}

REAL_DATA_ESTIMATORS = {
    "DM": (dm, {}),
    "IPW": (ips, {}),
    "SNIPW": (snips, {}),
    "DR": SIM_ESTIMATORS["DR"],
    "DR-clip": SIM_ESTIMATORS["DR-clip"],
    "RCF-DR-os": SIM_ESTIMATORS["RCF-DR-os"],
    "RCF-DR-VR": SIM_ESTIMATORS["RCF-DR-VR"],
    "RCF-DR-VR-SN": SIM_ESTIMATORS["RCF-DR-VR-SN"],
}
REAL_DATA_METHODS = list(REAL_DATA_ESTIMATORS)
