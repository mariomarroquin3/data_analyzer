"""
09_synthetic_control.py
════════════════════════════════════════════════════════════════════════
Layer 9 — Synthetic Control: The "Bukele Paradox" as a Quasi-Causal Case Study

MOTIVATION
──────────
Module 05's Leave-One-Country-Out check (item 4b) established that El
Salvador is the only country whose exclusion flips the sign of the
violence → voice_accountability relationship. That is a robustness
check, not a causal design: it tells us the panel-wide coefficient is
not driven by a general regional pattern, but it does not estimate what
would have happened to El Salvador's voice_accountability score absent
the post-2020 political-institutional shift (which includes, but is not
limited to, the 2022 state-of-exception security regime).

This module answers that narrower question with the Synthetic Control
Method (Abadie, Diamond & Hainmueller, 2010; Abadie, 2021): construct a
weighted combination of the other countries ("synthetic El Salvador")
that closely tracks El Salvador's own pre-treatment (2000-2020)
voice_accountability path, then compare the real post-2020 path with the
synthetic counterfactual.

WHAT "TREATMENT" MEANS HERE
───────────────────────────
Treatment onset is 2021 (first post-treatment year), the convention used
elsewhere in this project. Two facts matter for interpretation: the
formal state of exception was decreed in March 2022, but in May 2021 the
governing party's legislative supermajority had already dismissed the
Constitutional Chamber and the attorney general (Melendez-Sanchez, 2021).
The estimated gap therefore measures the post-2020 political-
institutional shift as a whole, not the exception regime in isolation.
A 2022-onset sensitivity run is reported below.

DESIGN CHOICES
──────────────
- Predictor set: the full pre-treatment OUTCOME PATH (2000-2020, excluding
  2001 which is missing WGI-wide) rather than a small set of covariates.
  With a modest donor pool, matching on a handful of covariates invites
  near-perfect (and therefore fragile) interpolation; matching on the
  full pre-trend is the more conservative choice (Doudchenko & Imbens,
  2016; Abadie, 2021, Section 4).
- Weights: w >= 0, sum(w) = 1, chosen to minimize squared pre-treatment
  prediction error (V = I in the Abadie et al. notation).
- Inference: placebo-in-space (Abadie et al., 2010). The identical
  procedure is re-run assigning the "treated" role to each other country
  in turn; El Salvador's post/pre RMSPE ratio is ranked against the
  placebo distribution (exact randomization p-value; minimum attainable
  value is 1/N for N countries).

ROBUSTNESS (all reported in the JSON export)
────────────────────────────────────────────
1. Leave-one-donor-out: refit after dropping each positively-weighted
   donor in turn (is the gap driven by a single donor?).
2. Placebo-in-time: pretend El Salvador was "treated" in earlier years
   (2008, 2012, 2016); a credible design should find no comparable gap.
3. Excluding donors with their own recent democratic deterioration
   (Nicaragua, Venezuela): a contaminated donor pulls the counterfactual
   DOWN, making the headline gap conservative -- this checks it.
4. 2022 onset (the formal decree date).
5. First stage: the same procedure applied to the homicide rate itself.
   El Salvador's 2015-16 peak lies outside what any convex combination of
   donors can reproduce, so a poor pre-treatment fit here is expected and
   is reported rather than hidden.

CAVEAT
──────
Even with ~20 donors the pool is small by synthetic-control standards
(most applications use 20-40+), pre-treatment fit should be inspected
directly, and the placebo-ranking p-value is coarse (minimum 1/N).

Reference
──────────────────────────────────────────────────────────────────────
Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
Methods for Comparative Case Studies. JASA, 105(490), 493-505.
Abadie, A. (2021). Using Synthetic Controls. Journal of Economic
Literature, 59(2), 391-425.
Doudchenko, N., & Imbens, G. (2016). NBER Working Paper No. 22791.
Melendez-Sanchez, M. (2021). Latin America Erupts: Millennial
Authoritarianism in El Salvador. Journal of Democracy, 32(3), 19-32.
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, bold, save_json,
    make_output_dirs, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)

BLUE  = "#2563EB"
RED_C = "#DC2626"
GREY  = "#9CA3AF"
AMBER = "#D97706"

COUNTRY_NAMES = {
    "ARG": "Argentina", "BLZ": "Belize", "BOL": "Bolivia", "BRA": "Brazil",
    "CHL": "Chile", "COL": "Colombia", "CRI": "Costa Rica",
    "DOM": "Dominican Republic", "ECU": "Ecuador", "GTM": "Guatemala",
    "GUY": "Guyana", "HND": "Honduras", "HTI": "Haiti", "JAM": "Jamaica",
    "MEX": "Mexico", "NIC": "Nicaragua", "PAN": "Panama", "PER": "Peru",
    "PRY": "Paraguay", "SLV": "El Salvador", "SUR": "Suriname",
    "TTO": "Trinidad and Tobago", "URY": "Uruguay", "VEN": "Venezuela",
}
nm = lambda c: COUNTRY_NAMES.get(c, c)

OUTCOME     = "voice_accountability"
TREATED     = "SLV"
TREAT_YEAR  = 2021                      # first post-treatment year
POST_LEN    = 4                          # 2021-2024
LAST_YEAR   = 2024
WGI_GAP     = 2001                       # missing for every country (biennial WGI before 2002)
DRIFT_DONORS = ["NIC", "VEN"]            # donors with their own recent democratic deterioration

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("09-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_enriched.csv"
df = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)

years_all = [y for y in range(2000, LAST_YEAR + 1) if y != WGI_GAP]
panel_voice = df.pivot_table(index=TIME_COL, columns=ENTITY_COL, values=OUTCOME).reindex(years_all)
panel_hom   = df.pivot_table(index=TIME_COL, columns=ENTITY_COL, values="homicide_rate").reindex(years_all)
COUNTRIES = sorted(panel_voice.columns.tolist())
print(ok(f"Panel: {len(COUNTRIES)} countries x {len(years_all)} years"))


def windows(onset: int):
    pre  = [y for y in years_all if y < onset]
    post = [y for y in years_all if onset <= y < onset + POST_LEN]
    return pre, post


# ════════════════════════════════════════════════════════════════════════
# CORE SCM SOLVER
# ════════════════════════════════════════════════════════════════════════
section("09-B — SYNTHETIC CONTROL SOLVER")


def fit_scm(treated: str, donors: list, panel_df: pd.DataFrame, pre: list, post: list,
            strict: bool = True) -> dict:
    """
    Fit donor weights w >= 0, sum(w) = 1 minimizing squared pre-treatment
    prediction error on the full outcome path.

    strict=True : donors with any missing value in the pre OR post window are
                  dropped (reported in 'dropped').
    strict=False: donors need complete data only in the PRE window; post-period
                  metrics use the years where every positively-weighted donor is
                  observed (used for the homicide series, where every country has
                  scattered missing years).
    """
    win = pre + post
    chk = win if strict else pre
    if panel_df.loc[chk, treated].isna().any():
        return {"treated": treated, "error": "treated unit has missing values"}
    ok_donors = [d for d in donors if d != treated and not panel_df.loc[chk, d].isna().any()]
    dropped = [d for d in donors if d != treated and d not in ok_donors]
    if not ok_donors:
        return {"treated": treated, "error": "no donor has complete data"}
    X1 = panel_df.loc[pre, treated].values.astype(float)
    X0 = panel_df.loc[pre, ok_donors].values.astype(float)
    J = len(ok_donors)

    def loss(w):
        return np.sum((X1 - X0 @ w) ** 2)

    res = minimize(loss, np.full(J, 1.0 / J), method="SLSQP", bounds=[(0.0, 1.0)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 2000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()

    pos = [(d, x) for d, x in zip(ok_donors, w) if x > 1e-4]
    pos_d = [d for d, _ in pos]
    pos_w = np.array([x for _, x in pos]) / sum(x for _, x in pos)
    synth = pd.Series(panel_df.loc[win, pos_d].values.astype(float) @ pos_w, index=win)
    actual = panel_df.loc[win, treated].astype(float)
    gap = actual - synth
    last = [y for y in post if not np.isnan(gap[y])][-1]
    pre_rmspe = float(np.sqrt(np.mean(gap[pre] ** 2)))
    post_rmspe = float(np.sqrt(np.nanmean(gap[post] ** 2)))
    return {
        "treated": treated,
        "n_donors": J,
        "dropped": dropped,
        "weights": {d: float(x) for d, x in zip(pos_d, pos_w)},
        "synthetic": synth,
        "actual": actual,
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "post_gap_avg": float(np.nanmean(gap[post])),
        "gap_last": float(gap[last]),
        "last_year": int(last),
        "ratio": post_rmspe / pre_rmspe if pre_rmspe > 1e-9 else np.nan,
        "converged": bool(res.success),
    }


def placebo_inference(panel_df, onset, units=None):
    pre, post = windows(onset)
    units = units or COUNTRIES
    fits = {}
    for c in units:
        f = fit_scm(c, [x for x in units if x != c], panel_df, pre, post)
        if "error" not in f:
            fits[c] = f
    ratios = {c: f["ratio"] for c, f in fits.items()}
    ranked = sorted(ratios.items(), key=lambda kv: -kv[1])
    rank = [i for i, (c, _) in enumerate(ranked, start=1) if c == TREATED][0]
    return fits, ratios, ranked, rank, rank / len(ratios)


# ── main specification ──────────────────────────────────────────────────
PRE, POST = windows(TREAT_YEAR)
fits, ratios, ranked, rank_slv, p_value = placebo_inference(panel_voice, TREAT_YEAR)
slv = fits[TREATED]
N_UNITS = len(fits)

subsection("El Salvador: synthetic weights")
for code, wi in sorted(slv["weights"].items(), key=lambda kv: -kv[1]):
    print(f"    {nm(code):<24} w = {wi:.3f}")
print(f"\n  Donors used: {slv['n_donors']}  (dropped for missing data: {slv['dropped'] or 'none'})")
print(f"  Pre-treatment RMSPE:  {slv['pre_rmspe']:.3f}")
print(f"  Post-treatment RMSPE: {slv['post_rmspe']:.3f}")
print(f"  Avg. post gap (actual - synthetic): {slv['post_gap_avg']:+.2f} points")
print(f"  Gap in {LAST_YEAR}: {slv['gap_last']:+.1f}  "
      f"(actual {slv['actual'][LAST_YEAR]:.1f} vs synthetic {slv['synthetic'][LAST_YEAR]:.1f})")
print(f"  Post/Pre RMSPE ratio: {slv['ratio']:.2f}")

section("09-C — PLACEBO-IN-SPACE INFERENCE")
for rk, (c, r) in enumerate(ranked[:8], start=1):
    mark = "  <-- El Salvador (actual case)" if c == TREATED else ""
    print(f"    #{rk:>2}  {nm(c):<24} ratio = {r:>6.2f}{mark}")
print(f"\n  El Salvador rank: {rank_slv} of {N_UNITS}")
print(f"  Exact randomization p-value: {p_value:.3f}  (minimum attainable = {1/N_UNITS:.3f})")

well_fit = {c: f for c, f in fits.items() if f["pre_rmspe"] <= 2 * slv["pre_rmspe"]}
ranked_wf = sorted(((c, f["ratio"]) for c, f in well_fit.items()), key=lambda kv: -kv[1])
rank_wf = [i for i, (c, _) in enumerate(ranked_wf, start=1) if c == TREATED][0]
p_value_wf = rank_wf / len(well_fit)
print(f"\n  Restricted to well-fitting placebos (pre-RMSPE <= 2x El Salvador's, N={len(well_fit)}):")
print(f"  El Salvador rank: {rank_wf} of {len(well_fit)}  ->  p = {p_value_wf:.3f}")

# ════════════════════════════════════════════════════════════════════════
# ROBUSTNESS
# ════════════════════════════════════════════════════════════════════════
section("09-D — ROBUSTNESS")
donors_all = [c for c in COUNTRIES if c != TREATED]

# 1. leave-one-donor-out
subsection("1. Leave-one-donor-out (drop each positively-weighted donor)")
loo = {}
for d in sorted(slv["weights"], key=lambda k: -slv["weights"][k]):
    f = fit_scm(TREATED, [x for x in donors_all if x != d], panel_voice, PRE, POST)
    loo[d] = {"gap_last": f["gap_last"], "post_gap_avg": f["post_gap_avg"],
              "pre_rmspe": f["pre_rmspe"], "ratio": f["ratio"]}
    print(f"    drop {nm(d):<22} gap {LAST_YEAR}: {f['gap_last']:+6.1f}   pre-RMSPE {f['pre_rmspe']:.2f}   ratio {f['ratio']:.1f}")
loo_gaps = [v["gap_last"] for v in loo.values()]
print(f"  Gap range across leave-one-donor-out fits: [{min(loo_gaps):+.1f}, {max(loo_gaps):+.1f}]")

# 2. placebo-in-time
subsection("2. Placebo-in-time (fake onset; El Salvador's real change comes later)")
pit = {}
for fake in (2008, 2012, 2016):
    pre_f, post_f = windows(fake)
    f = fit_scm(TREATED, donors_all, panel_voice, pre_f, post_f)
    pit[str(fake)] = {"post_gap_avg": f["post_gap_avg"], "pre_rmspe": f["pre_rmspe"], "ratio": f["ratio"]}
    print(f"    onset {fake}: avg post gap {f['post_gap_avg']:+5.2f}   pre-RMSPE {f['pre_rmspe']:.2f}   ratio {f['ratio']:.2f}")
print(f"    real onset {TREAT_YEAR}: avg post gap {slv['post_gap_avg']:+5.2f}   ratio {slv['ratio']:.2f}")

# 3. excluding donors with their own democratic drift
subsection("3. Excluding donors with their own democratic deterioration")
drift_present = [d for d in DRIFT_DONORS if d in donors_all]
f_nodrift = fit_scm(TREATED, [x for x in donors_all if x not in drift_present], panel_voice, PRE, POST)
print(f"    excluded: {drift_present}   gap {LAST_YEAR}: {f_nodrift['gap_last']:+.1f}   "
      f"pre-RMSPE {f_nodrift['pre_rmspe']:.2f}   ratio {f_nodrift['ratio']:.1f}")

# 4. 2022 onset
subsection("4. Onset 2022 (formal state-of-exception decree)")
pre22, post22 = windows(2022)
f22 = fit_scm(TREATED, donors_all, panel_voice, pre22, post22)
print(f"    avg post gap {f22['post_gap_avg']:+.2f}   gap {LAST_YEAR}: {f22['gap_last']:+.1f}   pre-RMSPE {f22['pre_rmspe']:.2f}")

# 5. first stage: homicides
subsection("5. First stage: synthetic control on the homicide rate")
fh = fit_scm(TREATED, donors_all, panel_hom, PRE, POST, strict=False)
if "error" in fh:
    print(f"    not estimable: {fh['error']}")
    hom_out = {"error": fh["error"]}
else:
    ly = fh["last_year"]
    print(f"    donors used: {fh['n_donors']}   pre-RMSPE {fh['pre_rmspe']:.2f} (per 100k)   "
          f"gap {ly}: {fh['gap_last']:+.1f}   actual {fh['actual'][ly]:.1f} vs synthetic {fh['synthetic'][ly]:.1f}")
    print("    NOTE: El Salvador's 2015-16 peak (~100/100k) is outside the donors' convex hull, so the")
    print("    pre-treatment fit is expected to be poor; interpret this as descriptive only.")
    hom_out = {"n_donors": fh["n_donors"], "pre_rmspe": fh["pre_rmspe"], "last_year": ly,
               "gap_last": fh["gap_last"], "actual_last": float(fh["actual"][ly]),
               "synthetic_last": float(fh["synthetic"][ly]), "weights": fh["weights"]}

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
section("09-E — FIGURES")

win = PRE + POST
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
axL.plot(win, slv["actual"].values, color=RED_C, marker="o", markersize=3, linewidth=1.8, label="El Salvador (actual)")
axL.plot(win, slv["synthetic"].values, color="black", linestyle="--", linewidth=1.8, label="Synthetic El Salvador")
axL.axvline(TREAT_YEAR - 0.5, color=AMBER, linestyle=":", linewidth=1.3)
axL.axvspan(TREAT_YEAR - 0.5, LAST_YEAR, color=AMBER, alpha=0.08)
axL.text(TREAT_YEAR + 1, 0.06, "Post-2020\nperiod", transform=axL.get_xaxis_transform(),
         ha="center", va="bottom", fontsize=7.5, color=AMBER)
axL.set_xlabel("Year"); axL.set_ylabel("Voice & accountability (0-100)")
axL.set_title("El Salvador: Actual vs. Synthetic Counterfactual", fontsize=10)
axL.legend(fontsize=8, loc="upper right")

for c, f in fits.items():
    if c == TREATED:
        continue
    axR.plot(win, (f["actual"] - f["synthetic"]).values, color=GREY, linewidth=0.8, alpha=0.6)
axR.plot(win, (slv["actual"] - slv["synthetic"]).values, color=RED_C, linewidth=2.2, label="El Salvador")
axR.axhline(0, color="black", linewidth=0.8)
axR.axvline(TREAT_YEAR - 0.5, color=AMBER, linestyle=":", linewidth=1.3)
axR.axvspan(TREAT_YEAR - 0.5, LAST_YEAR, color=AMBER, alpha=0.08)
axR.set_xlabel("Year"); axR.set_ylabel("Gap: actual - synthetic (points)")
axR.set_title(f"Placebo-in-Space (El Salvador vs. {N_UNITS-1} placebo gaps)\n"
              f"p = {p_value:.3f} (rank {rank_slv}/{N_UNITS})", fontsize=10)
axR.legend(fontsize=8, loc="lower left")
fig.suptitle("Synthetic Control: Was El Salvador's Voice-and-Accountability Decline "
             "Different from Its Counterfactual?", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "15_synthetic_control_bukele.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/15_synthetic_control_bukele.png"))

# robustness figure: leave-one-donor-out paths + placebo-in-time bars
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.plot(win, slv["actual"].values, color=RED_C, linewidth=2.2, label="El Salvador (actual)")
for d in loo:
    f = fit_scm(TREATED, [x for x in donors_all if x != d], panel_voice, PRE, POST)
    a1.plot(win, f["synthetic"].values, color=GREY, linewidth=1.0, alpha=0.8)
a1.plot(win, slv["synthetic"].values, color="black", linestyle="--", linewidth=1.8, label="Synthetic (all donors)")
a1.plot([], [], color=GREY, label="Leave-one-donor-out")
a1.axvline(TREAT_YEAR - 0.5, color=AMBER, linestyle=":", linewidth=1.3)
a1.set_xlabel("Year"); a1.set_ylabel("Voice & accountability (0-100)")
a1.set_title("Leave-one-donor-out synthetic paths", fontsize=10); a1.legend(fontsize=8)
labels = [f"fake {k}" for k in pit] + [f"real {TREAT_YEAR}"]
vals = [v["post_gap_avg"] for v in pit.values()] + [slv["post_gap_avg"]]
a2.bar(labels, vals, color=[GREY] * len(pit) + [RED_C])
a2.axhline(0, color="black", linewidth=0.8)
a2.set_ylabel(f"Avg. post-onset gap ({POST_LEN} yrs, points)")
a2.set_title("Placebo-in-time: fake onsets vs. real onset", fontsize=10)
fig2.tight_layout()
fig2.savefig(DIRS["figures"] / "15b_synthetic_control_robustness.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print(ok("Figure saved -> figures/15b_synthetic_control_robustness.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("09-F — EXPORT")
export = {
    "outcome": OUTCOME,
    "treated_unit": TREATED,
    "treatment_year": TREAT_YEAR,
    "pre_years": PRE,
    "post_years": POST,
    "el_salvador": {
        "weights": slv["weights"],
        "n_donors": slv["n_donors"],
        "dropped_donors": slv["dropped"],
        "pre_rmspe": slv["pre_rmspe"],
        "post_rmspe": slv["post_rmspe"],
        "post_gap_avg": slv["post_gap_avg"],
        "ratio": slv["ratio"],
        "actual_2024": float(slv["actual"][LAST_YEAR]),
        "synthetic_2024": float(slv["synthetic"][LAST_YEAR]),
        "gap_2024": slv["gap_last"],
    },
    "placebo_ratios": ratios,
    "rank_of_slv": rank_slv,
    "n_countries": N_UNITS,
    "p_value": p_value,
    "p_value_well_fitting_only": p_value_wf,
    "n_well_fitting": len(well_fit),
    "robustness": {
        "leave_one_donor_out": loo,
        "leave_one_donor_out_gap_range": [min(loo_gaps), max(loo_gaps)],
        "placebo_in_time": pit,
        "excluding_drift_donors": {"excluded": drift_present, "gap_2024": f_nodrift["gap_last"],
                                   "pre_rmspe": f_nodrift["pre_rmspe"], "ratio": f_nodrift["ratio"]},
        "onset_2022": {"post_gap_avg": f22["post_gap_avg"], "gap_2024": f22["gap_last"],
                       "pre_rmspe": f22["pre_rmspe"]},
        "first_stage_homicide": hom_out,
    },
}
save_json(export, DIRS["json"] / "09_synthetic_control.json")
print(f"\n{bold('Module 09 complete.')}")
