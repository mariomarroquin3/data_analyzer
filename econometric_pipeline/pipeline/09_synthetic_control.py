"""
09_synthetic_control.py
════════════════════════════════════════════════════════════════════════
Layer 9 — Synthetic Control: The "Bukele Paradox" as a Causal Case Study

MOTIVATION
──────────
Module 05's Leave-One-Country-Out check (item 4b) established that El
Salvador is the only country whose exclusion flips the sign of the
violence → voice_accountability relationship. That is a robustness
check, not a causal design: it tells us the panel-wide coefficient is
not driven by a general regional pattern, but it does not estimate what
would have happened to El Salvador's voice_accountability score absent
its 2021-2024 state-of-exception security crackdown.

This module answers that narrower, better-identified question with the
Synthetic Control Method (Abadie, Diamond & Hainmueller, 2010; Abadie,
2021): construct a weighted combination of the other 10 countries
("synthetic El Salvador") that closely tracks El Salvador's own
pre-treatment (2000-2020) voice_accountability path, then compare the
real post-2020 path to the synthetic counterfactual.

DESIGN CHOICES
──────────────
- Predictor set: the full pre-treatment OUTCOME PATH (20 years, 2000-
  2020, excluding 2001 which is missing WGI-wide) rather than a small
  set of covariates. With only J=10 donor countries, matching on a
  handful of covariates invites near-perfect (and therefore fragile)
  interpolation; matching on the full pre-trend is the more
  conservative choice recommended for small donor pools (Doudchenko &
  Imbens, 2016; Abadie, 2021, Section 4).
- Weights: w >= 0, sum(w) = 1, chosen to minimize squared pre-treatment
  prediction error (equivalent to V = I in the Abadie et al. notation
  -- no covariate-importance tuning, to avoid overfitting with so few
  donors and so many candidate predictors).
- Inference: placebo-in-space (Abadie et al., 2010). The identical
  procedure is re-run assigning the "treated" role to each of the other
  10 countries in turn; El Salvador's post/pre RMSPE ratio is ranked
  against the resulting placebo distribution to obtain an exact,
  randomization-based p-value (1/11 if El Salvador has the single most
  extreme ratio).

CAVEAT
──────
J=10 donors is a very small pool by synthetic-control standards (the
original Abadie-Gardeazabal 1998 Basque study had 17 regions; most
applications use 20-40+). Pre-treatment fit should be inspected
directly (not just the RMSPE), and the placebo-ranking p-value is
necessarily coarse (minimum attainable value = 1/11 = 0.091).

Reference
──────────────────────────────────────────────────────────────────────
Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
Methods for Comparative Case Studies. JASA, 105(490), 493-505.
Abadie, A. (2021). Using Synthetic Controls. Journal of Economic
Literature, 59(2), 391-425.
Doudchenko, N., & Imbens, G. (2016). Balancing, Regression,
Difference-in-Differences and Synthetic Control Methods: A Synthesis.
NBER Working Paper No. 22791.
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
    "COL": "Colombia", "CRI": "Costa Rica", "DOM": "Dominican Republic",
    "ECU": "Ecuador", "GTM": "Guatemala", "HND": "Honduras",
    "MEX": "Mexico", "NIC": "Nicaragua", "PAN": "Panama",
    "PER": "Peru", "SLV": "El Salvador",
}

OUTCOME     = "voice_accountability"
TREATED     = "SLV"
TREAT_YEAR  = 2021                      # first post-treatment year (consistent with Module 05)
PRE_YEARS   = [y for y in range(2000, 2021) if y != 2001]   # 2001: WGI gap, all countries
POST_YEARS  = [2021, 2022, 2023, 2024]
ALL_YEARS   = PRE_YEARS + POST_YEARS

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("09-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_enriched.csv"
df = pd.read_csv(DATA_PATH).sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True)

panel = df.pivot_table(index=TIME_COL, columns=ENTITY_COL, values=OUTCOME)
panel = panel.reindex(ALL_YEARS)
COUNTRIES = sorted(panel.columns.tolist())
DONORS_SLV = [c for c in COUNTRIES if c != TREATED]
print(ok(f"Panel: {len(COUNTRIES)} countries x {len(ALL_YEARS)} years ({OUTCOME})"))
print(ok(f"Pre-treatment window: {PRE_YEARS[0]}-{PRE_YEARS[-1]} (excl. 2001, N={len(PRE_YEARS)})"))
print(ok(f"Post-treatment window: {POST_YEARS[0]}-{POST_YEARS[-1]}"))

# ════════════════════════════════════════════════════════════════════════
# CORE SCM SOLVER
# ════════════════════════════════════════════════════════════════════════
section("09-B — SYNTHETIC CONTROL SOLVER")


def fit_scm(treated_unit: str, donor_units: list, panel_df: pd.DataFrame) -> dict:
    """
    Fit donor weights w >= 0, sum(w) = 1 minimizing squared pre-treatment
    prediction error on the full outcome path (see module docstring).

    Returns dict with weights, synthetic path (all years), pre/post RMSPE.
    """
    X1 = panel_df.loc[PRE_YEARS, treated_unit].values.astype(float)
    X0 = panel_df.loc[PRE_YEARS, donor_units].values.astype(float)
    J  = len(donor_units)

    def loss(w):
        return np.sum((X1 - X0 @ w) ** 2)

    w0 = np.full(J, 1.0 / J)
    bounds = [(0.0, 1.0)] * J
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                    constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()

    full_donor_matrix = panel_df[donor_units].values.astype(float)
    synthetic = full_donor_matrix @ w
    actual    = panel_df[treated_unit].values.astype(float)

    years = panel_df.index.values
    pre_mask  = np.isin(years, PRE_YEARS)
    post_mask = np.isin(years, POST_YEARS)

    pre_rmspe  = float(np.sqrt(np.mean((actual[pre_mask]  - synthetic[pre_mask])  ** 2)))
    post_rmspe = float(np.sqrt(np.mean((actual[post_mask] - synthetic[post_mask]) ** 2)))
    post_gap_avg = float(np.mean(actual[post_mask] - synthetic[post_mask]))

    return {
        "treated": treated_unit,
        "weights": {d: float(wi) for d, wi in zip(donor_units, w) if wi > 1e-4},
        "synthetic_path": pd.Series(synthetic, index=years),
        "actual_path": pd.Series(actual, index=years),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "post_gap_avg": post_gap_avg,
        "ratio": post_rmspe / pre_rmspe if pre_rmspe > 1e-9 else np.nan,
        "converged": bool(res.success),
    }


slv_fit = fit_scm(TREATED, DONORS_SLV, panel)

subsection("El Salvador: synthetic weights")
sorted_w = sorted(slv_fit["weights"].items(), key=lambda kv: -kv[1])
for code, wi in sorted_w:
    print(f"    {COUNTRY_NAMES[code]:<22} w = {wi:.3f}")
print(f"\n  Pre-treatment RMSPE:  {slv_fit['pre_rmspe']:.3f}")
print(f"  Post-treatment RMSPE: {slv_fit['post_rmspe']:.3f}")
print(f"  Avg. post-treatment gap (actual - synthetic): {slv_fit['post_gap_avg']:+.2f} points")
print(f"  Post/Pre RMSPE ratio: {slv_fit['ratio']:.2f}")

actual_2020 = panel.loc[2020, TREATED]
actual_2024 = panel.loc[2024, TREATED]
synth_2024  = slv_fit["synthetic_path"].loc[2024]
print(f"\n  2024 actual:    {actual_2024:.1f}")
print(f"  2024 synthetic: {synth_2024:.1f}  (counterfactual absent the security crackdown)")
print(f"  Implied gap in 2024: {actual_2024 - synth_2024:+.1f} points")

# ════════════════════════════════════════════════════════════════════════
# PLACEBO-IN-SPACE INFERENCE
# ════════════════════════════════════════════════════════════════════════
section("09-C — PLACEBO-IN-SPACE INFERENCE")

placebo_fits = {}
for c in COUNTRIES:
    donors_c = [x for x in COUNTRIES if x != c]
    placebo_fits[c] = fit_scm(c, donors_c, panel)

ratios = {c: f["ratio"] for c, f in placebo_fits.items()}
ranked = sorted(ratios.items(), key=lambda kv: -kv[1])
rank_of_slv = [i for i, (c, _) in enumerate(ranked, start=1) if c == TREATED][0]
p_value = rank_of_slv / len(COUNTRIES)

subsection("Post/Pre RMSPE ratio, all countries as placebo-treated")
for rank, (c, r) in enumerate(ranked, start=1):
    marker = "  <-- El Salvador (actual case)" if c == TREATED else ""
    print(f"    #{rank:>2}  {COUNTRY_NAMES[c]:<22} ratio = {r:>6.2f}{marker}")

print(f"\n  El Salvador rank: {rank_of_slv} of {len(COUNTRIES)}")
print(f"  Exact randomization p-value: {p_value:.3f}  (minimum attainable = {1/len(COUNTRIES):.3f})")

# Sensitivity check: restrict placebo pool to countries whose own
# pre-treatment fit is at least as good as El Salvador's (Abadie et al.,
# 2010 robustness recommendation -- poorly-fitting placebos are not
# informative and can only make the test more conservative, never less).
well_fit = {c: f for c, f in placebo_fits.items() if f["pre_rmspe"] <= 2 * slv_fit["pre_rmspe"]}
ranked_wf = sorted(((c, f["ratio"]) for c, f in well_fit.items()), key=lambda kv: -kv[1])
rank_wf = [i for i, (c, _) in enumerate(ranked_wf, start=1) if c == TREATED][0]
p_value_wf = rank_wf / len(well_fit)
print(f"\n  Restricted to well-fitting placebos (pre-RMSPE <= 2x El Salvador's, "
      f"N={len(well_fit)}):")
print(f"  El Salvador rank: {rank_wf} of {len(well_fit)}  ->  p = {p_value_wf:.3f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("09-D — FIGURE")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

axL.plot(ALL_YEARS, slv_fit["actual_path"].values, color=RED_C, marker="o",
         markersize=3, linewidth=1.8, label="El Salvador (actual)")
axL.plot(ALL_YEARS, slv_fit["synthetic_path"].values, color="black", linestyle="--",
         linewidth=1.8, label="Synthetic El Salvador")
axL.axvline(TREAT_YEAR - 0.5, color=AMBER, linestyle=":", linewidth=1.3)
axL.axvspan(TREAT_YEAR - 0.5, ALL_YEARS[-1], color=AMBER, alpha=0.08)
axL.text(TREAT_YEAR + 1, 0.06, "State of\nexception", transform=axL.get_xaxis_transform(),
         ha="center", va="bottom", fontsize=7.5, color=AMBER)
axL.set_xlabel("Year")
axL.set_ylabel("Voice & accountability (0-100)")
axL.set_title("El Salvador: Actual vs. Synthetic Counterfactual", fontsize=10)
axL.legend(fontsize=8, loc="upper right")

for c in COUNTRIES:
    if c == TREATED:
        continue
    f = placebo_fits[c]
    gap = f["actual_path"] - f["synthetic_path"]
    axR.plot(ALL_YEARS, gap.values, color=GREY, linewidth=0.8, alpha=0.6)

slv_gap = slv_fit["actual_path"] - slv_fit["synthetic_path"]
axR.plot(ALL_YEARS, slv_gap.values, color=RED_C, linewidth=2.2, label="El Salvador")
axR.axhline(0, color="black", linewidth=0.8)
axR.axvline(TREAT_YEAR - 0.5, color=AMBER, linestyle=":", linewidth=1.3)
axR.axvspan(TREAT_YEAR - 0.5, ALL_YEARS[-1], color=AMBER, alpha=0.08)
axR.set_xlabel("Year")
axR.set_ylabel("Gap: actual - synthetic (points)")
axR.set_title(f"Placebo-in-Space (El Salvador vs. {len(COUNTRIES)-1} placebo gaps)\n"
              f"p = {p_value:.3f} (rank {rank_of_slv}/{len(COUNTRIES)})", fontsize=10)
axR.legend(fontsize=8, loc="lower left")

fig.suptitle('Synthetic Control: Was El Salvador\'s Voice-and-Accountability Decline '
             'Different from Its Counterfactual?', fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "15_synthetic_control_bukele.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/15_synthetic_control_bukele.png"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("09-E — EXPORT")

export = {
    "outcome": OUTCOME,
    "treated_unit": TREATED,
    "treatment_year": TREAT_YEAR,
    "pre_years": PRE_YEARS,
    "post_years": POST_YEARS,
    "el_salvador": {
        "weights": slv_fit["weights"],
        "pre_rmspe": slv_fit["pre_rmspe"],
        "post_rmspe": slv_fit["post_rmspe"],
        "post_gap_avg": slv_fit["post_gap_avg"],
        "ratio": slv_fit["ratio"],
        "actual_2024": float(actual_2024),
        "synthetic_2024": float(synth_2024),
        "gap_2024": float(actual_2024 - synth_2024),
    },
    "placebo_ratios": {c: f["ratio"] for c, f in placebo_fits.items()},
    "rank_of_slv": rank_of_slv,
    "n_countries": len(COUNTRIES),
    "p_value": p_value,
    "p_value_well_fitting_only": p_value_wf,
    "n_well_fitting": len(well_fit),
}
save_json(export, DIRS["json"] / "09_synthetic_control.json")
print(f"\n{bold('Module 09 complete.')}")
