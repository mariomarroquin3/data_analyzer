"""
15_growth_ceiling_robustness.py
════════════════════════════════════════════════════════════════════════
Layer 15 — Robustness of the Growth-Ceiling Pattern

MOTIVATION
──────────
Module 08 finds that lagged violence lowers the UPPER tail of GDP growth
(q = 0.90 / 0.95) and, in the 18-country panel, raises its lower tail
(q = 0.05). With G = 18 and a few influential countries (El Salvador above
all) the natural questions are:

  1. Is the sign / significance carried by any single country? Leave-one-
     country-out (LOCO) refits of the frequentist LSDV quantile regression.
  2. How much of the Bayesian posterior depends on El Salvador? The
     hierarchical model of Module 08 re-fitted on the other 17 countries.

The frequentist p-values come from statsmodels' asymptotic QuantReg
covariance, which is unreliable with 18 country dummies and few observations
per tail; read the LOCO tables for sign and magnitude stability, not for exact
p-values.

CONSISTENCY CHECK
─────────────────
The Bayesian model below is a copy of Module 08's `fit_bayesian_qr`. To make
sure the copy is identical, the full-sample q = 0.90 fit is re-run and must
reproduce Module 08's posterior mean to numerical precision (the sampler is
deterministic for a fixed seed).
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from utils import set_plot_style, section, subsection, ok, warn, bold, save_json, make_output_dirs, ENTITY_COL, TIME_COL

set_plot_style()
HERE = Path(__file__).parent
DIRS = make_output_dirs(HERE)
BLUE, RED_C, GREY = "#2563EB", "#DC2626", "#9CA3AF"
SEED, DRAWS, TUNE, CHAINS, HDI_PROB = 42, 1000, 1000, 4, 0.94
KEY = "homicide_rate_log_lag1"
QUANTILES = [0.05, 0.90, 0.95]
EXCLUDE = "SLV"

section("15-A — LOAD")
raw = pd.read_csv(HERE / "panel_with_predictions.csv")
df = (raw.dropna(subset=["gdp_growth", KEY, "year_c", ENTITY_COL, TIME_COL])
      .sort_values([ENTITY_COL, TIME_COL]).reset_index(drop=True))
countries = sorted(df[ENTITY_COL].unique())
print(ok(f"{len(df)} country-years, {len(countries)} countries"))
with open(HERE / "json" / "08_growth_ceiling_risk.json", encoding="utf-8") as f:
    m08 = json.load(f)


def freq_qr(d, q):
    dm = pd.get_dummies(d[ENTITY_COL], prefix="c", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[[KEY, "year_c"]].reset_index(drop=True), dm.reset_index(drop=True)], axis=1))
    r = sm.QuantReg(d["gdp_growth"].reset_index(drop=True), X).fit(q=q, max_iter=5000)
    return float(r.params[KEY]), float(r.pvalues[KEY])


def bayes_qr(d, q):
    """Copy of Module 08's hierarchical ALD quantile regression (same priors, seed and sampler settings)."""
    idx = pd.factorize(d[ENTITY_COL])[0]
    n_groups = int(idx.max() + 1)
    y, x, yc = d["gdp_growth"].to_numpy(float), d[KEY].to_numpy(float), d["year_c"].to_numpy(float)
    with pm.Model():
        alpha_mu = pm.Normal("alpha_mu", mu=float(y.mean()), sigma=10.0)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=5.0)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=n_groups)
        alpha = pm.Deterministic("alpha", alpha_mu + z * alpha_sigma)
        beta = pm.Normal("beta", mu=0.0, sigma=5.0)
        gamma = pm.Normal("gamma", mu=0.0, sigma=5.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        mu = alpha[idx] + beta * x + gamma * yc
        resid = y - mu
        rho = pt.switch(resid >= 0, q * resid, (q - 1.0) * resid)
        pm.Potential("ald_loglik", pt.sum(pt.log(q * (1.0 - q) / sigma) - rho / sigma))
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9,
                          random_seed=SEED, progressbar=False)
    b = idata.posterior["beta"].values.flatten()
    h = az.hdi(b, prob=HDI_PROB)
    return {"beta_mean": float(b.mean()), "beta_hdi_lo": float(h[0]), "beta_hdi_hi": float(h[1]),
            "p_beta_negative": float((b < 0).mean()), "n_obs": int(len(d)), "n_countries": n_groups,
            "n_divergences": int(idata.sample_stats["diverging"].values.sum())}


# ════════════════════════════════════════════════════════════════════════
# 1. FREQUENTIST LEAVE-ONE-COUNTRY-OUT
# ════════════════════════════════════════════════════════════════════════
section("15-B — FREQUENTIST LEAVE-ONE-COUNTRY-OUT (LSDV quantile regression)")
loco = {}
for q in QUANTILES:
    full_c, full_p = freq_qr(df, q)
    rows = [{"dropped": c, **dict(zip(("coef", "p"), freq_qr(df[df[ENTITY_COL] != c], q)))} for c in countries]
    r = pd.DataFrame(rows)
    sign = np.sign(full_c)
    same = int((np.sign(r["coef"]) == sign).sum())
    sig = int(((np.sign(r["coef"]) == sign) & (r["p"] < 0.05)).sum())
    weak = r.loc[r["p"].idxmax()]
    loco[f"{q:.2f}"] = {"full_coef": full_c, "full_p": full_p, "n_same_sign": same, "n_same_sign_and_p05": sig,
                        "n_fits": len(r), "coef_min": float(r["coef"].min()), "coef_max": float(r["coef"].max()),
                        "weakest_dropped": str(weak["dropped"]), "weakest_coef": float(weak["coef"]),
                        "weakest_p": float(weak["p"]), "by_country": r.to_dict(orient="records")}
    print(f"    q={q:.2f}: full {full_c:+.2f} (p={full_p:.3f}) | same sign {same}/{len(r)} | same sign and p<0.05 {sig}/{len(r)} | "
          f"range {r['coef'].min():+.2f} to {r['coef'].max():+.2f} | weakest: drop {weak['dropped']} ({weak['coef']:+.2f}, p={weak['p']:.3f})")

# ════════════════════════════════════════════════════════════════════════
# 2. BAYESIAN WITHOUT EL SALVADOR
# ════════════════════════════════════════════════════════════════════════
section(f"15-C — HIERARCHICAL BAYESIAN MODEL WITHOUT {EXCLUDE}")
subsection("Consistency check: the full-sample fit must reproduce Module 08")
chk = bayes_qr(df, 0.90)
ref = m08["quantiles"]["0.90"]["bayes"]["beta_mean"]
assert abs(chk["beta_mean"] - ref) < 1e-6, f"model copy differs from Module 08 ({chk['beta_mean']} vs {ref})"
print(ok(f"q=0.90 full sample: posterior mean {chk['beta_mean']:+.4f} = Module 08 ({ref:+.4f})"))

without = {}
d17 = df[df[ENTITY_COL] != EXCLUDE]
for q in QUANTILES:
    b = bayes_qr(d17, q)
    full = m08["quantiles"][f"{q:.2f}"]["bayes"]
    b["full_sample_mean"], b["full_sample_p_beta_negative"] = full["beta_mean"], full["p_beta_negative"]
    without[f"{q:.2f}"] = b
    print(f"    q={q:.2f}: posterior mean {b['beta_mean']:+.2f} (all 18: {full['beta_mean']:+.2f}) | 94% HDI [{b['beta_hdi_lo']:+.2f}, {b['beta_hdi_hi']:+.2f}] | "
          f"P(beta<0) = {b['p_beta_negative']:.3f} | divergences {b['n_divergences']}", flush=True)

# ════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════
section("15-D — FIGURE")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
for a, q in zip(ax, QUANTILES):
    r = pd.DataFrame(loco[f"{q:.2f}"]["by_country"]).sort_values("coef")
    cols = [RED_C if c == EXCLUDE else (BLUE if p < 0.05 else GREY) for c, p in zip(r["dropped"], r["p"])]
    a.barh(range(len(r)), r["coef"], color=cols)
    a.set_yticks(range(len(r))); a.set_yticklabels(r["dropped"], fontsize=7)
    a.axvline(loco[f"{q:.2f}"]["full_coef"], color="black", ls="--", lw=1, label="all 18 countries")
    a.axvline(0, color="black", lw=0.6)
    a.set_title(f"q = {q:.2f}", fontsize=10)
    a.set_xlabel("coefficient of lagged log homicides\n(country dropped on the axis)")
    if q == QUANTILES[0]:
        a.legend(fontsize=8)
fig.suptitle("Leave-one-country-out, LSDV quantile regression (blue: p<0.05; red: El Salvador dropped; grey: p>=0.05)", fontsize=10)
fig.tight_layout()
fig.savefig(DIRS["figures"] / "20_growth_ceiling_loco.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/20_growth_ceiling_loco.png"))

save_json({
    "excluded_country": EXCLUDE,
    "frequentist_loco": loco,
    "bayesian_without_excluded": without,
    "consistency_check": {"module08_posterior_mean_q090": ref, "refit_posterior_mean_q090": chk["beta_mean"]},
    "caveat": "frequentist p-values are asymptotic QuantReg values and unreliable with 18 country dummies",
}, DIRS["json"] / "15_growth_ceiling_robustness.json")
print(bold("\nModule 15 complete."))
