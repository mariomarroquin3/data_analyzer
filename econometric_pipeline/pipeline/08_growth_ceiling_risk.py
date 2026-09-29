"""
08_growth_ceiling_risk.py
════════════════════════════════════════════════════════════════════════
Layer 8 — Growth-Ceiling-at-Risk: Bayesian Hierarchical Quantile
Regression (MCMC)

Motivation
----------
An exploratory frequentist quantile regression (Koenker; documented in
the README's "Exploratory Finding: A 'Growth-Ceiling' Pattern" section)
found that lagged violence has essentially no effect near the median
of the GDP growth distribution, but a large, statistically significant,
and Leave-One-Country-Out-robust NEGATIVE effect on the UPPER tail
(q=0.90/0.95): a "growth-ceiling" pattern -- the mirror image of the
classic Growth-at-Risk (GaR) framing (Adrian, Boyarchenko & Giannone,
2019, "Vulnerable Growth"), which targets the LOWER tail of growth.

This module re-estimates that pattern with a proper Bayesian model
instead of treating the scratch analysis as final, for two reasons
specific to this project's constraints:

1. G=11 countries is small even for the LSDV country dummies the
   exploratory script used, which consume 10 degrees of freedom with
   no regularisation and can overfit noisy small-country intercepts.
   A hierarchical (partial-pooling) prior on country intercepts
   shrinks each country's intercept toward the grand mean by an amount
   the data itself determines (Gelman & Hill, 2007) -- exactly the
   regularisation a small-N panel needs and LSDV cannot provide.
2. A full posterior distribution at each quantile gives an honest,
   finite-sample uncertainty statement (a credible interval and
   P(beta<0 | data)) instead of a p-value computed from the asymptotic
   approximation this pipeline has flagged as unreliable at G=11
   everywhere else (see Module 04/05's small-cluster caveats).

Method: Bayesian Quantile Regression via the Asymmetric Laplace
Distribution (ALD)
-------------------------------------------------------
Minimising the classic "check" (pinball) loss at quantile tau is
equivalent to maximum-likelihood estimation under an Asymmetric
Laplace likelihood (Yu & Moyeed, 2001; Yu & Zhang, 2005, for the
version with an estimated scale sigma used here):

    log p(y | mu, sigma, tau) = log(tau*(1-tau)/sigma) - rho_tau(y-mu)/sigma
    rho_tau(u) = u * (tau - 1[u<0])

PyMC has no built-in ALD, so this is implemented directly as a
`pm.Potential` -- this turns an otherwise ordinary hierarchical linear
model into a Bayesian quantile regression simply by swapping the usual
Normal likelihood for this potential.

Model (per quantile tau, primary regressor x = homicide_rate_log_lag1):
    gdp_growth_it ~ ALD(tau, mu_it, sigma)
    mu_it        = alpha_i + beta*x_it + gamma*year_c_it

    alpha_i        = alpha_mu + alpha_sigma * z_i     [non-centered]
    z_i            ~ Normal(0, 1)                     [partial pooling]
    alpha_mu       ~ Normal(mean(gdp_growth), 10)
    alpha_sigma    ~ HalfNormal(5)
    beta, gamma    ~ Normal(0, 5)
    sigma          ~ HalfNormal(5)

The non-centered parameterisation (alpha_i as a deterministic function
of standardised z_i rather than alpha_i ~ Normal(alpha_mu, alpha_sigma)
directly) avoids the classic hierarchical "funnel" pathology under NUTS
(Betancourt & Girolami, 2015).

A secondary check repeats the SAME model with x = inst_avg (institution
quality) at the two headline quantiles only (q=0.90/0.95), to test
whether the ceiling effect is specific to violence or a more general
feature of any conditioning variable in this panel.

Growth-Ceiling-at-Risk scenario
-------------------------------------------------------
For q=0.90 and q=0.95, the posterior directly answers the applied
question: holding country and year at their average levels, how much
lower is the q-th percentile of achievable growth when lagged violence
sits at its empirical 90th percentile instead of its 10th percentile?
This "ceiling drop" is the module's headline risk quantity -- analogous
to a classic GaR statistic, but for the growth ceiling rather than the
growth floor.

Caveats
-------------------------------------------------------
- This is exploratory, motivated by a scratch analysis rather than a
  pre-registered hypothesis -- every result here should be read as
  suggestive, consistent with every other small-sample caveat in this
  pipeline.
- G=11 is small even for a hierarchical model: partial pooling
  regularises but cannot manufacture information the data does not
  contain. Priors are weakly informative, not flat.
- The ALD likelihood targets one quantile at a time; it is not a joint
  model of the growth distribution and does not itself guarantee that
  fitted quantiles are monotonic in tau (they are not required to be,
  and the "growth-ceiling" finding is precisely that the effect is
  non-monotonic: null at the median, negative only in the upper tail).
- MCMC diagnostics (R-hat, effective sample size, divergences) are
  reported for every quantile; any quantile failing standard
  diagnostic thresholds is flagged rather than silently reported as if
  it were reliable.

Literature
----------
Adrian, Boyarchenko & Giannone (2019) -- Vulnerable Growth, American
  Economic Review 109(4), 1263-1289.
Yu & Moyeed (2001) -- Bayesian quantile regression, Statistics &
  Probability Letters 54(4), 437-447.
Yu & Zhang (2005) -- A three-parameter asymmetric Laplace distribution
  and its extension, Communications in Statistics 34(9-10), 1867-1879.
Betancourt & Girolami (2015) -- Hamiltonian Monte Carlo for Hierarchical
  Models (the funnel of hell / non-centered parameterisation).
Gelman & Hill (2007) -- Data Analysis Using Regression and
  Multilevel/Hierarchical Models (partial pooling).
════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pymc as pm
import pytensor.tensor as pt
import arviz as az

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_plot_style, section, subsection, ok, warn, err, bold, sig_stars,
    save_json, make_output_dirs, ENTITY_COL, TIME_COL,
)

set_plot_style()
DIRS = make_output_dirs(Path(__file__).parent)
SEED = 42
np.random.seed(SEED)

DRAWS, TUNE, CHAINS = 1000, 1000, 4
HDI_PROB = 0.94

# ════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════
section("08-A — LOAD")

DATA_PATH = Path(__file__).parent / "panel_with_predictions.csv"
KEY_X = "homicide_rate_log_lag1"
raw = pd.read_csv(DATA_PATH)

df = (
    raw.dropna(subset=["gdp_growth", KEY_X, "year_c", ENTITY_COL, TIME_COL])
    .sort_values([ENTITY_COL, TIME_COL])
    .reset_index(drop=True)
)
COUNTRIES = sorted(df[ENTITY_COL].unique().tolist())
G = len(COUNTRIES)
country_idx_map = {c: i for i, c in enumerate(COUNTRIES)}
df["country_idx"] = df[ENTITY_COL].map(country_idx_map)
print(ok(f"Primary sample loaded: {len(df)} obs, {G} countries, key regressor = {KEY_X}"))

df_inst = (
    raw.dropna(subset=["gdp_growth", "inst_avg", "year_c", ENTITY_COL, TIME_COL])
    .sort_values([ENTITY_COL, TIME_COL])
    .reset_index(drop=True)
)
COUNTRIES_INST = sorted(df_inst[ENTITY_COL].unique().tolist())
G_INST = len(COUNTRIES_INST)
inst_idx_map = {c: i for i, c in enumerate(COUNTRIES_INST)}
df_inst["country_idx"] = df_inst[ENTITY_COL].map(inst_idx_map)
print(ok(f"Secondary (institutions) sample loaded: {len(df_inst)} obs, {G_INST} countries"))

print(f"""
  Motivation: an exploratory frequentist quantile regression (see the
  README's "Exploratory Finding" section) found lagged violence has a
  large, LOCO-robust NEGATIVE effect on the UPPER tail (q=0.90/0.95) of
  GDP growth, and essentially no effect near the median -- a
  "growth-ceiling" pattern, the mirror image of the classic lower-tail
  Growth-at-Risk framing. This module re-estimates that pattern with a
  hierarchical Bayesian model (partial pooling across {G} countries,
  full posterior uncertainty via MCMC) instead of LSDV dummies and
  asymptotic quantile-regression standard errors.
""")

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TAIL_QUANTILES = [0.90, 0.95]

# ════════════════════════════════════════════════════════════════════════
# MODEL: hierarchical Bayesian quantile regression (ALD likelihood)
# ════════════════════════════════════════════════════════════════════════
section("08-B — BAYESIAN HIERARCHICAL QUANTILE REGRESSION (MCMC)")


def fit_bayesian_qr(q: float, y: np.ndarray, x: np.ndarray, yearc: np.ndarray,
                     idx: np.ndarray, n_groups: int, seed: int = SEED) -> dict:
    """
    Hierarchical Bayesian quantile regression at quantile q via an
    Asymmetric Laplace Distribution (ALD) likelihood (Yu & Moyeed,
    2001), with a non-centered partial-pooling prior on country
    intercepts. Returns posterior summaries plus the raw InferenceData
    (kept in-process only -- stripped before JSON export).
    """
    with pm.Model():
        alpha_mu    = pm.Normal("alpha_mu", mu=float(y.mean()), sigma=10.0)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=5.0)
        z           = pm.Normal("z", mu=0.0, sigma=1.0, shape=n_groups)
        alpha       = pm.Deterministic("alpha", alpha_mu + z * alpha_sigma)

        beta  = pm.Normal("beta", mu=0.0, sigma=5.0)
        gamma = pm.Normal("gamma", mu=0.0, sigma=5.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)

        mu    = alpha[idx] + beta * x + gamma * yearc
        resid = y - mu
        rho   = pt.switch(resid >= 0, q * resid, (q - 1.0) * resid)
        pm.Potential("ald_loglik", pt.sum(pt.log(q * (1.0 - q) / sigma) - rho / sigma))

        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
            target_accept=0.9, random_seed=seed, progressbar=False,
        )

    beta_post = idata.posterior["beta"].values.flatten()
    beta_hdi  = az.hdi(beta_post, prob=HDI_PROB)
    n_div     = int(idata.sample_stats["diverging"].values.sum())
    r_hat     = float(az.rhat(idata, var_names=["beta"])["beta"].values)
    ess_bulk  = float(az.ess(idata, var_names=["beta"], method="bulk")["beta"].values)

    alpha_means = idata.posterior["alpha"].values.mean(axis=(0, 1))

    return {
        "q": q,
        "beta_mean":       float(beta_post.mean()),
        "beta_sd":         float(beta_post.std()),
        "beta_hdi_lo":     float(beta_hdi[0]),
        "beta_hdi_hi":     float(beta_hdi[1]),
        "beta_rhat":       r_hat,
        "beta_ess_bulk":   ess_bulk,
        "p_beta_negative": float((beta_post < 0).mean()),
        "n_divergences":   n_div,
        "n_obs":           int(len(y)),
        "n_countries":     int(n_groups),
        "idata":           idata,
    }


def fit_frequentist_qr(q: float, df_in: pd.DataFrame, x_col: str) -> dict:
    """LSDV frequentist benchmark (same spec, no partial pooling) for comparison."""
    country_dummies = pd.get_dummies(df_in[ENTITY_COL], prefix="c", drop_first=True).astype(float)
    Xf = pd.concat([
        df_in[[x_col, "year_c"]].reset_index(drop=True),
        country_dummies.reset_index(drop=True),
    ], axis=1)
    Xf = sm.add_constant(Xf)
    yf = df_in["gdp_growth"].reset_index(drop=True)
    res = sm.QuantReg(yf, Xf).fit(q=q, max_iter=5000)
    return {
        "coef": float(res.params[x_col]),
        "se":   float(res.bse[x_col]),
        "pval": float(res.pvalues[x_col]),
    }


Y      = df["gdp_growth"].values
X      = df[KEY_X].values
YEARC  = df["year_c"].values
IDX    = df["country_idx"].values

results_by_q = {}
for q in QUANTILES:
    subsection(f"q = {q:.2f}  (x = {KEY_X})")
    bayes = fit_bayesian_qr(q, Y, X, YEARC, IDX, G)
    freq  = fit_frequentist_qr(q, df, KEY_X)

    diag_ok = bayes["beta_rhat"] <= 1.01 and bayes["n_divergences"] == 0
    diag_msg = ok("MCMC diagnostics clean") if diag_ok else \
        warn(f"MCMC diagnostics borderline (R-hat={bayes['beta_rhat']:.4f}, "
             f"divergences={bayes['n_divergences']})")

    print(f"    Frequentist (LSDV):  coef = {freq['coef']:>8.4f}   "
          f"SE = {freq['se']:.4f}   p = {freq['pval']:.4f}  {sig_stars(freq['pval'])}")
    print(f"    Bayesian (hierarch): mean = {bayes['beta_mean']:>8.4f}   "
          f"94% HDI [{bayes['beta_hdi_lo']:.4f}, {bayes['beta_hdi_hi']:.4f}]   "
          f"P(beta<0|data) = {bayes['p_beta_negative']:.3f}")
    print(f"    {diag_msg}")

    results_by_q[q] = {"bayes": bayes, "freq": freq}

# ════════════════════════════════════════════════════════════════════════
# SECONDARY CHECK: is the ceiling effect specific to violence?
# ════════════════════════════════════════════════════════════════════════
section("08-C — SECONDARY CHECK: institutions (inst_avg) at the tail quantiles")
print("""
  The exploratory frequentist check also tested inst_avg as the
  conditioning variable and found nothing at any quantile (all
  p>0.27). Repeated here at the two headline tail quantiles under the
  same hierarchical Bayesian model, to confirm the ceiling effect is
  specific to violence rather than a generic feature of any regressor
  in this panel.
""")

Y_I     = df_inst["gdp_growth"].values
X_I     = df_inst["inst_avg"].values
YEARC_I = df_inst["year_c"].values
IDX_I   = df_inst["country_idx"].values

secondary_results = {}
for q in TAIL_QUANTILES:
    subsection(f"q = {q:.2f}  (x = inst_avg)")
    bayes = fit_bayesian_qr(q, Y_I, X_I, YEARC_I, IDX_I, G_INST)
    freq  = fit_frequentist_qr(q, df_inst, "inst_avg")
    print(f"    Frequentist (LSDV):  coef = {freq['coef']:>8.4f}   p = {freq['pval']:.4f}  {sig_stars(freq['pval'])}")
    print(f"    Bayesian (hierarch): mean = {bayes['beta_mean']:>8.4f}   "
          f"94% HDI [{bayes['beta_hdi_lo']:.4f}, {bayes['beta_hdi_hi']:.4f}]   "
          f"P(beta<0|data) = {bayes['p_beta_negative']:.3f}")
    if abs(bayes["p_beta_negative"] - 0.5) < 0.35:
        print(ok("  Posterior is roughly symmetric around zero -- no ceiling effect from "
                 "institutions, consistent with the exploratory frequentist check."))
    else:
        print(warn("  Posterior shows some directional signal for institutions too -- "
                    "re-examine the 'violence-specific' framing before over-stating it."))
    secondary_results[q] = {"bayes": bayes, "freq": freq}

# ════════════════════════════════════════════════════════════════════════
# GROWTH-CEILING-AT-RISK SCENARIO
# ════════════════════════════════════════════════════════════════════════
section("08-D — GROWTH-CEILING-AT-RISK SCENARIO")

x_lo = float(np.percentile(X, 10))
x_hi = float(np.percentile(X, 90))
print(f"  Low-violence scenario  (p10 of {KEY_X}): {x_lo:.3f}")
print(f"  High-violence scenario (p90 of {KEY_X}): {x_hi:.3f}")
print("  Both evaluated at the average country (alpha_mu) and average year (year_c=0).\n")

scenario_results = {}
for q in TAIL_QUANTILES:
    idata = results_by_q[q]["bayes"]["idata"]
    alpha_mu_post = idata.posterior["alpha_mu"].values.flatten()
    beta_post     = idata.posterior["beta"].values.flatten()

    ceiling_lo = alpha_mu_post + beta_post * x_lo
    ceiling_hi = alpha_mu_post + beta_post * x_hi
    drop       = ceiling_lo - ceiling_hi
    drop_hdi   = az.hdi(drop, prob=HDI_PROB)

    scenario_results[q] = {
        "x_low_violence":  x_lo,
        "x_high_violence": x_hi,
        "ceiling_low_violence_mean":  float(ceiling_lo.mean()),
        "ceiling_high_violence_mean": float(ceiling_hi.mean()),
        "ceiling_drop_mean":    float(drop.mean()),
        "ceiling_drop_hdi_lo":  float(drop_hdi[0]),
        "ceiling_drop_hdi_hi":  float(drop_hdi[1]),
        "p_drop_positive":      float((drop > 0).mean()),
        "ceiling_lo_samples":   ceiling_lo,
        "ceiling_hi_samples":   ceiling_hi,
    }
    r = scenario_results[q]
    print(f"  q={q}: ceiling at low violence  = {r['ceiling_low_violence_mean']:6.2f} pts of growth")
    print(f"         ceiling at high violence = {r['ceiling_high_violence_mean']:6.2f} pts of growth")
    print(f"         drop = {r['ceiling_drop_mean']:5.2f} pts  "
          f"94% HDI [{r['ceiling_drop_hdi_lo']:.2f}, {r['ceiling_drop_hdi_hi']:.2f}]  "
          f"P(drop>0|data) = {r['p_drop_positive']:.3f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
section("08-E — FIGURES")

# Figure 12: Frequentist vs Bayesian coefficient across quantiles
fig, ax = plt.subplots(figsize=(8, 5))
qs = QUANTILES
freq_coef = [results_by_q[q]["freq"]["coef"] for q in qs]
freq_se   = [results_by_q[q]["freq"]["se"] for q in qs]
bayes_mean = [results_by_q[q]["bayes"]["beta_mean"] for q in qs]
bayes_lo   = [results_by_q[q]["bayes"]["beta_hdi_lo"] for q in qs]
bayes_hi   = [results_by_q[q]["bayes"]["beta_hdi_hi"] for q in qs]

offset = 0.008
ax.errorbar([q - offset for q in qs], freq_coef, yerr=[1.645 * s for s in freq_se],
            fmt="o", color="#DC2626", capsize=4, label="Frequentist QR, LSDV (90% CI)")
ax.errorbar([q + offset for q in qs], bayes_mean,
            yerr=[np.array(bayes_mean) - np.array(bayes_lo), np.array(bayes_hi) - np.array(bayes_mean)],
            fmt="s", color="#2563EB", capsize=4, label=f"Bayesian hierarchical ({int(HDI_PROB*100)}% HDI)")
ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
ax.axvspan(0.875, 0.975, color="#DC2626", alpha=0.06, label="Growth-ceiling region (q=0.90/0.95)")
ax.set_xlabel("Quantile of GDP growth (τ)")
ax.set_ylabel(f"Coefficient on {KEY_X}")
ax.set_title("Growth-Ceiling Effect of Violence: Frequentist vs. Bayesian Hierarchical\n"
             "(11-country panel; partial pooling regularises country intercepts)",
             fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "12_growth_ceiling_bayesian.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/12_growth_ceiling_bayesian.png"))

# Figure 13: Growth-Ceiling-at-Risk scenario (posterior of ceiling under low vs high violence)
fig, axes = plt.subplots(1, len(TAIL_QUANTILES), figsize=(6.2 * len(TAIL_QUANTILES), 4.5))
if len(TAIL_QUANTILES) == 1:
    axes = [axes]
for ax, q in zip(axes, TAIL_QUANTILES):
    r = scenario_results[q]
    ax.hist(r["ceiling_lo_samples"], bins=40, density=True, alpha=0.55,
            color="#16A34A", label=f"Low violence (x=p10={r['x_low_violence']:.2f})")
    ax.hist(r["ceiling_hi_samples"], bins=40, density=True, alpha=0.55,
            color="#DC2626", label=f"High violence (x=p90={r['x_high_violence']:.2f})")
    ax.axvline(r["ceiling_low_violence_mean"], color="#16A34A", linestyle="--", linewidth=1.2)
    ax.axvline(r["ceiling_high_violence_mean"], color="#DC2626", linestyle="--", linewidth=1.2)
    ax.set_title(f"q = {q}\nCeiling drop = {r['ceiling_drop_mean']:.2f} pts "
                 f"(P>0 | data = {r['p_drop_positive']:.3f})", fontsize=10)
    ax.set_xlabel("Posterior predictive GDP growth ceiling (pts)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
fig.suptitle("Growth-Ceiling-at-Risk: Posterior Growth Ceiling, Low vs. High Violence Scenario",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "13_growth_ceiling_scenario.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(ok("Figure saved -> figures/13_growth_ceiling_scenario.png"))

# ════════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ════════════════════════════════════════════════════════════════════════
section("08-F — LATEX TABLE")

table_rows = []
for q in QUANTILES:
    b = results_by_q[q]["bayes"]
    f = results_by_q[q]["freq"]
    table_rows.append({
        "Quantile": q,
        "Freq. coef.": f["coef"],
        "Freq. p": f["pval"],
        "Bayes mean": b["beta_mean"],
        f"Bayes {int(HDI_PROB*100)}\\% HDI lo": b["beta_hdi_lo"],
        f"Bayes {int(HDI_PROB*100)}\\% HDI hi": b["beta_hdi_hi"],
        "P(beta<0)": b["p_beta_negative"],
        "R-hat": b["beta_rhat"],
    })
gcar_df = pd.DataFrame(table_rows).set_index("Quantile")
latex_gcar = gcar_df.round(4).to_latex(
    caption=("Growth-ceiling effect of lagged violence on GDP growth: frequentist "
             "quantile regression (LSDV) vs. Bayesian hierarchical quantile regression "
             "(ALD likelihood, partial pooling across 11 countries)."),
    label="tab:growth_ceiling_bayesian",
    column_format="l" + "r" * gcar_df.shape[1],
)
(DIRS["tables"] / "growth_ceiling_bayesian.tex").write_text(latex_gcar, encoding="utf-8")
print(ok("LaTeX table saved -> tables/growth_ceiling_bayesian.tex"))

# ════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════
section("08-G — EXPORT")

export = {
    "key_x": KEY_X,
    "n_obs": int(len(df)),
    "n_countries": int(G),
    "countries": COUNTRIES,
    "quantiles": {
        f"{q:.2f}": {
            "bayes": {k: v for k, v in results_by_q[q]["bayes"].items() if k != "idata"},
            "freq":  results_by_q[q]["freq"],
        }
        for q in QUANTILES
    },
    "secondary_institutions": {
        f"{q:.2f}": {
            "bayes": {k: v for k, v in secondary_results[q]["bayes"].items() if k != "idata"},
            "freq":  secondary_results[q]["freq"],
        }
        for q in TAIL_QUANTILES
    },
    "growth_ceiling_scenario": {
        f"{q:.2f}": {k: v for k, v in scenario_results[q].items()
                     if k not in ("ceiling_lo_samples", "ceiling_hi_samples")}
        for q in TAIL_QUANTILES
    },
}
save_json(export, DIRS["json"] / "08_growth_ceiling_risk.json")
print(f"\n{bold('Module 08 complete.')}")
