"""
Bayesian Models for Nobel Nomination Network Analysis
======================================================
All PyMC models and ArviZ diagnostic helpers used to replace frequentist
statistical tests in the original network_analysis.py.

Models:
1. bayesian_two_group_comparison — robust t-test (replaces Mann-Whitney U)
2. bayesian_beta_binomial — proportion comparison (replaces Fisher's exact)
3. bayesian_hierarchical_logistic — stratified analysis (replaces CMH test)
4. bayesian_logistic_single_predictor — single-predictor logistic (replaces point-biserial)
5. bayesian_logistic_regression — multi-predictor logistic (replaces sklearn LogReg)
6. posterior_predictive_roc — ROC with credible bands (replaces single ROC curve)
7. render_bayesian_diagnostics — reusable ArviZ display helper
"""

import numpy as np
import pymc as pm
import arviz as az
import streamlit as st
import matplotlib.pyplot as plt
import warnings


# ---------------------------------------------------------------------------
# 1. ROBUST T-TEST (replaces Mann-Whitney U)
# ---------------------------------------------------------------------------

def bayesian_two_group_comparison(group1, group2, metric_name,
                                   draws=1000, tune=500, chains=2):
    """
    Bayesian robust two-group comparison using Student-t likelihood.

    Parameters
    ----------
    group1, group2 : array-like
        Observations for group 1 (e.g., winners) and group 2 (e.g., near-misses).
    metric_name : str
        Name of the metric being compared (for labeling).
    draws, tune, chains : int
        MCMC sampling parameters.

    Returns
    -------
    idata : arviz.InferenceData
    summary : dict with keys diff_mean, hdi_low, hdi_high, p_greater, mu1_mean, mu2_mean
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    pooled = np.concatenate([g1, g2])
    pooled_mean = float(np.mean(pooled))
    pooled_std = float(np.std(pooled)) + 1e-6

    with pm.Model() as model:
        # Robust degrees of freedom
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # Group means
        mu1 = pm.Normal("mu_winners", mu=pooled_mean, sigma=pooled_std * 10)
        mu2 = pm.Normal("mu_near_misses", mu=pooled_mean, sigma=pooled_std * 10)

        # Shared scale
        sigma = pm.HalfNormal("sigma", sigma=pooled_std * 2)

        # Likelihoods
        pm.StudentT("y1", nu=nu, mu=mu1, sigma=sigma, observed=g1)
        pm.StudentT("y2", nu=nu, mu=mu2, sigma=sigma, observed=g2)

        # Derived quantity
        pm.Deterministic("diff", mu1 - mu2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=42, progressbar=False,
                              return_inferencedata=True)

    diff_samples = idata.posterior["diff"].values.flatten()
    hdi = az.hdi(idata, var_names=["diff"], hdi_prob=0.94)
    hdi_vals = hdi["diff"].values

    summary = {
        "diff_mean": float(np.mean(diff_samples)),
        "hdi_low": float(hdi_vals[0]),
        "hdi_high": float(hdi_vals[1]),
        "p_greater": float(np.mean(diff_samples > 0)),
        "mu1_mean": float(idata.posterior["mu_winners"].values.flatten().mean()),
        "mu2_mean": float(idata.posterior["mu_near_misses"].values.flatten().mean()),
        "metric_name": metric_name,
    }
    return idata, summary


# ---------------------------------------------------------------------------
# 2. BETA-BINOMIAL (replaces Fisher's exact test)
# ---------------------------------------------------------------------------

def bayesian_beta_binomial(n1, k1, n2, k2, group_names,
                            draws=1000, tune=500, chains=2):
    """
    Bayesian comparison of two proportions using Beta-Binomial model.

    Parameters
    ----------
    n1, k1 : int
        Total and successes for group 1.
    n2, k2 : int
        Total and successes for group 2.
    group_names : tuple of str
        Names for the two groups.
    draws, tune, chains : int
        MCMC sampling parameters.

    Returns
    -------
    idata : arviz.InferenceData
    summary : dict with keys p1_mean, p2_mean, diff_mean, hdi_low, hdi_high,
              p_greater, odds_ratio_mean
    """
    with pm.Model() as model:
        p1 = pm.Beta(f"p_{group_names[0]}", alpha=1, beta=1)
        p2 = pm.Beta(f"p_{group_names[1]}", alpha=1, beta=1)

        pm.Binomial(f"k_{group_names[0]}", n=n1, p=p1, observed=k1)
        pm.Binomial(f"k_{group_names[1]}", n=n2, p=p2, observed=k2)

        diff = pm.Deterministic("diff", p1 - p2)
        # Odds ratio with clipping to avoid division by zero
        pm.Deterministic("odds_ratio",
                         (p1 / pm.math.clip(1 - p1, 1e-8, 1)) /
                         (p2 / pm.math.clip(1 - p2, 1e-8, 1)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=42, progressbar=False,
                              return_inferencedata=True)

    diff_samples = idata.posterior["diff"].values.flatten()
    or_samples = idata.posterior["odds_ratio"].values.flatten()
    hdi = az.hdi(idata, var_names=["diff"], hdi_prob=0.94)
    hdi_vals = hdi["diff"].values

    summary = {
        "p1_mean": float(idata.posterior[f"p_{group_names[0]}"].values.flatten().mean()),
        "p2_mean": float(idata.posterior[f"p_{group_names[1]}"].values.flatten().mean()),
        "diff_mean": float(np.mean(diff_samples)),
        "hdi_low": float(hdi_vals[0]),
        "hdi_high": float(hdi_vals[1]),
        "p_greater": float(np.mean(diff_samples > 0)),
        "odds_ratio_mean": float(np.median(or_samples)),
        "group_names": group_names,
    }
    return idata, summary


# ---------------------------------------------------------------------------
# 3. HIERARCHICAL LOGISTIC REGRESSION (replaces CMH test)
# ---------------------------------------------------------------------------

def bayesian_hierarchical_logistic(strata_data,
                                    draws=1000, tune=500, chains=2):
    """
    Bayesian hierarchical logistic regression, stratified by nomination-count band.

    Parameters
    ----------
    strata_data : list of dict
        Each dict: {"group": 0/1, "outcome": 0/1, "stratum": int}.
        group=1 is the "treatment" group (e.g., campaign nominees).
    draws, tune, chains : int
        MCMC sampling parameters.

    Returns
    -------
    idata : arviz.InferenceData
    summary : dict with beta_mean, hdi_low, hdi_high, p_positive,
              odds_ratio_mean, stratum_effects
    """
    if not strata_data:
        return None, {"error": "No data for hierarchical model"}

    groups = np.array([d["group"] for d in strata_data])
    outcomes = np.array([d["outcome"] for d in strata_data])
    strata = np.array([d["stratum"] for d in strata_data])

    # Merge strata with <3 observations into neighboring strata
    unique_strata = sorted(set(strata))
    stratum_counts = {s: int(np.sum(strata == s)) for s in unique_strata}
    stratum_remap = {}
    merged_strata = []
    for s in unique_strata:
        if stratum_counts[s] < 3 and merged_strata:
            stratum_remap[s] = merged_strata[-1]
        else:
            merged_strata.append(s)
            stratum_remap[s] = s

    strata_remapped = np.array([stratum_remap[s] for s in strata])
    unique_final = sorted(set(strata_remapped))
    stratum_to_idx = {s: i for i, s in enumerate(unique_final)}
    stratum_idx = np.array([stratum_to_idx[s] for s in strata_remapped])
    n_strata = len(unique_final)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta_group = pm.Normal("beta_group", mu=0, sigma=1)

        if n_strata > 1:
            sigma_stratum = pm.HalfNormal("sigma_stratum", sigma=1)
            alpha_stratum = pm.Normal("alpha_stratum", mu=0,
                                       sigma=sigma_stratum, shape=n_strata)
            logit_p = alpha + beta_group * groups + alpha_stratum[stratum_idx]
        else:
            logit_p = alpha + beta_group * groups

        pm.Bernoulli("y", logit_p=logit_p, observed=outcomes)

        # Derived: odds ratio
        pm.Deterministic("odds_ratio", pm.math.exp(beta_group))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=42, progressbar=False,
                              target_accept=0.95,
                              return_inferencedata=True)

    beta_samples = idata.posterior["beta_group"].values.flatten()
    or_samples = idata.posterior["odds_ratio"].values.flatten()
    hdi = az.hdi(idata, var_names=["beta_group"], hdi_prob=0.94)
    hdi_vals = hdi["beta_group"].values

    summary = {
        "beta_mean": float(np.mean(beta_samples)),
        "hdi_low": float(hdi_vals[0]),
        "hdi_high": float(hdi_vals[1]),
        "p_positive": float(np.mean(beta_samples > 0)),
        "odds_ratio_mean": float(np.median(or_samples)),
    }

    # Stratum effects
    if n_strata > 1 and "alpha_stratum" in idata.posterior:
        stratum_effects = {}
        for i, s in enumerate(unique_final):
            vals = idata.posterior["alpha_stratum"].values[:, :, i].flatten()
            stratum_effects[s] = {
                "mean": float(np.mean(vals)),
                "hdi": [float(x) for x in az.hdi(vals, hdi_prob=0.94)],
            }
        summary["stratum_effects"] = stratum_effects

    return idata, summary


# ---------------------------------------------------------------------------
# 4. LOGISTIC REGRESSION — SINGLE PREDICTOR (replaces point-biserial)
# ---------------------------------------------------------------------------

def bayesian_logistic_single_predictor(predictor, outcome, name,
                                        draws=1000, tune=500, chains=2):
    """
    Bayesian logistic regression with a single predictor.

    Parameters
    ----------
    predictor : array-like
        Continuous predictor values.
    outcome : array-like
        Binary outcome (0/1).
    name : str
        Name of the predictor (for labeling).
    draws, tune, chains : int
        MCMC sampling parameters.

    Returns
    -------
    idata : arviz.InferenceData
    summary : dict with beta_mean, hdi_low, hdi_high, p_positive
    """
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcome, dtype=int)

    # Standardize
    x_mean, x_std = float(np.mean(x)), float(np.std(x)) + 1e-6
    x_z = (x - x_mean) / x_std

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal(f"beta_{name}", mu=0, sigma=1)

        logit_p = alpha + beta * x_z
        pm.Bernoulli("y", logit_p=logit_p, observed=y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=42, progressbar=False,
                              return_inferencedata=True)

    beta_key = f"beta_{name}"
    beta_samples = idata.posterior[beta_key].values.flatten()
    hdi = az.hdi(idata, var_names=[beta_key], hdi_prob=0.94)
    hdi_vals = hdi[beta_key].values

    summary = {
        "beta_mean": float(np.mean(beta_samples)),
        "hdi_low": float(hdi_vals[0]),
        "hdi_high": float(hdi_vals[1]),
        "p_positive": float(np.mean(beta_samples > 0)),
        "predictor_name": name,
        "x_mean": x_mean,
        "x_std": x_std,
    }
    return idata, summary


# ---------------------------------------------------------------------------
# 5. LOGISTIC REGRESSION — MULTI-PREDICTOR (replaces sklearn LogReg)
# ---------------------------------------------------------------------------

def bayesian_logistic_regression(X, y, feature_names, model_name,
                                  draws=1000, tune=500, chains=2):
    """
    Bayesian logistic regression with multiple predictors.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix (will be standardized internally).
    y : array-like, shape (n,)
        Binary outcome (0/1).
    feature_names : list of str
        Names for each feature column.
    model_name : str
        Identifier for this model configuration.
    draws, tune, chains : int
        MCMC sampling parameters.

    Returns
    -------
    idata : arviz.InferenceData (includes log_likelihood for WAIC/LOO)
    summary : dict with coefficients, intercept, model_name
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # Standardize features
    x_means = X.mean(axis=0)
    x_stds = X.std(axis=0) + 1e-6
    X_z = (X - x_means) / x_stds

    n_features = X_z.shape[1]

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)

        logit_p = alpha + pm.math.dot(X_z, beta)
        pm.Bernoulli("y", logit_p=logit_p, observed=y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=42, progressbar=False,
                              return_inferencedata=True,
                              idata_kwargs={"log_likelihood": True})

    beta_samples = idata.posterior["beta"].values  # shape (chains, draws, n_features)
    # Reshape to (total_draws, n_features)
    flat_beta = beta_samples.reshape(-1, n_features)

    coefficients = {}
    for i, fname in enumerate(feature_names):
        vals = flat_beta[:, i]
        hdi_vals = az.hdi(vals, hdi_prob=0.94)
        coefficients[fname] = {
            "mean": float(np.mean(vals)),
            "hdi_low": float(hdi_vals[0]),
            "hdi_high": float(hdi_vals[1]),
            "p_positive": float(np.mean(vals > 0)),
        }

    summary = {
        "model_name": model_name,
        "coefficients": coefficients,
        "intercept_mean": float(idata.posterior["alpha"].values.flatten().mean()),
        "x_means": x_means.tolist(),
        "x_stds": x_stds.tolist(),
        "feature_names": feature_names,
    }
    return idata, summary


# ---------------------------------------------------------------------------
# 6. POSTERIOR PREDICTIVE ROC (replaces single ROC curve)
# ---------------------------------------------------------------------------

def posterior_predictive_roc(idata, X, y, feature_names, n_draws=200):
    """
    Compute ROC curves from posterior draws for credible-band ROC plots.

    Parameters
    ----------
    idata : arviz.InferenceData
        From bayesian_logistic_regression.
    X : array-like, shape (n, p)
        Feature matrix (raw, will be standardized).
    y : array-like, shape (n,)
        True binary outcomes.
    feature_names : list of str
        Feature names.
    n_draws : int
        Number of posterior draws to use for ROC curves.

    Returns
    -------
    dict with:
        mean_fpr, mean_tpr : arrays for the mean ROC curve
        tpr_low, tpr_high : 94% credible band
        auc_samples : array of AUC values from posterior draws
        auc_mean, auc_hdi_low, auc_hdi_high : summary stats
    """
    from sklearn.metrics import roc_curve, auc as sk_auc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    x_means = X.mean(axis=0)
    x_stds = X.std(axis=0) + 1e-6
    X_z = (X - x_means) / x_stds

    alpha_samples = idata.posterior["alpha"].values.flatten()
    beta_samples = idata.posterior["beta"].values.reshape(-1, X_z.shape[1])

    total_draws = len(alpha_samples)
    draw_indices = np.random.RandomState(42).choice(total_draws, size=min(n_draws, total_draws),
                                                     replace=False)

    # Interpolate all ROC curves onto common FPR grid
    base_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for idx in draw_indices:
        a = alpha_samples[idx]
        b = beta_samples[idx]
        logits = a + X_z @ b
        proba = 1 / (1 + np.exp(-logits))

        fpr, tpr, _ = roc_curve(y, proba)
        auc_val = sk_auc(fpr, tpr)
        if not np.isnan(auc_val):
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc_val)

    if not tprs:
        return {"error": "No valid ROC curves from posterior draws"}

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    mean_tpr = np.mean(tprs, axis=0)
    tpr_low = np.percentile(tprs, 3, axis=0)
    tpr_high = np.percentile(tprs, 97, axis=0)

    auc_hdi = az.hdi(aucs, hdi_prob=0.94)

    return {
        "mean_fpr": base_fpr,
        "mean_tpr": mean_tpr,
        "tpr_low": tpr_low,
        "tpr_high": tpr_high,
        "auc_samples": aucs,
        "auc_mean": float(np.mean(aucs)),
        "auc_hdi_low": float(auc_hdi[0]),
        "auc_hdi_high": float(auc_hdi[1]),
    }


# ---------------------------------------------------------------------------
# 7. ArviZ DIAGNOSTICS DISPLAY HELPER
# ---------------------------------------------------------------------------

def render_bayesian_diagnostics(idata, var_names=None, key_prefix="diag"):
    """
    Render standard ArviZ diagnostics in Streamlit expanders.

    Parameters
    ----------
    idata : arviz.InferenceData
    var_names : list of str or None
        Variables to show diagnostics for. None = all.
    key_prefix : str
        Unique prefix for Streamlit widget keys.
    """
    with st.expander("Trace Plots"):
        fig_trace = plt.figure()
        try:
            axes = az.plot_trace(idata, var_names=var_names, compact=True)
            fig_trace = axes.ravel()[0].get_figure() if hasattr(axes, 'ravel') else plt.gcf()
            fig_trace.tight_layout()
            st.pyplot(fig_trace)
        except Exception as e:
            st.warning(f"Could not render trace plots: {e}")
        finally:
            plt.close("all")

    with st.expander("Convergence Diagnostics"):
        try:
            summary_df = az.summary(idata, var_names=var_names,
                                    hdi_prob=0.94)
            st.dataframe(summary_df, use_container_width=True)

            # Warnings for poor convergence
            if "r_hat" in summary_df.columns:
                bad_rhat = summary_df[summary_df["r_hat"] > 1.05]
                if len(bad_rhat) > 0:
                    st.warning(
                        f"R-hat > 1.05 for {len(bad_rhat)} parameter(s): "
                        f"{', '.join(bad_rhat.index.tolist()[:5])}. "
                        "Consider increasing tune/draws or reparameterizing."
                    )
            if "ess_bulk" in summary_df.columns:
                low_ess = summary_df[summary_df["ess_bulk"] < 100]
                if len(low_ess) > 0:
                    st.warning(
                        f"Low ESS (< 100) for {len(low_ess)} parameter(s). "
                        "Consider increasing draws."
                    )
        except Exception as e:
            st.warning(f"Could not compute diagnostics summary: {e}")


# ---------------------------------------------------------------------------
# 8. CONJUGATE BETA-BINOMIAL WITH BAYES FACTOR (no MCMC)
# ---------------------------------------------------------------------------

def conjugate_beta_binomial_bf(n1, k1, n2, k2, group_names,
                                prior_alpha=1, prior_beta=1, n_samples=200_000):
    """
    Exact conjugate Beta-Binomial inference with Bayes Factor.

    Computes separate Beta posteriors for two groups and a Bayes Factor
    comparing the hypothesis that the two groups have different rates (H1)
    vs. a common rate (H0), using analytic marginal likelihoods.

    Parameters
    ----------
    n1, k1 : int
        Total and successes for group 1.
    n2, k2 : int
        Total and successes for group 2.
    group_names : tuple of str
        Names for the two groups.
    prior_alpha, prior_beta : float
        Beta prior hyperparameters (default: uniform Beta(1,1)).
    n_samples : int
        Number of Monte Carlo draws for posterior summaries.

    Returns
    -------
    dict with:
        p1_alpha, p1_beta, p2_alpha, p2_beta : posterior parameters
        p1_mean, p2_mean : posterior means
        p1_hdi, p2_hdi : 94% HDI intervals
        diff_mean, diff_hdi : difference posterior summary
        bf10 : Bayes Factor (H1: separate rates vs H0: common rate)
        bf10_interpretation : Jeffreys scale label
        samples1, samples2 : posterior draws (for plotting)
    """
    from scipy.special import betaln
    from scipy.stats import beta

    a = prior_alpha
    b = prior_beta

    # Posterior parameters: Beta(a + k, b + n - k)
    p1_a, p1_b = a + k1, b + n1 - k1
    p2_a, p2_b = a + k2, b + n2 - k2

    # Log marginal likelihoods
    # H1 (separate rates): log P(data | H1) = log B(a+k1, b+n1-k1) - log B(a, b)
    #                                        + log B(a+k2, b+n2-k2) - log B(a, b)
    log_prior = betaln(a, b)
    log_m1 = (betaln(p1_a, p1_b) - log_prior) + (betaln(p2_a, p2_b) - log_prior)

    # H0 (common rate): pool both datasets under one Beta
    k_pool = k1 + k2
    n_pool = n1 + n2
    pool_a, pool_b = a + k_pool, b + n_pool - k_pool
    log_m0 = betaln(pool_a, pool_b) - log_prior

    # BF10 = P(data | H1) / P(data | H0)
    log_bf10 = log_m1 - log_m0
    bf10 = float(np.exp(np.clip(log_bf10, -500, 500)))

    # Jeffreys scale interpretation
    if bf10 < 1:
        interpretation = "evidence for equal rates"
    elif bf10 < 3:
        interpretation = "anecdotal"
    elif bf10 < 10:
        interpretation = "moderate"
    elif bf10 < 30:
        interpretation = "strong"
    elif bf10 < 100:
        interpretation = "very strong"
    else:
        interpretation = "decisive"

    # Posterior samples for summaries and plotting
    rng = np.random.RandomState(42)
    samples1 = beta.rvs(p1_a, p1_b, size=n_samples, random_state=rng)
    samples2 = beta.rvs(p2_a, p2_b, size=n_samples, random_state=rng)
    diff_samples = samples1 - samples2

    # HDI via percentiles (94%)
    def _hdi(samples, prob=0.94):
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        interval_size = int(np.ceil(prob * n))
        widths = sorted_s[interval_size:] - sorted_s[:n - interval_size]
        best = np.argmin(widths)
        return [float(sorted_s[best]), float(sorted_s[best + interval_size])]

    return {
        "p1_alpha": p1_a, "p1_beta": p1_b,
        "p2_alpha": p2_a, "p2_beta": p2_b,
        "p1_mean": float(p1_a / (p1_a + p1_b)),
        "p2_mean": float(p2_a / (p2_a + p2_b)),
        "p1_hdi": _hdi(samples1),
        "p2_hdi": _hdi(samples2),
        "diff_mean": float(np.mean(diff_samples)),
        "diff_hdi": _hdi(diff_samples),
        "p_greater": float(np.mean(diff_samples > 0)),
        "bf10": bf10,
        "bf10_interpretation": interpretation,
        "group_names": group_names,
        "samples1": samples1,
        "samples2": samples2,
        "n1": n1, "k1": k1, "n2": n2, "k2": k2,
    }
