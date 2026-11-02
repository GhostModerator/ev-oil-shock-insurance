"""
Step 4b: joint monthly scenarios of gas, electricity and rideshare revenue, then scenario reduction.

Links to gas (estimated on data through 2025):
  Electricity (IL commercial price, used to scale DC fast-charging price):
      12-month log change regressed on gas's 12-month log change, 2015-2025.
      Simulated as  log e_t = log e_0 + beta_e * (log gas_t - log gas_0) + AR(1) noise.
  Both also get a random-walk part so that shifts can persist (see constants below).
  Revenue per hour (fare per trip-hour x utilization), 2022-2025 (after COVID):
      The regression on gas is not causal here (trips recovered from COVID while gas was high in 2022),
      so the base case has no link to gas; revenue moves with its own AR(1) noise.
      REV_GAS_BETA lets us test a link: negative (2026: revenue per hour fell ~10% as gas rose)
      or positive (a fuel surcharge passes gas costs to riders).

Monte Carlo paths are reduced with k-means on the path vectors; each scenario gets the share of
paths in its cluster as probability.

Run from repo root:  python src/step4_scenarios.py
Outputs: data/scenarios/scen_{model}.csv (long format), data/params/joint_links.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_gas_models import (FIT_START, fit_jump, fit_regime, load_weekly, simulate_jump,  # noqa: E402
                              simulate_regime)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FETCHED = os.path.join(ROOT, "data", "fetched")
PARAMS = os.path.join(ROOT, "data", "params")
OUT = os.path.join(ROOT, "data", "scenarios")
os.makedirs(OUT, exist_ok=True)

N_PATHS = 5000
N_SCEN = 300
MONTHS = 48
BETA_U = 0.3          # utilization elasticity to demand (same as step 3)
REV_GAS_BETA = 0.0    # base: revenue per hour not linked to gas (sensitivity: -0.3, +0.2)
# Persistent parts (random walks), so a bad year can stay bad. Annual sd from 12-month log changes:
# revenue per hour moved +7.7%, -7.0%, +1.6%, +1.2% in 2022-25 (sd about 6%); IL commercial
# electricity rose 3.7%/yr on average in 2015-25 with a 12-month sd of about 7%, part of which is gas.


def monthly_inputs():
    g = load_weekly().resample("MS").mean()
    p = pd.read_csv(os.path.join(FETCHED, "eia_power_illinois_monthly.csv"))
    p["date"] = pd.to_datetime(p.period)
    e = p[p.sectorid.eq("COM")].set_index("date")["price"].sort_index()
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FETCHED, "trips", "tnp_trips_month_hour_*.csv"))])
    t["date"] = pd.to_datetime(t.month)
    m = t.groupby("date")[["trips", "fare", "seconds"]].sum()
    return g, e, m


def ar1(resid):
    r = resid.dropna()
    o = sm.OLS(r.values[1:], r.values[:-1]).fit()
    return float(o.params[0]), float(np.sqrt(o.scale))


def fit_links(g, e, m):
    links = {}
    # electricity: yoy elasticity, then AR(1) on the part not explained by gas
    d = pd.DataFrame({"gas": np.log(g), "elec": np.log(e)}).dropna()["2015":"2025"]
    y = d.diff(12).dropna()
    o = sm.OLS(y.elec, sm.add_constant(y.gas)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    links["beta_elec"], links["beta_elec_t"] = o.params["gas"], o.tvalues["gas"]
    res = d.elec - d.elec.rolling(12, center=True, min_periods=6).mean() - o.params["gas"] * (
        d.gas - d.gas.rolling(12, center=True, min_periods=6).mean())
    seas_e = res.groupby(res.index.month).mean()
    links["rho_elec"], links["sd_elec"] = ar1(res - seas_e.reindex(res.index.month).values)
    # revenue per online hour index: fare per trip-hour x utilization (trips ** BETA_U)
    r = pd.DataFrame({"gas": np.log(g), "fph": np.log(m.fare / m.seconds), "trips": np.log(m.trips)}).dropna()
    r = r["2022":"2025"]
    r["rev"] = r.fph + BETA_U * (r.trips - r.trips.mean())
    yr = r.diff(12).dropna()
    o2 = sm.OLS(yr.rev, sm.add_constant(yr.gas)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    links["beta_rev_estimated"], links["beta_rev_t"] = o2.params["gas"], o2.tvalues["gas"]
    links["beta_rev_used"] = REV_GAS_BETA
    seas_r = (r.rev - r.rev.rolling(12, center=True, min_periods=6).mean()).groupby(r.index.month).mean()
    seas_r -= seas_r.mean()
    dev = r.rev - seas_r.reindex(r.index.month).values
    dev = dev - dev.mean()
    links["rho_rev"], links["sd_rev"] = ar1(dev)
    links["rw_rev_annual_sd"] = r.rev.diff(12)["2022":"2025"].std() * 0.75     # part not from the AR(1) noise
    le = np.log(e)["2015":"2025"]
    links["elec_drift_annual"] = le.diff(12).mean()
    links["rw_elec_annual_sd"] = 0.05
    return links, seas_e, seas_r


def simulate_joint(gas, links, seas_e, seas_r, e0, seed=0):
    """gas: DataFrame months x paths. Returns gas, elec index (1 = Dec 2025 level), revenue index."""
    rng = np.random.default_rng(seed)
    n, T = gas.shape[1], gas.shape[0]
    lg = np.log(gas.values)
    lg0 = lg[0].mean()                    # the gas link is relative to the first simulated month
    ue, ur, we, wr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    rw_e, rw_r = links["rw_elec_annual_sd"] / np.sqrt(12), links["rw_rev_annual_sd"] / np.sqrt(12)
    elec, rev = np.empty((T, n)), np.empty((T, n))
    for k, mth in enumerate(gas.index):
        ue = links["rho_elec"] * ue + links["sd_elec"] * rng.standard_normal(n)
        ur = links["rho_rev"] * ur + links["sd_rev"] * rng.standard_normal(n)
        we = we + rw_e * rng.standard_normal(n)
        wr = wr + rw_r * rng.standard_normal(n)
        elec[k] = np.exp(links["elec_drift_annual"] * (k + 1) / 12 + links["beta_elec"] * (lg[k] - lg0)
                         + seas_e.get(mth.month, 0) + ue + we)
        rev[k] = np.exp(links["beta_rev_used"] * (lg[k] - lg0) + seas_r.get(mth.month, 0) + ur + wr)
    return elec, rev


def reduce(gas, elec, rev, k, seed=0):
    """k-means on standardized log paths; scenario = path closest to each cluster center."""
    F = np.hstack([np.log(gas.T), np.log(elec.T), np.log(rev.T)])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Z)
    reps, prob = [], []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        best = idx[np.argmin(((Z[idx] - km.cluster_centers_[c]) ** 2).sum(1))]
        reps.append(best)
        prob.append(len(idx) / len(Z))
    return np.array(reps), np.array(prob)


def make_scen(model="A_spike", rev_beta=None, n_scen=N_SCEN):
    """Scenario dict (as step5_model.load_scen returns) built in memory, e.g. with another revenue link."""
    g, e, m = monthly_inputs()
    links, seas_e, seas_r = fit_links(g, e, m)
    if rev_beta is not None:
        links["beta_rev_used"] = rev_beta
    fit = load_weekly()[FIT_START:"2025-12-31"]
    if model == "A_spike":
        pA, seasA = fit_jump(fit)[:2]
        gas = simulate_jump(pA, seasA, fit.iloc[-1], fit.index[-1], MONTHS * 52 // 12 + 2, N_PATHS,
                            prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=11).resample("MS").mean().iloc[:MONTHS]
    else:
        pB, seasB = fit_regime(fit)
        gas = simulate_regime(pB, seasB, fit.resample("MS").mean(), MONTHS, N_PATHS, seed=12)[0]
    elec, rev = simulate_joint(gas, links, seas_e, seas_r, None, seed=13)
    reps, prob = reduce(gas.values, elec, rev, n_scen, seed=14)
    return {"gas": gas.values[:, reps].T, "elec": elec[:, reps].T, "rev": rev[:, reps].T, "prob": prob}


def main():
    g, e, m = monthly_inputs()
    links, seas_e, seas_r = fit_links(g, e, m)
    print("Links to gas (fit through 2025):")
    print(f"  electricity: yoy elasticity {links['beta_elec']:.2f} (t={links['beta_elec_t']:.1f}), "
          f"AR(1) rho {links['rho_elec']:.2f}, sd {links['sd_elec']:.3f}; drift {links['elec_drift_annual']:.1%}/yr, "
          f"random walk {links['rw_elec_annual_sd']:.0%}/yr")
    print(f"  revenue/hour: estimated elasticity {links['beta_rev_estimated']:.2f} (t={links['beta_rev_t']:.1f}) "
          f"-> used {links['beta_rev_used']:.2f}; AR(1) rho {links['rho_rev']:.2f}, sd {links['sd_rev']:.3f}; "
          f"random walk {links['rw_rev_annual_sd']:.1%}/yr")
    pd.Series(links).to_csv(os.path.join(PARAMS, "joint_links.csv"), header=["value"])

    w = load_weekly()
    fit = w[FIT_START:"2025-12-31"]
    pA, seasA = fit_jump(fit)[:2]
    pB, seasB = fit_regime(fit)
    gasA = simulate_jump(pA, seasA, fit.iloc[-1], fit.index[-1], MONTHS * 52 // 12 + 2, N_PATHS,
                         prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=11).resample("MS").mean().iloc[:MONTHS]
    gasB = simulate_regime(pB, seasB, fit.resample("MS").mean(), MONTHS, N_PATHS, seed=12)[0]
    e0 = e["2025-12"].iloc[0]
    for name, gas in (("A_spike", gasA), ("B_regime", gasB)):
        elec, rev = simulate_joint(gas, links, seas_e, seas_r, e0, seed=13)
        reps, prob = reduce(gas.values, elec, rev, N_SCEN, seed=14)
        rows = []
        for s, (j, pr) in enumerate(zip(reps, prob)):
            rows.append(pd.DataFrame({"scenario": s, "prob": pr, "month": gas.index.strftime("%Y-%m"),
                                      "t": np.arange(MONTHS), "gas": gas.values[:, j],
                                      "elec_index": elec[:, j], "rev_index": rev[:, j]}))
        sc = pd.concat(rows)
        sc.to_csv(os.path.join(OUT, f"scen_{name}.csv"), index=False)
        # check: reduced set vs full Monte Carlo
        full_avg = gas.values.mean(0)
        red_avg = sc.groupby("scenario").gas.mean().values
        wq = lambda v, w, q: np.interp(q, np.cumsum(w[np.argsort(v)]) - w[np.argsort(v)] / 2, np.sort(v))
        print(f"\nModel {name}: {N_PATHS} paths -> {N_SCEN} scenarios")
        print(f"  4-yr avg gas: full MC mean {full_avg.mean():.2f}, p90 {np.quantile(full_avg, .9):.2f}, "
              f"p99 {np.quantile(full_avg, .99):.2f} | reduced mean {np.dot(prob, red_avg):.2f}, "
              f"p90 {wq(red_avg, prob, .9):.2f}, p99 {wq(red_avg, prob, .99):.2f}")
        print(f"  elec index 4-yr avg mean {elec.mean():.3f}; revenue index mean {rev.mean():.3f}, "
              f"sd of 4-yr avg {rev.mean(0).std():.3f}")
        corr = np.corrcoef(np.log(gas.values).mean(0), np.log(elec).mean(0))[0, 1]
        print(f"  corr(4-yr avg log gas, log elec) {corr:.2f}")


if __name__ == "__main__":
    main()
