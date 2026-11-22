"""
Step 7: validation. Why these models, and not simpler ones?

Forecast side
  1. Jumps:      are weekly gas changes fat-tailed enough to need a jump term?
  2. Benchmarks: do Model A / B forecast better than a random walk or a plain AR(1)?
                 Two out-of-sample tests: fit through 2021 -> forecast 2022, fit through 2025 -> forecast 2026.
  3. Revenue:    does revenue per hour need a random-walk part on top of the AR(1)? (variance ratio)
  4. Electricity: is the 0.12 gas elasticity stable across sub-periods?

Optimization side
  5. Recourse:   how much is it worth to set hours after seeing each month's prices? (vs. one fixed plan)
  6. VSS:        does planning with one average scenario pick the same car as the stochastic program?
  7. Risk measure: does a mean-variance rule pick the same cars as mean-CVaR?
  8. Behavior:   does the model's hours response to 2026 match what drivers actually did?

Run from repo root:  python src/step7_validate.py
Outputs: results/step7_*.csv, results/step7_log.txt, figures/step7_forecast_benchmarks.png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_gas_models as g4          # noqa: E402
import step4_scenarios as s4           # noqa: E402
import step5_model as m                # noqa: E402

RES = m.RES
FIG = g4.FIG
N = 5000


# ---------------------------------------------------------------- 1. jumps
def check_jumps():
    fit = g4.load_weekly()[g4.FIT_START:"2025-12-31"]
    seas = g4.seasonal(fit)
    x = np.log(fit) - seas.reindex(g4.week_of_year(fit.index)).values
    d = pd.DataFrame({"x": x.values, "t": np.arange(len(x)) / 52}, index=fit.index)
    d["dx"], d["xl"] = d.x.diff(), d.x.shift()
    d["dl"] = d.dx.shift()
    d = d.dropna()
    e = sm.OLS(d.dx, sm.add_constant(d[["t", "xl", "dl"]])).fit().resid      # same regression, no jump filter
    sd = e.std()
    z = np.abs(e / sd)
    p, _, J = g4.fit_jump(fit)
    out = pd.DataFrame([{
        "weeks": len(e), "excess_kurtosis": stats.kurtosis(e), "jarque_bera_p": stats.jarque_bera(e).pvalue,
        "share_beyond_3sd": (z > 3).mean(), "normal_share_beyond_3sd": 2 * stats.norm.sf(3),
        "largest_move_sd": z.max(), "jumps_flagged": len(J),
        "excess_kurtosis_after_jumps": stats.kurtosis(e[~e.index.isin(J.index)])}])
    out.to_csv(os.path.join(RES, "step7_jumps.csv"), index=False)
    r = out.iloc[0]
    print(f"  weekly residuals without a jump term: excess kurtosis {r.excess_kurtosis:.1f} (normal = 0), "
          f"Jarque-Bera p = {r.jarque_bera_p:.1e}")
    print(f"  {r.share_beyond_3sd:.1%} of weeks move more than 3 sd (normal: {r.normal_share_beyond_3sd:.2%}); "
          f"largest move {r.largest_move_sd:.1f} sd")
    print(f"  after removing the {int(r.jumps_flagged)} flagged jump weeks: excess kurtosis {r.excess_kurtosis_after_jumps:.1f}")
    return out


# ---------------------------------------------------------------- 2. benchmarks
def sim_random_walk(m_hist, months, n, seed):
    """Monthly log random walk, no drift, sd from monthly log changes in the fit window."""
    rng = np.random.default_rng(seed)
    sd = np.log(m_hist).diff().std()
    dates = pd.date_range(m_hist.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    lx = np.log(m_hist.iloc[-1]) + np.cumsum(sd * rng.standard_normal((months, n)), axis=0)
    return pd.DataFrame(np.exp(lx), index=dates)


def sim_ar1(m_hist, months, n, seed):
    """Monthly AR(1) in log price around a constant mean, normal shocks (a 'plain ARIMA' benchmark)."""
    rng = np.random.default_rng(seed)
    y = np.log(m_hist)
    o = sm.OLS(y.values[1:], sm.add_constant(y.values[:-1])).fit()
    c, rho, sd = o.params[0], o.params[1], np.sqrt(o.scale)
    dates = pd.date_range(m_hist.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    lx, out = np.full(n, y.iloc[-1]), np.empty((months, n))
    for k in range(months):
        lx = c + rho * lx + sd * rng.standard_normal(n)
        out[k] = lx
    return pd.DataFrame(np.exp(out), index=dates)


def crps(sample, y):
    """Continuous ranked probability score of a sample forecast (lower is better)."""
    s = np.sort(sample)
    n = len(s)
    return np.mean(np.abs(s - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2


def forecast_all(g, fit_end, months):
    fit = g[g4.FIT_START:fit_end]
    m_hist = fit.resample("MS").mean()
    crises = g4.CRISES
    g4.CRISES = [c for c in crises if c[1] <= fit_end[:7]]       # only shocks known at the forecast date
    try:
        pA, seasA, _ = g4.fit_jump(fit)
        pB, seasB = g4.fit_regime(fit)
    finally:
        g4.CRISES = crises
    simA = g4.simulate_jump(pA, seasA, fit.iloc[-1], fit.index[-1], months * 52 // 12 + 2, N,
                            prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=1).resample("MS").mean().iloc[:months]
    simB = g4.simulate_regime(pB, seasB, m_hist, months, N, seed=2)[0]
    return {"A (jump)": simA, "B (regime)": simB,
            "Random walk": sim_random_walk(m_hist, months, N, 3), "AR(1)": sim_ar1(m_hist, months, N, 4)}


def score(sims, act, label):
    rows = []
    for name, sim in sims.items():
        sim = sim.loc[act.index]
        lo, hi = sim.quantile(0.05, axis=1), sim.quantile(0.95, axis=1)
        rank = np.array([(sim.loc[t] < a).mean() for t, a in act.items()])
        peak = act.idxmax()
        rows.append({"test": label, "model": name,
                     "coverage_90": float(((act >= lo) & (act <= hi)).mean()),
                     "crps": float(np.mean([crps(sim.loc[t].values, a) for t, a in act.items()])),
                     "peak_month": peak.strftime("%Y-%m"), "peak_pct_rank": float((sim.loc[peak] < act[peak]).mean()),
                     "p_avg_at_least_actual": float((sim.mean() >= act.mean()).mean()),
                     "mean_pct_rank": float(rank.mean())})
    return pd.DataFrame(rows)


def check_benchmarks():
    g = g4.load_weekly()
    gm = g.resample("MS").mean()
    tests = [("2022 (fit to 2021)", "2021-12-31", gm["2022-01":"2022-12"]),
             ("2026 (fit to 2025)", "2025-12-31", gm["2026-01":"2026-10"])]
    out, sims_all = [], {}
    for label, end, act in tests:
        act = act.dropna()
        sims = forecast_all(g, end, 12)
        sims_all[label] = (sims, act)
        out.append(score(sims, act, label))
    out = pd.concat(out)
    out.to_csv(os.path.join(RES, "step7_benchmarks.csv"), index=False)
    print(out.round(3).to_string(index=False))
    plot_benchmarks(sims_all)
    return out


def plot_benchmarks(sims_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["A (jump)", "B (regime)", "Random walk", "AR(1)"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey="row")
    for i, (label, (sims, act)) in enumerate(sims_all.items()):
        for j, nm in enumerate(names):
            ax = axes[i, j]
            q = sims[nm].loc[act.index].quantile([0.05, 0.5, 0.95], axis=1).T
            ax.fill_between(q.index, q[0.05], q[0.95], color="#9bb7d4", alpha=0.6, label="90% band")
            ax.plot(q.index, q[0.5], color="#1f3f66", lw=1, label="median")
            ax.plot(act.index, act.values, color="#c0392b", lw=1.8, label="actual")
            ax.set_title(f"{nm}, {label[:4]}", fontsize=10)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("$/gal")
    axes[1, 0].set_ylabel("$/gal")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "step7_forecast_benchmarks.png"), dpi=150)


# ---------------------------------------------------------------- 3. revenue random walk
def check_revenue():
    g, e, mm = s4.monthly_inputs()
    links = s4.fit_links(g, e, mm)[0]
    r = pd.DataFrame({"fph": np.log(mm.fare / mm.seconds), "trips": np.log(mm.trips)}).dropna()["2022":"2025"]
    rev = r.fph + s4.BETA_U * (r.trips - r.trips.mean())
    d12 = rev.diff(12).dropna()
    obs_sd12 = d12.std()

    def sim_sd12(rw_annual, seed=0, n=4000):
        rng = np.random.default_rng(seed)
        T = 48
        u, w = np.zeros(n), np.zeros(n)
        out = np.empty((T, n))
        for k in range(T):
            u = links["rho_rev"] * u + links["sd_rev"] * rng.standard_normal(n)
            w = w + rw_annual / np.sqrt(12) * rng.standard_normal(n)
            out[k] = u + w
        return np.std(out[12:] - out[:-12]), np.std(np.exp(out).mean(0))

    a0, a1 = sim_sd12(0.0), sim_sd12(links["rw_rev_annual_sd"])
    rows = [{"version": "observed 2022-25", "sd_12m_change": obs_sd12, "sd_4yr_avg_index": np.nan},
            {"version": "AR(1) only", "sd_12m_change": a0[0], "sd_4yr_avg_index": a0[1]},
            {"version": f"AR(1) + random walk {links['rw_rev_annual_sd']:.1%}/yr (used)",
             "sd_12m_change": a1[0], "sd_4yr_avg_index": a1[1]}]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "step7_revenue.csv"), index=False)
    print(out.round(4).to_string(index=False))
    return out


# ---------------------------------------------------------------- 4. electricity elasticity stability
def check_electricity():
    g, e, _ = s4.monthly_inputs()
    d = pd.DataFrame({"gas": np.log(g), "elec": np.log(e)}).dropna()
    rows = []
    for a, b in (("2015", "2025"), ("2015", "2019"), ("2020", "2025"), ("2015", "2021")):
        y = d[a:b].diff(12).dropna()
        o = sm.OLS(y.elec, sm.add_constant(y.gas)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        lo, hi = o.conf_int().loc["gas"]
        rows.append({"period": f"{a}-{b}", "elasticity": o.params["gas"], "t": o.tvalues["gas"],
                     "ci_low": lo, "ci_high": hi, "months": len(y)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "step7_electricity.csv"), index=False)
    print(out.round(3).to_string(index=False))
    return out


# ---------------------------------------------------------------- 5. value of recourse
def solve_fixed_plan(x, typ, tb, sc, P):
    """Same as step5 build_and_solve at lambda = 0, but hours h[t,b] must be the same in every scenario
    (decided today, no adjusting to each month's prices). Borrowing w[s,t] still adjusts."""
    S, T = sc["gas"].shape
    B = len(tb)
    rev, mi, avail = tb.driver_rev_per_hour.values, tb.miles_per_hour.values, tb.hours_available.values
    r, Hmax = typ["reservation_wage"], typ["hours_max"]
    cpm, charge, fixed, outflow = m.car_terms(x, P, sc["gas"], sc["elec"], T)
    p = sc["prob"]
    cash_c = rev[None, None, :] * sc["rev"][:, :, None] - mi[None, None, :] * cpm[:, :, None]
    surp_c = cash_c - r * (1 + charge[None, :, None] * mi[None, None, :])
    nh, nw = T * B, S * T
    c = np.concatenate([-(p[:, None, None] * surp_c).sum(0).ravel(), m.KAPPA * np.repeat(p, T)])
    hidx = np.arange(nh).reshape(T, B)
    widx = nh + np.arange(nw).reshape(S, T)
    rows, cols, vals, rhs = [], [], [], []
    row = 0
    tcoef = 1 + charge[:, None] * mi[None, :]
    for t in range(T):
        rows += [row] * B
        cols += list(hidx[t])
        vals += list(tcoef[t])
        rhs.append(Hmax)
        row += 1
    for s in range(S):
        for t in range(T):
            rows += [row] * (B + 1)
            cols += list(hidx[t]) + [widx[s, t]]
            vals += list(-cash_c[s, t]) + [-1.0]
            rhs.append(-(typ["need"] + outflow))
            row += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(row, nh + nw))
    ub = np.concatenate([np.tile(avail, T), np.full(nw, np.inf)])
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=np.column_stack([np.zeros(nh + nw), ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    h = res.x[:nh].reshape(T, B)
    w = res.x[nh:].reshape(S, T)
    return -(surp_c * h[None]).sum((1, 2)) + m.KAPPA * w.sum(1) + fixed


def with_need(typ, P):
    typ = dict(typ)
    typ["need"] = max(m.NEED_SHARE * (typ["revenue_base"] - typ["miles_base"] * (3.80 / P["keep_gas"]["mpg"]
                      + P["keep_gas"]["maint"])) - P["keep_gas"]["insurance"] / 12, 0)
    return typ


def check_recourse_and_vss():
    P = m.load_params()
    tb = pd.read_csv(os.path.join(m.PARAMS, "time_blocks.csv"))
    types = m.driver_types()[0]
    sc = m.load_scen("A_spike")
    L = pd.read_csv(os.path.join(RES, "step5_losses_A_spike.csv"))
    mean_sc = {"gas": (sc["prob"] @ sc["gas"])[None], "elec": (sc["prob"] @ sc["elec"])[None],
               "rev": (sc["prob"] @ sc["rev"])[None], "prob": np.array([1.0])}
    rec, vss = [], []
    for tname in types.index:
        typ = with_need(types.loc[tname].to_dict(), P)
        rp, fx, ev = {}, {}, {}
        for x in m.CARS:
            lrec = L[(L.type == tname) & (L.car == x)].sort_values("scenario").loss.values
            lfix = solve_fixed_plan(x, typ, tb, sc, P)
            rp[x] = (sc["prob"] @ lrec, m.cvar(lrec, sc["prob"]))
            fx[x] = (sc["prob"] @ lfix, m.cvar(lfix, sc["prob"]))
            ev[x] = m.build_and_solve(x, typ, tb, mean_sc, P, 0.0)["loss"][0]
            rec.append({"type": tname, "car": x, "mean_loss_recourse": rp[x][0], "mean_loss_fixed_plan": fx[x][0],
                        "value_of_recourse_mean": fx[x][0] - rp[x][0],
                        "cvar_loss_recourse": rp[x][1], "cvar_loss_fixed_plan": fx[x][1],
                        "value_of_recourse_cvar": fx[x][1] - rp[x][1]})
        for lam in (0.0, 1.0):
            k = 0 if lam == 0 else 1
            best_rp = min(m.CARS, key=lambda x: rp[x][k])
            best_fx = min(m.CARS, key=lambda x: fx[x][k])
            best_ev = min(m.CARS, key=lambda x: ev[x])
            vss.append({"type": tname, "lam": lam, "car_stochastic": best_rp, "car_mean_value": best_ev,
                        "car_fixed_plan": best_fx,
                        "VSS": rp[best_ev][k] - rp[best_rp][k],
                        "loss_if_hours_not_adjusted": rp[best_fx][k] - rp[best_rp][k],
                        "mean_value_problem_loss": ev[best_ev], "stochastic_mean_loss": rp[best_rp][0]})
    rec, vss = pd.DataFrame(rec), pd.DataFrame(vss)
    rec.to_csv(os.path.join(RES, "step7_recourse.csv"), index=False)
    vss.to_csv(os.path.join(RES, "step7_vss.csv"), index=False)
    print("  value of adjusting hours each month (fixed plan minus recourse, $ over 4 years):")
    print(rec[["type", "car", "value_of_recourse_mean", "value_of_recourse_cvar"]].round(0).to_string(index=False))
    print("\n  car choice: stochastic program vs. one average scenario vs. fixed hours plan:")
    print(vss.round(0).to_string(index=False))
    return rec, vss


# ---------------------------------------------------------------- 7. mean-variance vs mean-CVaR
def check_risk_measure():
    L = pd.read_csv(os.path.join(RES, "step5_losses_A_spike.csv"))
    z = stats.norm.pdf(stats.norm.ppf(m.ALPHA)) / (1 - m.ALPHA)   # CVaR_0.9 of a normal = mean + 1.75 sd
    rows = []
    for tname, d in L.groupby("type"):
        for lam in (0.0, 0.5, 1.0):
            obj_cv, obj_mv, tail = {}, {}, {}
            for x, dx in d.groupby("car"):
                v, p = dx.loss.values, dx.prob.values
                mu = p @ v
                sd = np.sqrt(p @ (v - mu) ** 2)
                cv = m.cvar(v, p)
                obj_cv[x] = (1 - lam) * mu + lam * cv
                obj_mv[x] = mu + lam * z * sd
                tail[x] = (cv - mu) / (z * sd)
            rows.append({"type": tname, "lam": lam, "car_mean_cvar": min(obj_cv, key=obj_cv.get),
                         "car_mean_variance": min(obj_mv, key=obj_mv.get),
                         **{f"tail_ratio_{x}": tail[x] for x in m.CARS}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "step7_risk_measure.csv"), index=False)
    print("  tail_ratio = actual CVaR gap / CVaR gap if losses were normal (1 = normal tail)")
    print(out.round(2).to_string(index=False))
    return out


# ---------------------------------------------------------------- 8. hours response in 2026
def check_hours_2026():
    """Model: change in hours, 2026 actual inputs vs. the calibration year (indexes 1, gas at its Oct 2025-Sep 2026
    average, which is where the reservation wages were set).
    Data: change in trip-hours per active vehicle, Jan-Sep 2026 vs Jan-Sep 2025."""
    import step6_evaluate as e6
    act = e6.actual_2026()
    T = len(act)
    P = m.load_params()
    tb = pd.read_csv(os.path.join(m.PARAMS, "time_blocks.csv"))
    types = m.driver_types()[0]
    g_cal = g4.load_weekly()["2025-10":"2026-09"].mean()
    sc_a = {"gas": act.gas.values[None], "elec": act.elec_index.values[None], "rev": act.rev_index.values[None],
            "prob": np.array([1.0])}
    sc_n = {"gas": np.full((1, T), g_cal), "elec": np.ones((1, T)), "rev": np.ones((1, T)), "prob": np.array([1.0])}
    hrs_a = hrs_n = 0.0
    by_type = []
    for tname in types.index:
        typ = types.loc[tname].to_dict()
        typ["need"] = 0.0
        w = types.loc[tname, "share_of_cars"]
        ha = m.build_and_solve("keep_gas", typ, tb, sc_a, P, 0.0)["hours"][0]
        hn = m.build_and_solve("keep_gas", typ, tb, sc_n, P, 0.0)["hours"][0]
        hrs_a, hrs_n = hrs_a + w * ha, hrs_n + w * hn
        by_type.append(f"{tname} {hn:.0f} -> {ha:.0f} h")
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(s4.FETCHED, "trips", "tnp_trips_month_hour_*.csv"))])
    t["date"] = pd.to_datetime(t.month)
    sec = t.groupby("date").seconds.sum()
    v = pd.read_csv(os.path.join(s4.FETCHED, "tnp_vehicle_agg.csv"))
    v = v[v.n_trips > 0]
    v["date"] = pd.to_datetime(v.month_reported)
    veh = v.groupby("date").n_vehicles.sum()
    per = (sec / 3600 / veh).dropna()
    a, b = per["2026-01":"2026-09"], per["2025-01":"2025-09"]
    out = pd.DataFrame([{"model_change": hrs_a / hrs_n - 1, "data_change_trip_hours_per_vehicle": a.mean() / b.mean() - 1,
                         "data_months_2026": len(a), "data_months_2025": len(b),
                         "data_change_active_vehicles": veh["2026-01":"2026-09"].mean() / veh["2025-01":"2025-09"].mean() - 1}])
    out.to_csv(os.path.join(RES, "step7_hours_2026.csv"), index=False)
    print("  model, gas car, hours per month: " + ", ".join(by_type))
    print(out.round(3).to_string(index=False))
    return out


def main():
    print("1. Jumps in weekly gas changes")
    check_jumps()
    print("\n2. Forecast benchmarks (12-month out-of-sample, monthly averages)")
    check_benchmarks()
    print("\n3. Revenue per hour: sd of 12-month log change")
    check_revenue()
    print("\n4. Electricity elasticity to gas, by period")
    check_electricity()
    print("\n5-6. Value of recourse and value of the stochastic solution (Model A)")
    check_recourse_and_vss()
    print("\n7. Mean-variance vs. mean-CVaR")
    check_risk_measure()
    print("\n8. Hours response, 2026 vs. 2025")
    check_hours_2026()


if __name__ == "__main__":
    main()
