"""
Step 4: gas price models.

Model A, "spike": mean-reverting jump model on weekly Chicago gas.
    x_t = log price minus a week-of-year seasonal factor
    dx_t = c + g*t - kappa*x_{t-1} + phi*dx_{t-1} + sigma*e_t + J_t
    J_t = jump with prob p_up (size N(mu_up, sd_up)) or p_down (size N(mu_down, sd_down))
    Fit by a robust filter: OLS, flag residuals beyond 3 robust SDs as jumps, refit, repeat.

Model B, "regime": calm and crisis regimes on monthly Chicago gas.
    Crisis months are labeled from supply shocks known before 2026 (CRISES below), not estimated.
    In each regime: dx = a_s + b_s * gap_{t-1} + phi_s * dx_{t-1} + sigma_s * e,
    gap = deseasonalized log price minus the long-run trend. A crisis pulls prices toward
    trend + premium (premium = -a/b). Entry and exit rates come from how often crises started
    and how long they lasted.

Run from repo root:  python src/step4_gas_models.py
Outputs: data/params/gas_model_jump.csv, data/params/gas_model_regime.csv,
         data/params/gas_backtest_2026.csv, figures/gas_backtest_2026.png
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FETCHED = os.path.join(ROOT, "data", "fetched")
PARAMS = os.path.join(ROOT, "data", "params")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load_weekly():
    g = pd.read_csv(os.path.join(FETCHED, "eia_gas_chicago_weekly.csv"), parse_dates=["date"]).set_index("date").iloc[:, 0]
    return g.dropna()


def week_of_year(idx):
    return np.minimum(idx.isocalendar().week.values, 52)


def seasonal(g):
    """Week-of-year factor in log price: average gap from a centered 1-year mean, smoothed over 5 weeks."""
    ly = np.log(g)
    r = ly - ly.rolling(53, center=True, min_periods=30).mean()
    s = r.groupby(week_of_year(g.index)).mean().rolling(5, center=True, min_periods=1).mean()
    return s - s.mean()


def fit_jump(g, z=3.0):
    seas = seasonal(g)
    x = np.log(g) - seas.reindex(week_of_year(g.index)).values
    t = np.arange(len(x)) / 52
    d = pd.DataFrame({"x": x.values, "t": t}, index=g.index)
    d["dx"], d["xl"] = d.x.diff(), d.x.shift()
    d["dl"] = d.dx.shift()
    d = d.dropna()
    X = sm.add_constant(d[["t", "xl", "dl"]])
    jump = np.zeros(len(d), bool)
    for _ in range(30):
        ols = sm.OLS(d.dx[~jump], X[~jump]).fit()
        e = d.dx - ols.predict(X)
        s = 1.4826 * np.median(np.abs(e - np.median(e)))
        new = (np.abs(e) > z * s).values
        if (new == jump).all():
            break
        jump = new
    J = e[jump]
    n = len(d)
    up, dn = J[J > 0], J[J < 0]
    p = dict(c=ols.params["const"], g=ols.params["t"], kappa=-ols.params["xl"], phi=ols.params["dl"], sigma=s,
             p_up=len(up) / n, mu_up=up.mean(), sd_up=up.std(ddof=0) if len(up) > 1 else 0.0,
             p_down=len(dn) / n, mu_down=dn.mean() if len(dn) else 0.0,
             sd_down=dn.std(ddof=0) if len(dn) > 1 else 0.0, t_end=t[-1])
    return p, seas, J


def simulate_jump(p, seas, start_price, start_date, weeks, n, prev_dx=0.0, seed=0):
    """Weekly paths of price, starting the week after start_date."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date + pd.Timedelta(days=7), periods=weeks, freq="7D")
    s = seas.reindex(week_of_year(dates)).values
    s0 = seas.reindex(week_of_year(pd.DatetimeIndex([start_date]))).values[0]
    x = np.full(n, np.log(start_price) - s0)
    dl = np.full(n, prev_dx)
    out = np.empty((n, weeks))
    for k in range(weeks):
        t = p["t_end"] + (k + 1) / 52
        u = rng.random(n)
        jmp = np.where(u < p["p_up"], rng.normal(p["mu_up"], p["sd_up"], n),
                       np.where(u < p["p_up"] + p["p_down"], rng.normal(p["mu_down"], p["sd_down"], n), 0.0))
        dx = p["c"] + p["g"] * t - p["kappa"] * x + p["phi"] * dl + p["sigma"] * rng.standard_normal(n) + jmp
        x, dl = x + dx, dx
        out[:, k] = np.exp(x + s[k])
    return pd.DataFrame(out.T, index=dates)


# Fit window. Starting in 2000 gives a 2.8%/yr long-run trend driven by the cheap early 2000s, which puts
# the end-2025 "normal" price near $3.87 and makes both models drift up for reasons unrelated to shocks.
# From 2005 the trend is about 1%/yr and the end-2025 normal price about $3.37 (2023-25 averaged $3.63).
FIT_START = "2005-01-01"

# Supply shocks known before 2026 (month the shock hit to the month prices peaked or the crisis eased).
CRISES = [
    ("2005-09", "2005-11", "Hurricanes Katrina and Rita shut Gulf refineries"),
    ("2008-03", "2008-08", "2008 oil spike, Brent peaked at about $145 in July"),
    ("2011-02", "2011-07", "Libyan civil war cut exports"),
    ("2012-02", "2012-05", "Iran sanctions and Strait of Hormuz threats"),
    ("2022-03", "2022-08", "Russia invades Ukraine"),
]


def monthly_x(g):
    m = g.resample("MS").mean()
    ly = np.log(m)
    r = ly - ly.rolling(13, center=True, min_periods=7).mean()
    seas = r.groupby(r.index.month).mean()
    seas -= seas.mean()
    x = ly - seas.reindex(m.index.month).values
    return m, x, seas


def fit_regime(g):
    m, x, seas = monthly_x(g)
    t = np.arange(len(x)) / 12
    tr = np.polyfit(t, x.values, 1)                       # long-run trend in log price
    d = pd.DataFrame({"x": x.values, "t": t}, index=m.index)
    d["gap"] = d.x - np.polyval(tr, d.t)
    d["dx"], d["gl"] = d.x.diff(), d.gap.shift()
    d["dl"] = d.dx.shift()
    d["crisis"] = False
    for a, b, _ in CRISES:
        d.loc[a:b, "crisis"] = True
    d = d.dropna()
    p = {"trend": tr[0], "level0": tr[1], "t_end": t[-1]}
    for name, mask in (("calm", ~d.crisis), ("crisis", d.crisis)):
        o = sm.OLS(d.dx[mask], sm.add_constant(d.loc[mask, ["gl", "dl"]])).fit()
        p.update({f"a_{name}": o.params["const"], f"b_{name}": o.params["gl"], f"phi_{name}": o.params["dl"],
                  f"sigma_{name}": np.sqrt(o.scale), f"n_{name}": int(mask.sum())})
    calm_months = (~d.crisis).sum()
    crisis_months = d.crisis.sum()
    p["p_enter"] = len(CRISES) / calm_months               # per calm month
    p["p_exit"] = len(CRISES) / crisis_months              # per crisis month (mean length = 1/p_exit)
    p["premium"] = -p["a_crisis"] / p["b_crisis"]          # crisis pulls the gap toward this level
    return p, seas


def simulate_regime(p, seas, hist_monthly, months, n, start_crisis=False, seed=0):
    """Monthly paths starting the month after the last month of hist_monthly."""
    rng = np.random.default_rng(seed)
    last = hist_monthly.index[-1]
    dates = pd.date_range(last + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    x_hist = np.log(hist_monthly.values[-2:]) - seas.reindex(hist_monthly.index[-2:].month).values
    t0 = p["t_end"]
    x = np.full(n, x_hist[-1])
    dl = np.full(n, x_hist[-1] - x_hist[-2])
    crisis = np.full(n, start_crisis)
    out, reg = np.empty((n, months)), np.empty((n, months), bool)
    for k in range(months):
        u = rng.random(n)
        crisis = np.where(crisis, u > p["p_exit"], u < p["p_enter"])
        gap = x - (p["level0"] + p["trend"] * (t0 + k / 12))
        a = np.where(crisis, p["a_crisis"], p["a_calm"])
        b = np.where(crisis, p["b_crisis"], p["b_calm"])
        ph = np.where(crisis, p["phi_crisis"], p["phi_calm"])
        sg = np.where(crisis, p["sigma_crisis"], p["sigma_calm"])
        dx = a + b * gap + ph * dl + sg * rng.standard_normal(n)
        x, dl = x + dx, dx
        out[:, k] = np.exp(x + seas.reindex([dates[k].month]).values[0])
        reg[:, k] = crisis
    return pd.DataFrame(out.T, index=dates), pd.DataFrame(reg.T, index=dates)


def backtest(m_sim, m_act):
    q = m_sim.quantile([0.05, 0.5, 0.95], axis=1).T
    rows = []
    for mth, a in m_act.items():
        if mth in m_sim.index:
            rows.append({"month": mth.strftime("%Y-%m"), "actual": a, "p05": q.loc[mth, 0.05],
                         "median": q.loc[mth, 0.5], "p95": q.loc[mth, 0.95],
                         "pct_rank": (m_sim.loc[mth] < a).mean()})
    return pd.DataFrame(rows)


def main():
    g = load_weekly()
    fit = g[FIT_START:"2025-12-31"]
    p, seas, J = fit_jump(fit)
    lr = np.exp((p["c"] + p["g"] * p["t_end"]) / p["kappa"])
    print("Model A (spike), fit on", fit.index[0].date(), "to", fit.index[-1].date())
    print(f"  mean reversion half-life {np.log(2) / p['kappa']:.0f} weeks, long-run level end-2025 ${lr:.2f} "
          f"(trend {p['g'] / p['kappa'] * 100:.1f}%/yr), momentum {p['phi']:.2f}, weekly sd {p['sigma']:.3f}")
    print(f"  up jumps {p['p_up'] * 52:.2f}/yr, mean {p['mu_up']:+.1%}; down jumps {p['p_down'] * 52:.2f}/yr, "
          f"mean {p['mu_down']:+.1%}")
    pd.Series(p).to_csv(os.path.join(PARAMS, "gas_model_jump.csv"), header=["value"])

    # Model B
    pb, seas_m = fit_regime(fit)
    print("\nModel B (regime), crisis months labeled from", len(CRISES), "supply shocks before 2026")
    print(f"  long-run trend {pb['trend'] * 100:.1f}%/yr; crisis starts {pb['p_enter'] * 12:.2f}/yr, "
          f"lasts {1 / pb['p_exit']:.1f} months on average")
    for r in ("calm", "crisis"):
        print(f"  {r:6s}: n={pb[f'n_{r}']:3d}  drift {pb[f'a_{r}']:+.3f}/mo  pull-back {-pb[f'b_{r}']:.3f}/mo  "
              f"momentum {pb[f'phi_{r}']:+.2f}  sd {pb[f'sigma_{r}']:.3f}")
    print(f"  crisis pulls price toward {np.exp(pb['premium']) - 1:+.0%} above trend")
    pd.Series(pb).to_csv(os.path.join(PARAMS, "gas_model_regime.csv"), header=["value"])

    # 2026 backtest: fit through 2025, simulate from Jan 2026, compare with what happened
    n = 5000
    m_act = g["2026-01-01":].resample("MS").mean()
    simA = simulate_jump(p, seas, fit.iloc[-1], fit.index[-1], 48 * 52 // 12, n,
                         prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=1).resample("MS").mean()
    simB, regB = simulate_regime(pb, seas_m, fit.resample("MS").mean(), 48, n, seed=2)
    out = []
    for name, sim in (("A_spike", simA), ("B_regime", simB)):
        bt = backtest(sim, m_act).assign(model=name)
        out.append(bt)
        print(f"\n2026 backtest, model {name} (monthly average $/gal):")
        print(bt.drop(columns="model").round(2).to_string(index=False))
        yr = sim.loc["2026"]
        print(f"  P(some 2026 month >= $4.50) {(yr.max() >= 4.5).mean():.1%};  "
              f"P(2026 Mar-Oct average >= actual {m_act['2026-03':'2026-10'].mean():.2f}) "
              f"{(yr.loc['2026-03':'2026-10'].mean() >= m_act['2026-03':'2026-10'].mean()).mean():.1%};  "
              f"4-yr average: median {sim.mean().median():.2f}, 95th pct {sim.mean().quantile(.95):.2f}")
    pd.concat(out).to_csv(os.path.join(PARAMS, "gas_backtest_2026.csv"), index=False)
    plot(fit, m_act, simA, simB)
    return p, pb


def plot(fit, m_act, simA, simB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = fit["2022-01-01":].resample("MS").mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sim, title in ((axes[0], simA, "Model A: spike (jumps, mean reversion)"),
                           (axes[1], simB, "Model B: calm / crisis regimes")):
        s = sim.loc[:"2027-12"]
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
        ax.fill_between(q.index, q[0.05], q[0.95], color="#9bb7d4", alpha=0.5, label="90% band")
        ax.fill_between(q.index, q[0.25], q[0.75], color="#4a7fb5", alpha=0.5, label="50% band")
        ax.plot(q.index, q[0.5], color="#1f3f66", lw=1, label="median")
        ax.plot(hist.index, hist.values, color="black", lw=1.2, label="actual (fit data)")
        ax.plot(m_act.index, m_act.values, color="#c0392b", lw=1.8, label="actual 2026 (not used in fit)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Chicago regular gas, $/gal (monthly avg)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gas_backtest_2026.png"), dpi=150)


if __name__ == "__main__":
    main()
