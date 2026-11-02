"""
Step 4: gas price models.

Model A, "spike": mean-reverting jump model on weekly Chicago gas.
    x_t = log price minus a week-of-year seasonal factor
    dx_t = c + g*t - kappa*x_{t-1} + phi*dx_{t-1} + sigma*e_t + J_t
    J_t = jump with prob p_up (size N(mu_up, sd_up)) or p_down (size N(mu_down, sd_down))
    Fit by a robust filter: OLS, flag residuals beyond 3 robust SDs as jumps, refit, repeat.

Run from repo root:  python src/step4_gas_models.py
Outputs: data/params/gas_model_jump.csv, figures/gas_backtest_2026.png
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


def main():
    g = load_weekly()
    fit = g[:"2025-12-31"]
    p, seas, J = fit_jump(fit)
    lr = np.exp((p["c"] + p["g"] * p["t_end"]) / p["kappa"])
    print("Model A (spike), fit on", fit.index[0].date(), "to", fit.index[-1].date())
    print(f"  mean reversion half-life {np.log(2) / p['kappa']:.0f} weeks, long-run level end-2025 ${lr:.2f} "
          f"(trend {p['g'] / p['kappa'] * 100:.1f}%/yr), momentum {p['phi']:.2f}, weekly sd {p['sigma']:.3f}")
    print(f"  up jumps {p['p_up'] * 52:.2f}/yr, mean {p['mu_up']:+.1%}; down jumps {p['p_down'] * 52:.2f}/yr, "
          f"mean {p['mu_down']:+.1%}")
    pd.Series(p).to_csv(os.path.join(PARAMS, "gas_model_jump.csv"), header=["value"])

    # 2026 backtest: start from the last 2025 week, compare with what happened
    actual = g["2026-01-01":]
    sim = simulate_jump(p, seas, fit.iloc[-1], fit.index[-1], 48 * 52 // 12, 5000,
                        prev_dx=np.log(fit.iloc[-1] / fit.iloc[-2]), seed=1)
    m_sim = sim.resample("MS").mean()
    m_act = actual.resample("MS").mean()
    q = m_sim.quantile([0.05, 0.5, 0.95], axis=1).T
    rows = []
    for mth, a in m_act.items():
        if mth in m_sim.index:
            rows.append({"month": mth.strftime("%Y-%m"), "actual": a, "p05": q.loc[mth, 0.05],
                         "median": q.loc[mth, 0.5], "p95": q.loc[mth, 0.95],
                         "pct_rank": (m_sim.loc[mth] < a).mean()})
    bt = pd.DataFrame(rows)
    print("\n2026 backtest (monthly average, $/gal):")
    print(bt.round(2).to_string(index=False))
    print(f"\nP(any month of 2026 averages >= $4.50) under model A: {(m_sim.loc['2026'].max() >= 4.5).mean():.1%}")
    return p, sim, bt


if __name__ == "__main__":
    main()
