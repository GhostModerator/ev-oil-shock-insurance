"""
Step 6: evaluation.

  1. Sensitivity: which car wins for each driver type under other assumptions.
  2. Break-even gas price: how much higher gas would have to be, on every path, to flip the choice.
  3. 2026 backtest: run each car through what actually happened in Jan-Sep 2026.
  4. Implied fleet mix vs the real Chicago TNP fleet.

Run from repo root:  python src/step6_evaluate.py
Outputs: results/step6_*.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step4_scenarios as s4  # noqa: E402
import step5_model as m  # noqa: E402

RES = m.RES
FETCHED = os.path.join(m.ROOT, "data", "fetched")
LABELS = os.path.join(m.ROOT, "data", "labels")


def ev_price_case(price, extra=None):
    P = m.load_params()
    case = {"ev.price": price, "ev.resale": P["ev"]["resale"] / P["ev"]["price"] * price}
    case.update(extra or {})
    return case


CASES = [
    ("base", "A_spike", {}, None),
    ("gas model B", "B_regime", {}, None),
    ("home charging", "A_spike", {"home_charging": 1}, None),
    ("subsidies $6k", "A_spike", {"rebate": 6000}, None),
    ("home + subsidies", "A_spike", {"home_charging": 1, "rebate": 6000}, None),
    ("new EV $37k, home + subsidies", "A_spike", ev_price_case(37000, {"home_charging": 1, "rebate": 6000}), None),
    ("DC fast $0.40/kWh", "A_spike", {"dcfc": 0.40}, None),
    ("loan APR 13.9%", "A_spike", {"apr": 0.139}, None),
    ("revenue falls with gas (-0.3)", "A_spike", {}, -0.3),
    ("fuel surcharge (+0.2)", "A_spike", {}, 0.2),
]


def sensitivity():
    rows = []
    cache = {}
    for name, model, case, rev_beta in CASES:
        key = (model, rev_beta)
        if key not in cache:
            cache[key] = m.load_scen(model) if rev_beta is None else s4.make_scen(model, rev_beta)
        res, _ = m.run(model, case=case, lambdas=[0.0, 1.0], verbose=False, tag=name, sc=cache[key])
        rows.append(res)
        b = res[res.chosen].pivot(index="type", columns="lam", values="car")
        print(f"  {name:32s} " + "  ".join(f"{t}: {b.loc[t, 0.0]}/{b.loc[t, 1.0]}" for t in b.index), flush=True)
    out = pd.concat(rows)
    out.to_csv(os.path.join(RES, "step6_sensitivity.csv"), index=False)
    return out


def breakeven():
    """Scale every gas path by k and find where the choice flips."""
    sc0 = m.load_scen("A_spike")
    base_avg = np.dot(sc0["prob"], sc0["gas"].mean(1))
    rows = []
    for k in [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]:
        sc = dict(sc0, gas=sc0["gas"] * k)
        res, _ = m.run("A_spike", lambdas=[0.0], types=["light", "regular", "heavy"], verbose=False, sc=sc)
        w = res.pivot(index="type", columns="car", values="mean_loss")
        for t in w.index:
            rows.append({"k": k, "avg_gas": k * base_avg, "type": t, **w.loc[t].to_dict()})
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(RES, "step6_breakeven.csv"), index=False)

    def cross(t, a, c):
        d = b[b.type == t].sort_values("k")
        v = (d[a] - d[c]).values               # >0 means c better
        g = d.avg_gas.values
        for i in range(1, len(v)):
            if v[i - 1] <= 0 < v[i] or v[i - 1] > 0 >= v[i]:
                return g[i - 1] + (g[i] - g[i - 1]) * (-v[i - 1]) / (v[i] - v[i - 1])
        return None
    print(f"  base 4-yr average gas ${base_avg:.2f}")
    for t, a, c in (("light", "keep_gas", "hybrid"), ("regular", "keep_gas", "hybrid"),
                    ("heavy", "hybrid", "ev"), ("regular", "hybrid", "ev")):
        x = cross(t, a, c)
        print(f"  {t:8s} {a} vs {c}: flips at 4-yr average gas " + (f"${x:.2f}" if x else "not in $%.2f-%.2f"
              % (b.avg_gas.min(), b.avg_gas.max())))
    return b


def actual_2026():
    """Realized monthly inputs Jan-Sep 2026 on the same scale as the scenarios."""
    g = pd.read_csv(os.path.join(FETCHED, "eia_gas_chicago_weekly.csv"), parse_dates=["date"]).set_index("date").iloc[:, 0]
    gas = g.resample("MS").mean()["2026-01":"2026-09"]
    p = pd.read_csv(os.path.join(FETCHED, "eia_power_illinois_monthly.csv"))
    p["date"] = pd.to_datetime(p.period)
    e = p[p.sectorid.eq("COM")].set_index("date")["price"].sort_index()
    elec = (e / e["2025-12"].iloc[0]).reindex(gas.index).ffill()
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(FETCHED, "trips", "tnp_trips_month_hour_*.csv"))])
    t["date"] = pd.to_datetime(t.month)
    mm = t.groupby("date")[["trips", "fare", "seconds"]].sum()
    idx = (mm.fare / mm.seconds) * (mm.trips / mm.trips["2025-10":"2026-09"].mean()) ** s4.BETA_U
    rev = (idx / idx["2025-10":"2026-09"].mean()).reindex(gas.index)
    return pd.DataFrame({"gas": gas, "elec_index": elec, "rev_index": rev})


def wq(v, w, q):
    o = np.argsort(v)
    return float(np.interp(q, np.cumsum(w[o]) - w[o] / 2, v[o]))


def backtest_2026():
    act = actual_2026()
    T = len(act)
    print(act.round(3).to_string())
    P = m.load_params()
    tb = pd.read_csv(os.path.join(m.PARAMS, "time_blocks.csv"))
    types = m.driver_types()[0]
    sc_act = {"gas": act.gas.values[None], "elec": act.elec_index.values[None], "rev": act.rev_index.values[None],
              "prob": np.array([1.0])}
    full = m.load_scen("A_spike")
    sc_mod = {k: (v[:, :T] if k != "prob" else v) for k, v in full.items()}
    rows = []
    for tname in types.index:
        typ = types.loc[tname].to_dict()
        typ["need"] = 0.0          # compare driving results only
        for x in m.CARS:
            # 9-month result with the car's 4-year fixed cost (payments + insurance - resale) spread evenly,
            # so buying is not charged 9 payments with none of the value it keeps
            ra = m.build_and_solve(x, typ, tb, sc_act, P, 0.0)
            rm = m.build_and_solve(x, typ, tb, sc_mod, P, 0.0)
            fixed_T = m.car_terms(x, P, sc_act["gas"], sc_act["elec"], T)[2]
            fixed_48 = m.car_terms(x, P, full["gas"], full["elec"], full["gas"].shape[1])[2]
            adj = -fixed_T + fixed_48 * T / full["gas"].shape[1]
            la = ra["loss"][0] + adj
            lm = rm["loss"] + adj
            rows.append({"type": tname, "car": x, "actual_gain": -la, "model_mean_gain": -np.dot(full["prob"], lm),
                         "model_p05": -wq(lm, full["prob"], 0.95), "model_p95": -wq(lm, full["prob"], 0.05),
                         "actual_pct_rank": float(np.dot(full["prob"], lm > la)),
                         "actual_cash_income": ra["income"][0] - P[x]["resale"]})
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(RES, "step6_backtest_2026.csv"), index=False)
    print(b.round({"actual_pct_rank": 3}).round({c: 0 for c in b.columns if c not in ("type", "car", "actual_pct_rank")}).to_string(index=False))
    w = b.pivot(index="type", columns="car", values="actual_gain")
    wm = b.pivot(index="type", columns="car", values="model_mean_gain")
    print("\n  Hybrid minus keep, Jan-Sep 2026: actual vs model mean")
    print(pd.DataFrame({"actual": w.hybrid - w.keep_gas, "model": wm.hybrid - wm.keep_gas}).round(0).to_string())
    return b


def fleet_mix(sens):
    types = m.driver_types()[0]
    share = types.share_of_cars
    out = []
    for case in ("base", "home charging", "home + subsidies"):
        b = sens[(sens.case == case) & sens.chosen & (sens.lam == 0)].set_index("type").car
        mix = share.groupby(b.reindex(share.index)).sum()
        out.append(mix.rename(case))
    mix = pd.concat(out, axis=1).fillna(0)
    act = pd.read_csv(os.path.join(LABELS, "monthly_share_step2.csv"), parse_dates=[0], index_col=0).loc["2026-09-01"]
    mix["actual Sep 2026"] = pd.Series({"ev": act.bev, "hybrid": act.hybrid_low, "keep_gas": 1 - act.bev - act.phev - act.hybrid_low})
    mix.loc["hybrid (upper bound)", "actual Sep 2026"] = act.hybrid_high
    mix.to_csv(os.path.join(RES, "step6_fleet_mix.csv"))
    print((mix * 100).round(1).to_string())
    return mix


def main():
    print("1. Sensitivity (best car at lam=0 / lam=1):")
    sens = sensitivity()
    print("\n2. Break-even gas price:")
    breakeven()
    print("\n3. 2026 backtest (Jan-Sep):")
    backtest_2026()
    print("\n4. Implied fleet mix, share of cars (%):")
    fleet_mix(sens)


if __name__ == "__main__":
    main()
