"""
Step 5: two-stage stochastic program with a superquantile (CVaR) objective, one driver type at a time.

Stage 1 (now): car x in {keep_gas, hybrid, ev}.
Stage 2 (each month t of scenario s, after prices and demand are known):
    h[s,t,b] >= 0   hours driven in clock-hour block b (24 blocks, at most ~30.4 h each per month)
    w[s,t]   >= 0   cash shortfall borrowed that month
  time:  sum_b h * (1 + charge_h_per_mile_x * miles_b) <= H_max      (EV fast charging eats hours)
  cash:  sum_b h * (rev_b * R[s,t] - miles_b * cpm_x[s,t]) - car_payment_x - insurance_x + w >= need
Loss in scenario s (minus the driver's 4-year surplus):
  L_x(s) = sum_t [ -sum_b h * (rev_b*R - miles_b*cpm - r*(1 + charge*miles_b)) + kappa*w ]
           + fixed_x - resale_x
  r is the driver type's reservation wage (step 3), so an hour is only worth driving if it beats r.
Objective:  min_x  (1-lam) * E[L_x] + lam * CVaR_alpha(L_x)
  CVaR by Rockafellar-Uryasev:  eta + 1/(1-alpha) * sum_s p_s * z_s,  z_s >= L_x(s) - eta, z_s >= 0.

Stage 1 has three options, so we enumerate them; that is exactly the MILP with a binary x.
For a fixed car, the best recourse in each scenario is the one with the smallest loss in that scenario,
whatever lam is (mean and CVaR both get better when any scenario's loss goes down). So we solve the
recourse LP once per car and then evaluate every lam on the scenario losses. build_and_solve can also
solve the full Rockafellar-Uryasev LP for a given lam; tests/test_step5.py checks the two agree.

Run from repo root:  python src/step5_model.py [A_spike|B_regime]
Outputs: results/step5_{model}.csv, results/step5_losses_{model}.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "data", "params")
SCEN = os.path.join(ROOT, "data", "scenarios")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

CARS = ["keep_gas", "hybrid", "ev"]
ALPHA = 0.90
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
KAPPA = 0.15          # extra cost per $ borrowed (fees + interest on short-term credit)
NEED_SHARE = 0.85     # monthly cash the driver counts on = 85% of usual net driving income with the gas car


def load_params(case=None):
    case = case or {}
    c = pd.read_csv(os.path.join(PARAMS, "cars.csv")).set_index("param")
    mk = pd.read_csv(os.path.join(PARAMS, "market.csv")).set_index("param")["base"].astype(float)
    num = lambda col, row: float(c.loc[row, col]) if pd.notna(c.loc[row, col]) else np.nan
    P = {"dcfc": mk["dcfc_price"], "home_charging": mk["home_charging"],
         "rebate": mk["il_ev_rebate"] + mk["uber_ev_grant"], "fuel_relief": mk["fuel_relief"],
         "trade_in": num("keep_gas", "trade_in_value_now"), "apr": num("ev", "loan_apr"),
         "term": int(num("ev", "loan_term"))}
    for x in CARS:
        P[x] = {
            "price": num(x, "purchase_price"), "maint": num(x, "maintenance"),
            "insurance": num(x, "insurance") + num(x, "rideshare_endorsement"),
        }
    P["keep_gas"]["mpg"] = num("keep_gas", "mpg_epa") * num("keep_gas", "city_mpg_factor")
    P["hybrid"]["mpg"] = num("hybrid", "mpg_epa") * num("hybrid", "city_mpg_factor")
    P["ev"]["kwh_mi"] = num("ev", "kwh_per_mile_epa") * (1 + num("ev", "charging_loss"))
    P["ev"]["fade"] = num("ev", "battery_fade_per_year")
    P["ev"]["fc_hours"] = num("ev", "fast_charge_minutes") / 60
    P["ev"]["fc_miles"] = num("ev", "fast_charge_range_added")
    P["keep_gas"]["resale"] = num("keep_gas", "resale_after_4y_keep")
    P["hybrid"]["resale"] = num("hybrid", "resale_after_4y_share") * P["hybrid"]["price"]
    P["ev"]["resale"] = num("ev", "resale_after_4y_share") * P["ev"]["price"]
    P["home_price"] = 0.17                  # $/kWh residential, only if home_charging = 1
    for k, v in case.items():               # sensitivity overrides, e.g. {"ev.price": 37000}
        if "." in k:
            a, b = k.split(".")
            P[a][b] = v
        else:
            P[k] = v
    return P


def loan_payment(principal, apr, n):
    if principal <= 0:
        return 0.0
    r = apr / 12
    return principal * r / (1 - (1 + r) ** -n)


def car_terms(x, P, gas, elec, months):
    """cost per mile [S,T], charging hours per mile [T], fixed cost over horizon, monthly cash outflow."""
    T = months
    yr = np.arange(T) // 12
    if x == "ev":
        price = P["dcfc"] * elec if not P["home_charging"] else P["home_price"] * elec
        cpm = P["ev"]["kwh_mi"] * price + P["ev"]["maint"]
        charge = (0.0 if P["home_charging"] else P["ev"]["fc_hours"]) / (P["ev"]["fc_miles"] * (1 - P["ev"]["fade"] * yr))
    else:
        cpm = np.maximum(gas - P["fuel_relief"], 0) / P[x]["mpg"] + P[x]["maint"]
        charge = np.zeros(T)
    if x == "keep_gas":
        pay = 0.0
    else:
        pay = loan_payment(P[x]["price"] - P["trade_in"] - (P["rebate"] if x == "ev" else 0), P["apr"], P["term"])
    ins = P[x]["insurance"] / 12
    fixed = T * (pay + ins) - P[x]["resale"]
    return cpm, charge, fixed, pay + ins


def build_and_solve(x, typ, tb, sc, P, lam, alpha=ALPHA):
    """Extensive-form LP for one car and one lambda. Returns per-scenario loss and stats."""
    S, T = sc["gas"].shape
    B = len(tb)
    rev, mi, avail = tb.driver_rev_per_hour.values, tb.miles_per_hour.values, tb.hours_available.values
    r, Hmax = typ["reservation_wage"], typ["hours_max"]
    cpm, charge, fixed, outflow = car_terms(x, P, sc["gas"], sc["elec"], T)
    need = typ["need"]
    p = sc["prob"]

    nh = S * T * B
    nw = S * T
    # variable order: h (s,t,b), w (s,t), z (s), eta
    iz, ieta = nh + nw, nh + nw + S
    nv = ieta + 1
    # per-hour surplus and cash coefficients [S,T,B]
    cash_c = rev[None, None, :] * sc["rev"][:, :, None] - mi[None, None, :] * cpm[:, :, None]
    surp_c = cash_c - r * (1 + charge[None, :, None] * mi[None, None, :])
    # loss_s = -sum surp*h + kappa*sum w + fixed
    c = np.zeros(nv)
    wgt = (1 - lam) * p
    c[:nh] = (-surp_c * wgt[:, None, None]).ravel()
    c[nh:nh + nw] = (KAPPA * np.repeat(wgt, T))
    c[iz:iz + S] = lam * p / (1 - alpha)
    c[ieta] = lam

    rows, cols, vals, rhs = [], [], [], []
    hidx = np.arange(nh).reshape(S, T, B)
    widx = nh + np.arange(nw).reshape(S, T)
    row = 0
    # time: sum_b h*(1+charge*mi) <= Hmax
    tcoef = 1 + charge[:, None] * mi[None, :]                       # [T,B]
    for s in range(S):
        for t in range(T):
            rows += [row] * B
            cols += list(hidx[s, t])
            vals += list(tcoef[t])
            rhs.append(Hmax)
            row += 1
    # cash: -(sum cash_c*h) - w <= -(need + outflow)
    for s in range(S):
        for t in range(T):
            rows += [row] * (B + 1)
            cols += list(hidx[s, t]) + [widx[s, t]]
            vals += list(-cash_c[s, t]) + [-1.0]
            rhs.append(-(need + outflow))
            row += 1
    # cvar: loss_s - eta - z_s <= -fixed   (loss_s without the fixed part)
    if lam > 0:
        for s in range(S):
            rows += [row] * (T * B + T + 2)
            cols += list(hidx[s].ravel()) + list(widx[s]) + [iz + s, ieta]
            vals += list(-surp_c[s].ravel()) + [KAPPA] * T + [-1.0, -1.0]
            rhs.append(-fixed)
            row += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(row, nv))
    ub = np.full(nv, np.inf)
    ub[:nh] = np.tile(avail, S * T)
    lb = np.zeros(nv)
    lb[ieta] = -np.inf
    if lam == 0:
        ub[iz:] = 0
        lb[ieta] = 0
        ub[ieta] = 0
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    h = res.x[:nh].reshape(S, T, B)
    w = res.x[nh:nh + nw].reshape(S, T)
    loss = -(surp_c * h).sum((1, 2)) + KAPPA * w.sum(1) + fixed
    income = (cash_c * h).sum((1, 2)) - w.sum(1) * 0 - T * outflow + P[x]["resale"]   # cash net of car costs
    hours = (h * tcoef[None]).sum((1, 2)) / T
    miles = (h * mi[None, None, :]).sum((1, 2)) / T
    return {"loss": loss, "income": income, "hours": hours, "miles": miles, "borrow": w.sum(1),
            "obj": res.fun + (1 - lam) * fixed}          # constant part of the mean term


def cvar(v, p, a=ALPHA):
    o = np.argsort(v)
    v, p = v[o], p[o]
    cum = np.cumsum(p)
    k = np.searchsorted(cum, a)
    var = v[k]
    return var + np.dot(p, np.maximum(v - var, 0)) / (1 - a)


def load_scen(model, n=None):
    d = pd.read_csv(os.path.join(SCEN, f"scen_{model}.csv"))
    piv = lambda col: d.pivot(index="scenario", columns="t", values=col).values
    sc = {"gas": piv("gas"), "elec": piv("elec_index"), "rev": piv("rev_index"),
          "prob": d.groupby("scenario").prob.first().values}
    if n:
        for k in ("gas", "elec", "rev"):
            sc[k] = sc[k][:n]
        sc["prob"] = sc["prob"][:n] / sc["prob"][:n].sum()
    return sc


def driver_types():
    t = pd.read_csv(os.path.join(PARAMS, "driver_types.csv")).set_index("type")
    base_cpm = None
    return t, base_cpm


def run(model="A_spike", case=None, lambdas=LAMBDAS, types=None, n_scen=None, tag="base", verbose=True,
        types_df=None, sc=None):
    P = load_params(case)
    tb = pd.read_csv(os.path.join(PARAMS, "time_blocks.csv"))
    sc = sc or load_scen(model, n_scen)
    t = types_df if types_df is not None else driver_types()[0]
    out, losses = [], []
    for typ_name in (types or t.index):
        typ = t.loc[typ_name].to_dict()
        # cash the driver counts on: share of usual net driving income with the gas car at $3.80
        typ["need"] = NEED_SHARE * (typ["revenue_base"] - typ["miles_base"] * (3.80 / P["keep_gas"]["mpg"] + P["keep_gas"]["maint"])) \
            - P["keep_gas"]["insurance"] / 12
        typ["need"] = max(typ["need"], 0)
        sol = {x: build_and_solve(x, typ, tb, sc, P, 0.0) for x in CARS}
        for x, r in sol.items():
            losses.append(pd.DataFrame({"case": tag, "model": model, "type": typ_name, "car": x,
                                        "scenario": np.arange(len(r["loss"])), "prob": sc["prob"],
                                        "loss": r["loss"], "income": r["income"]}))
        for lam in lambdas:
            rows = []
            for x, r in sol.items():
                mean, cv = np.dot(sc["prob"], r["loss"]), cvar(r["loss"], sc["prob"])
                rows.append({"case": tag, "model": model, "type": typ_name, "lam": lam, "car": x,
                             "objective": (1 - lam) * mean + lam * cv, "mean_loss": mean, "cvar_loss": cv,
                             "mean_income": np.dot(sc["prob"], r["income"]),
                             "cvar_income_low": -cvar(-r["income"], sc["prob"]),
                             "hours_per_month": np.dot(sc["prob"], r["hours"]),
                             "miles_per_month": np.dot(sc["prob"], r["miles"]),
                             "borrow": np.dot(sc["prob"], r["borrow"])})
            best = min(rows, key=lambda o: o["objective"])
            for o in rows:
                o["chosen"] = o["car"] == best["car"]
            out += rows
            if verbose and lam in (0.0, 1.0):
                print(f"  {typ_name:9s} lam={lam:.2f}: best {best['car']:8s} "
                      + "  ".join(f"{o['car']}: E={o['mean_loss']:,.0f} CVaR={o['cvar_loss']:,.0f}" for o in rows),
                      flush=True)
    return pd.DataFrame(out), pd.concat(losses)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "A_spike"
    print(f"Gas model {model}, alpha={ALPHA}")
    res, losses = run(model)
    res.to_csv(os.path.join(RES, f"step5_{model}.csv"), index=False)
    losses.to_csv(os.path.join(RES, f"step5_losses_{model}.csv"), index=False)


if __name__ == "__main__":
    main()
