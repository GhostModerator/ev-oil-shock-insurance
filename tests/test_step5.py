"""Small checks of the step 5 LP against hand calculations.

Run from repo root:  python tests/test_step5.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import step5_model as m  # noqa: E402


def one_scenario(gas, elec=1.0, rev=1.0, T=1):
    return {"gas": np.full((1, T), gas), "elec": np.full((1, T), elec), "rev": np.full((1, T), rev),
            "prob": np.array([1.0])}


def hand_greedy(tb, cpm, charge, r, Hmax):
    """Fractional knapsack: sort blocks by surplus per unit of time used, fill while surplus > 0."""
    tcoef = 1 + charge * tb.miles_per_hour.values
    surp = tb.driver_rev_per_hour.values - tb.miles_per_hour.values * cpm - r * tcoef
    order = np.argsort(-surp / tcoef)
    left, total = Hmax, 0.0
    for b in order:
        if surp[b] <= 0 or left <= 0:
            break
        h = min(tb.hours_available.values[b], left / tcoef[b])
        total += h * surp[b]
        left -= h * tcoef[b]
    return total


def main():
    P = m.load_params()
    tb = pd.read_csv(os.path.join(m.PARAMS, "time_blocks.csv"))
    typ = {"reservation_wage": 14.0, "hours_max": 120.0, "need": 0.0}
    ok = True
    for x in m.CARS:
        for gas in (3.0, 5.0):
            sc = one_scenario(gas)
            r = m.build_and_solve(x, typ, tb, sc, P, lam=0.0)
            cpm, charge, fixed, _ = m.car_terms(x, P, sc["gas"], sc["elec"], 1)
            hand = hand_greedy(tb, cpm[0, 0], charge[0], typ["reservation_wage"], typ["hours_max"])
            lp = -(r["loss"][0] - fixed)
            good = abs(hand - lp) < 1e-4 * max(1, abs(hand))
            ok &= good
            print(f"{x:8s} gas ${gas:.2f}: LP surplus {lp:9.2f}  hand {hand:9.2f}  {'OK' if good else 'MISMATCH'}")
    # CVaR check: with lam = 1 the LP objective must equal CVaR of its own scenario losses
    sc = {"gas": np.array([[3.0, 3.0], [6.0, 6.0]]), "elec": np.ones((2, 2)), "rev": np.ones((2, 2)),
          "prob": np.array([0.7, 0.3])}
    r1 = m.build_and_solve("keep_gas", typ, tb, sc, P, lam=1.0)
    want = m.cvar(r1["loss"], sc["prob"], a=m.ALPHA)
    good = abs(r1["obj"] - want) < 1e-6 * abs(want)
    ok &= good
    print(f"lam=1 LP objective {r1['obj']:.2f} vs CVaR of losses {want:.2f} (worst loss {r1['loss'].max():.2f})"
          f"  {'OK' if good else 'MISMATCH'}")
    # full RU LP at lam=0.5 must give the same objective as the lam=0 recourse evaluated at lam=0.5
    sc = m.load_scen("A_spike", n=20)
    typ2 = {"reservation_wage": 14.0, "hours_max": 200.0, "need": 1500.0}
    r0 = m.build_and_solve("keep_gas", typ2, tb, sc, P, lam=0.0)
    r5 = m.build_and_solve("keep_gas", typ2, tb, sc, P, lam=0.5)
    via0 = 0.5 * np.dot(sc["prob"], r0["loss"]) + 0.5 * m.cvar(r0["loss"], sc["prob"])
    good = abs(r5["obj"] - via0) < 1e-6 * abs(via0) + 1e-6
    ok &= good
    print(f"RU LP at lam=0.5: {r5['obj']:.2f}; lam=0 recourse evaluated at lam=0.5: {via0:.2f}  {'OK' if good else 'MISMATCH'}")
    cv = m.cvar(np.array([1.0, 2.0, 3.0, 10.0]), np.full(4, 0.25), a=0.75)
    print(f"cvar([1,2,3,10], alpha=.75) = {cv:.2f} (should be 10.00)")
    ok &= abs(cv - 10) < 1e-9
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
