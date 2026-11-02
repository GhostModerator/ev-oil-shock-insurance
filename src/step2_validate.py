"""
Step 2 check: how good are the powertrain labels?

1. Taxi test. Chicago taxi licenses list each car's real fuel source. We run the same
   labeling rules on taxi make/model/year and compare.
2. Hand-label sample. A random sample of TNP make/model/year rows (weighted by volume,
   stratified by labeling method) for a person to check by hand.

Run from repo root:  python src/step2_validate.py
Outputs: data/labels/taxi_check.csv, data/labels/hand_label_sample.csv
"""
import os

import pandas as pd

from step2_classify import FETCHED, OUT, build_epa_index, label_one, load_llm_map

TRUTH = {"Hybrid": "HYBRID", "Gasoline": "ICE", "Flex Fuel": "ICE",
         "Compressed Natural Gas": "ICE", "Electric": "BEV"}


def taxi_test(opts, epa_models, llm_map):
    t = pd.read_csv(os.path.join(FETCHED, "public_passenger_vehicle_licenses.csv"), low_memory=False)
    t = t[t["Vehicle Type"].eq("Taxi") & t["Vehicle Fuel Source"].isin(TRUTH)].copy()
    t["truth"] = t["Vehicle Fuel Source"].map(TRUTH)
    t["year"] = pd.to_numeric(t["Vehicle Model Year"], errors="coerce")
    res = [label_one(a, b, c, opts, epa_models, llm_map)
           for a, b, c in t[["Vehicle Make", "Vehicle Model", "year"]].itertuples(index=False)]
    t[["label", "options", "method"]] = pd.DataFrame(res, index=t.index)
    t.to_csv(os.path.join(OUT, "taxi_check.csv"), index=False)

    n = len(t)
    det = t[~t["label"].isin(["AMBIGUOUS", "UNMATCHED"])]
    amb = t[t["label"].eq("AMBIGUOUS")]
    print(f"Taxis with a known fuel source: {n:,}")
    print(f"  labeled one powertrain: {len(det):,} ({len(det)/n:.0%}); "
          f"accuracy {det['label'].eq(det['truth']).mean():.1%}")
    print(f"  ambiguous:              {len(amb):,} ({len(amb)/n:.0%}); "
          f"true label is one of the options {amb.apply(lambda r: r['truth'] in r['options'], axis=1).mean():.1%}")
    print(f"  unmatched:              {t['label'].eq('UNMATCHED').sum():,}")
    print("\nAccuracy by method (single-label rows):")
    print(det.groupby("method").apply(lambda g: pd.Series(
        {"n": len(g), "accuracy": g["label"].eq(g["truth"]).mean()}), include_groups=False).round(3))
    print("\nConfusion table (single-label rows):")
    print(pd.crosstab(det["truth"], det["label"]))
    wrong = det[det["label"].ne(det["truth"])]
    print("\nMost common errors:")
    print(wrong.groupby(["Vehicle Make", "Vehicle Model", "truth", "label"]).size()
          .sort_values(ascending=False).head(10).to_string())


def hand_sample(n_per_method=None, seed=7):
    lab = pd.read_csv(os.path.join(OUT, "powertrain_labels.csv"))
    lab = lab[lab["vehicle_months"] >= 50]      # skip one-off typos; sample uniformly among the rest
    n_per_method = n_per_method or {"epa": 100, "name": 60, "llm": 80, "pre2005": 40, "none": 20}
    parts = []
    for m, k in n_per_method.items():
        g = lab[lab["method"].eq(m)]
        if len(g):
            parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    s = pd.concat(parts)[["make", "model", "year", "cls", "options", "method", "vehicle_months"]]
    s["hand_label"] = ""        # fill: ICE / HYBRID / PHEV / BEV, or several with |
    s["agree"] = ""             # fill: y / n
    s["note"] = ""
    s.to_csv(os.path.join(OUT, "hand_label_sample.csv"), index=False)
    print(f"\nHand-label sample: {len(s)} rows -> data/labels/hand_label_sample.csv")


def main():
    opts, eff, epa_models = build_epa_index()
    llm_map = load_llm_map()
    taxi_test(opts, epa_models, llm_map)
    hand_sample()


if __name__ == "__main__":
    main()
