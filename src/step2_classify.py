"""
Step 2: label every TNP make/model/year with a powertrain.

Order of rules
  1. Name says it (e.g. "Camry Hybrid", "Prius Prime", "Bolt EV")          -> method = name
  2. EPA fuel economy table for that make + model + year (+/- 1 year)       -> method = epa
       one powertrain offered that year  -> that label
       several offered (e.g. Camry 2019) -> AMBIGUOUS, with the options listed
  3. LLM labels for strings EPA can't match (data/llm/llm_labels.csv)      -> method = llm
  4. Otherwise UNMATCHED

Run from repo root:  python src/step2_classify.py
Inputs : data/fetched/tnp_vehicle_agg.csv, data/fetched/epa_vehicles_slim.csv, data/llm/llm_labels.csv (optional)
Outputs: data/labels/powertrain_labels.csv    one row per make/model/year
         data/labels/unmatched_models.csv     strings for the LLM pass, by volume
         data/labels/monthly_share_step2.csv  monthly shares with the new labels
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FETCHED = os.path.join(ROOT, "data", "fetched")
OUT = os.path.join(ROOT, "data", "labels")
LLM = os.path.join(ROOT, "data", "llm", "llm_labels.csv")
os.makedirs(OUT, exist_ok=True)

ATV = {"Hybrid": "HYBRID", "Plug-in Hybrid": "PHEV", "EV": "BEV"}   # everything else -> ICE
MAKE_ALIAS = {"CHEVY": "CHEVROLET", "VW": "VOLKSWAGEN", "MERCEDES": "MERCEDESBENZ",
              "MERCEDESBENZ": "MERCEDESBENZ", "LANDROVER": "LANDROVER", "MINI": "MINI"}


def key(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def make_key(s):
    k = key(s)
    return MAKE_ALIAS.get(k, k)


def name_rule(model):
    m = " " + re.sub(r"[^A-Z0-9]+", " ", str(model).upper()) + " "
    if re.search(r" (PLUG IN|PHEV|PRIME|ENERGI|4XE) ", m) or " VOLT " in m:
        return "PHEV"
    if re.search(r" (HYBRID|HYB|HEV) ", m) or re.search(r" PRIUS ", m):
        return "HYBRID"
    if re.search(r" (EV|ELECTRIC|E TRON|ETRON|BOLT|LEAF|MACH E|IONIQ 5|IONIQ 6|EV6|EV9|BZ4X|ID 4) ", m):
        return "BEV"
    return None


def build_epa_index():
    e = pd.read_csv(os.path.join(FETCHED, "epa_vehicles_slim.csv"), low_memory=False)
    e["cls"] = e["atvType"].map(ATV).fillna("ICE")
    e["mk"] = e["make"].map(make_key)
    rows = []
    for col in ("baseModel", "model"):
        t = e[["mk", "year", "cls", "comb08", "combE", col]].copy()
        t["md"] = t[col].map(key)
        rows.append(t.drop(columns=[col]))
    idx = pd.concat(rows).drop_duplicates()
    # per make/model/year: which powertrains exist, and average efficiency by powertrain
    g = idx.groupby(["mk", "md", "year"])
    opts = g["cls"].agg(lambda s: "|".join(sorted(set(s)))).rename("options")
    eff = idx.groupby(["mk", "md", "year", "cls"])[["comb08", "combE"]].mean()
    return opts, eff, set(idx["md"])


def epa_lookup(opts, mk, md, year):
    for dy in (0, -1, 1):
        k = (mk, md, year + dy)
        if k in opts.index:
            return opts.loc[k]
    return None


def label_one(make, model, year, opts, epa_models, llm_map):
    """Return (label, options, method) for one make/model/year."""
    mk, md = make_key(make), key(model)
    y = int(year) if not pd.isna(year) else None
    label, options, method = None, None, None
    nr = name_rule(model)
    if nr:
        label, options, method = nr, nr, "name"
    elif y is not None and 1980 < y < 2005:
        # The first hybrids sold under a shared name (Escape, Accord) came in model year 2005.
        # Older hybrids (Prius, Insight) are caught by the name rule above.
        label, options, method = "ICE", "ICE", "pre2005"
    elif y is not None and y <= 2030:
        md2 = md[len(mk):] if md.startswith(mk) and len(md) > len(mk) else md   # "MAZDA3" -> "3"
        o = epa_lookup(opts, mk, md, y) or epa_lookup(opts, mk, md2, y)
        if o is None and len(md) >= 4:   # EPA names that contain the TNP string, e.g. "CARAVANGRANDCARAVAN"
            found = [epa_lookup(opts, mk, c, y) for c in epa_models if md in c]
            found = [f for f in found if f is not None]
            if found:
                o = "|".join(sorted(set("|".join(found).split("|"))))
        if o is not None:
            options, method = o, "epa"
            label = o if "|" not in o else "AMBIGUOUS"
    if label is None and (mk, md) in llm_map:
        o = llm_map[(mk, md)]
        label, options, method = ("AMBIGUOUS" if "|" in o else o), o, "llm"
    if label is None:
        label, options, method = "UNMATCHED", "", "none"
    return label, options, method


def load_llm_map():
    llm = pd.read_csv(LLM) if os.path.exists(LLM) else pd.DataFrame(columns=["make", "model", "label"])
    return {(make_key(a), key(b)): c for a, b, c in llm[["make", "model", "label"]].itertuples(index=False)}


def main():
    v = pd.read_csv(os.path.join(FETCHED, "tnp_vehicle_agg.csv"), dtype=str)
    v["n_vehicles"] = pd.to_numeric(v["n_vehicles"], errors="coerce").fillna(0)
    v["n_trips"] = pd.to_numeric(v["n_trips"], errors="coerce").fillna(0)
    v["year"] = pd.to_numeric(v["year"], errors="coerce")

    keys = (v.groupby(["make", "model", "year"], dropna=False)[["n_vehicles", "n_trips"]].sum().reset_index())
    opts, eff, epa_models = build_epa_index()
    llm_map = load_llm_map()

    out = []
    for r in keys.itertuples(index=False):
        label, options, method = label_one(r.make, r.model, r.year, opts, epa_models, llm_map)
        out.append((r.make, r.model, r.year, label, options, method, r.n_vehicles, r.n_trips))

    lab = pd.DataFrame(out, columns=["make", "model", "year", "cls", "options", "method",
                                     "vehicle_months", "trips"])
    lab.to_csv(os.path.join(OUT, "powertrain_labels.csv"), index=False)

    tot = lab["vehicle_months"].sum()
    print("Share of vehicle-months by label:")
    print((lab.groupby("cls")["vehicle_months"].sum() / tot).sort_values(ascending=False).map("{:.1%}".format))
    print("\nBy method:")
    print((lab.groupby("method")["vehicle_months"].sum() / tot).map("{:.1%}".format))
    print("\nAmbiguous option sets:")
    amb = lab[lab.cls == "AMBIGUOUS"]
    print((amb.groupby("options")["vehicle_months"].sum() / tot).sort_values(ascending=False).head(6).map("{:.1%}".format))

    um = (lab[lab.cls == "UNMATCHED"].groupby(["make", "model"])["vehicle_months"].sum()
          .sort_values(ascending=False).reset_index())
    um["cum_share_of_unmatched"] = um["vehicle_months"].cumsum() / max(um["vehicle_months"].sum(), 1)
    um.to_csv(os.path.join(OUT, "unmatched_models.csv"), index=False)
    print(f"\nUnmatched strings: {len(um)}, {um['vehicle_months'].sum() / tot:.1%} of vehicle-months")

    # monthly shares with the new labels; hybrid as a range (ambiguous HYBRID|ICE counted 0..1)
    v = v.merge(lab[["make", "model", "year", "cls", "options"]], on=["make", "model", "year"], how="left")
    v["month"] = pd.to_datetime(v["month_reported"].str[:7], format="%Y-%m", errors="coerce")
    p = v.pivot_table(index="month", columns="cls", values="n_vehicles", aggfunc="sum").fillna(0)
    amb_h = (v[(v.cls == "AMBIGUOUS") & v.options.str.contains("HYBRID", na=False)]
             .groupby("month")["n_vehicles"].sum())
    total = p.sum(axis=1)
    share = pd.DataFrame({
        "total": total,
        "bev": p.get("BEV", 0) / total,
        "phev": p.get("PHEV", 0) / total,
        "hybrid_low": p.get("HYBRID", 0) / total,
        "hybrid_high": (p.get("HYBRID", 0) + amb_h.reindex(p.index).fillna(0)) / total,
        "unmatched": p.get("UNMATCHED", 0) / total,
    })
    share.to_csv(os.path.join(OUT, "monthly_share_step2.csv"))
    print("\nSelected months:")
    print(share.loc[["2022-02-01", "2022-12-01", "2026-02-01", "2026-09-01"]].round(4).to_string())


if __name__ == "__main__":
    main()
