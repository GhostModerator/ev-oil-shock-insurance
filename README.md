# EVs as Oil-Shock Insurance for a Rideshare Fleet

ISE 599 Project 2 (USC, Fall 2026)

Project outline (shared doc): https://claude.ai/artifact/FcTbCvwgAdEPkGHzxtr4EL
Save an exported copy in [`docs/`](docs/outline.md) once the plan is final.

## Question

How should a Chicago rideshare fleet split its cars between gas, hybrid and battery-electric when gas prices can jump with no warning? We treat electrification as insurance: it costs extra in a normal year and pays off in a shock year. We measure the premium and the payout.

Two shocks frame it. In 2022 gas spiked and fell back within months. In 2026 the Iran war started at the end of February and the US average went from $2.98 to $4.15 by early September, and stayed high. A good risk model has to handle both a short spike and a long one.

## Approach

1. **Data**: Chicago TNP open data (vehicles, trips, drivers), EIA gas, Brent and Illinois electricity prices, EPA fuel economy.
2. **AI-assisted cleaning**: an LLM matches messy make/model strings to EPA entries and labels the powertrain. We check it against a rule-based baseline and a hand-labeled sample.
3. **Price model**: two candidates fit on 2000-2025 weekly Chicago gas. A mean-reverting jump model (fits 2022) and a calm/crisis regime-switching model (fits 2026). Electricity and demand are simulated jointly with gas.
4. **Optimization**: two-stage stochastic program. Stage 1 buys gas / hybrid / EV cars. Stage 2 runs the fleet month by month in each scenario (fuel, charging, charging downtime, extra rentals if short).
5. **Risk**: mean-superquantile (CVaR) objective. Sweep the risk weight to get a mean-risk frontier. The gap between the risk-neutral and risk-averse mix is the value of the insurance.
6. **Evaluation**: fit through 2025, then run the chosen mix through the real 2026 prices. Plus sensitivity to EV price premium, resale value, electricity price and jump size.

Key questions: Is a hybrid the cheaper hedge? How much of the hedge is left if power prices and fares move with gas? Did the Chicago market actually shift after 2022?

## Repo layout

```
src/
  step1_tnp_powertrain_share.py   data feasibility: monthly BEV / hybrid share of TNP vehicles
  download_data.py                pulls all other datasets into data/raw/
data/
  README.md                       full data source list with links
  raw/  processed/                (git-ignored)
figures/
docs/
  outline.md                      project outline sent to the professor
```

## Run

```bash
pip install -r requirements.txt
python src/step1_tnp_powertrain_share.py
python src/download_data.py               # or: python src/download_data.py gas brent power
```

Environment variables:

- `SOCRATA_APP_TOKEN` (optional) avoids throttling on the Chicago portal. Free at https://data.cityofchicago.org/profile/edit/developer_settings
- `EIA_API_KEY` (needed for electricity prices). Free at https://www.eia.gov/opendata/register.php

## Status

- [x] Step 1 script: fleet powertrain mix over time
- [x] Download script for all datasets
- [ ] Run step 1, check data quality and the share of ambiguous models
- [ ] LLM classification + hand audit
- [ ] Gas price models (jump vs. regime) + scenarios
- [ ] Stochastic program + superquantile
- [ ] 2026 backtest, sensitivity, slides
