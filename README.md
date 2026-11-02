# EVs as Oil-Shock Insurance for a Rideshare Fleet

ISE 599 Project 2 (USC, Fall 2026)

## Question

Gas prices can jump fast, like in 2022 and again in 2026 with the Iran conflict. A rideshare fleet that runs mostly gas cars eats that shock directly. EVs cost more up front but are mostly cut off from oil prices. So how much of the fleet should be gas, hybrid, and EV? And how much extra is a risk-averse fleet willing to pay for that protection?

## Approach

1. **Data**: Chicago TNP open data (vehicles, trips), EIA gas and electricity prices, fueleconomy.gov efficiency numbers.
2. **AI-assisted cleaning**: an LLM classifies messy make/model records as BEV / hybrid / gas, checked against a rule-based baseline and a hand-labeled sample.
3. **Price model**: fit a jump-diffusion process to weekly Chicago gas prices and simulate scenarios.
4. **Optimization**: two-stage stochastic program. Stage 1 picks the fleet mix. Stage 2 runs the fleet under each price scenario.
5. **Risk**: solve once for minimum expected cost and once with a CVaR limit on the cost tail. The cost gap between the two plans is the "price of the insurance."
6. **Check**: back-test on the 2022 and 2026 shocks; sensitivity to jump size, charging cost, EV price.

## Repo layout

```
src/
  step1_tnp_powertrain_share.py   data feasibility: monthly BEV / hybrid share
  download_data.py                pulls all other datasets
data/
  README.md                       full data source list
  raw/  processed/                (git-ignored)
figures/
docs/                             proposal, slides, notes
```

## Run

```bash
pip install -r requirements.txt
python src/step1_tnp_powertrain_share.py
python src/download_data.py
```

Set `SOCRATA_APP_TOKEN` (optional) and `EIA_API_KEY` (needed for electricity prices) as environment variables.

## Status

- [x] Step 1 script: fleet powertrain mix over time
- [ ] Run step 1, check data quality
- [ ] LLM classification + hand check
- [ ] Gas price jump model
- [ ] Stochastic program + CVaR
- [ ] Results, sensitivity, slides
