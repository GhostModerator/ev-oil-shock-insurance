# Key assumptions and the evidence behind them

Parameter tables: `data/params/cars.csv` and `data/params/market.csv` (each row lists its source).

## 1. All three options are used cars

Our own TNP vehicle data: the average Chicago rideshare car is about 6 years old (6.0 in July 2022, 6.2 in Sept 2026), and only about 20% are model-year-new. EVs were 0.7 years old on average in July 2022 (mostly the Hertz rental Teslas), but 3.1 years by Sept 2026, so used EVs are now common too. A new-EV case is run as a sensitivity.

## 2. Base case: no home charging

- Chicago: 54.5% of households rent and 70.6% of housing units are in buildings with 2+ units ([CMAP, ACS 2019-2023](https://cmap.illinois.gov/wp-content/uploads/Chicago-Housing-Profile.pdf)).
- A stratified random survey of 1,357 California ridehail drivers (2023-24) found most do not see home charging as available ([TRID](https://trid.trb.org/View/2713773)). A 2019 survey of 148 LA drivers estimated about 80% rent and about 60% have no outlet at their parking spot ([UCLA](https://www.ioes.ucla.edu/wp-content/uploads/rajagopal_ucla_ev_tnc-survey-report.pdf)).
- Home charging is one of the strongest predictors of EV adoption among ridehail drivers ([UC Berkeley TSRC](https://tsrc.berkeley.edu/publications/charging-ahead-perceptions-and-adoption-electric-vehicles-among-full-and-part-time)).
- Limit: driver-level surveys are from California; Chicago evidence is housing-level only. Home charging is run as a separate case.

## 3. Base case: no EV subsidies

- The federal credit ended on Sept 30, 2025.
- Illinois issued 3,001 EV rebates statewide in the FY26 cycle; FY27 funding is $14M, at most about 7,000 rebates at $2,000 ([NPR Illinois](https://www.nprillinois.org/illinois/2026-07-22/illinois-electric-vehicle-rebate-program-application-cycle-to-open-august-1)). Chicago alone has about 86,000 active rideshare drivers per month.
- Uber's $4,000 Go Electric grant is limited to Platinum/Diamond drivers and 100 trips by end of 2026 ([electrive](https://www.electrive.com/2026/04/02/usa-uber-expands-ev-grants-nationwide/)).
- Only 10% of surveyed California drivers had used any EV incentive.
- A "with subsidies" case ($2,000 + $4,000) tests whether subsidies flip the choice.
