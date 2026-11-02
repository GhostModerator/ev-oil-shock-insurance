# LLM powertrain labels

`llm_labels.csv` holds labels for TNP make/model strings that the EPA match in `src/step2_classify.py` could not resolve (typos, merged words, trim names, non-US names, very old models).

How it was made:

1. `step2_classify.py` writes `data/processed/unmatched_models.csv`, sorted by vehicle-months.
2. The top ~160 strings (about 90% of unmatched volume) were given to an LLM with this instruction:

   > For each US car make and model string below, give the powertrain(s) that were sold under that name in the US: ICE, HYBRID, PHEV or BEV. If more than one was sold under the same name, list all of them separated by "|". Do not guess a trim that is not in the string.

3. The answers were saved here without edits. Labels with "|" are treated as AMBIGUOUS, the same as EPA cases with several powertrains.

Strings below the top ~160 stay UNMATCHED. They are under 0.2% of vehicle-months.

Known judgment calls to check by hand: `Toyota Corolla Ascent` (non-US name), `Honda Cr` (CR-V or CR-Z), `Volkswagen Sedan` (too vague; labeled ICE).
