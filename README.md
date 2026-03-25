# Public code package

## Files
- `stage2_feature_screening.py`
- `stage3_model_comparison.py`
- `stage4_ensemble_and_interpretation.py`
- `stage5_compare_with_esrs.py`

## Expected input
A tabular dataset in `.xlsx`, `.xls`, or `.csv` format containing the variables used in the manuscript.

## Example commands
```bash
python stage2_feature_screening.py --data data.xlsx --output-dir outputs
python stage3_model_comparison.py --data data.xlsx --output-dir outputs
python stage4_ensemble_and_interpretation.py --data data.xlsx --output-dir outputs
python stage5_compare_with_esrs.py --data data.xlsx --oof outputs/OOF_Probabilities_For_Stage5.csv --output-dir outputs
