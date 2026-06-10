# Final submission CSVs

The key prediction files behind our best Kaggle submission (**2.93**), a
`0.70 × curriculum-MART + 0.30 × EqMotion` blend.

| File | Role |
|---|---|
| `solution_blend_curriculum0.70_eqm0.30.csv` | **Final submission — Kaggle 2.93.** The 0.70/0.30 blend of the two files below. |
| `mart_curriculum_4k_sw1k_lrdrop.csv` | Curriculum-MART predictions (0.70 weight) — augmented MART, min-ADE → mean-MSE curriculum with an LR drop at epoch 1000. |
| `solution_ens5_iso_hoops_val3.25.csv` | EqMotion 5-seed ensemble predictions (0.30 weight) — iso-norm + cosine + hoops (val 3.25 / Kaggle 3.2). |

## Reproduce the blend

```bash
cd ../src/mart
python blend_csvs.py \
    --inputs ../../final_csvs/mart_curriculum_4k_sw1k_lrdrop.csv \
             ../../final_csvs/solution_ens5_iso_hoops_val3.25.csv \
    --weights 0.70 0.30 \
    --out ../../final_csvs/solution_blend_curriculum0.70_eqm0.30.csv
```

See the top-level `README.md` ("Best model") for how each input model is trained.
