"""Average / weighted-blend Kaggle submission CSVs by id.

All inputs must share the same `id` index and prediction columns (verified).
Used for: (a) snapshot ensembles (equal-weight mean of cycle CSVs), and (b)
weighted cross-arch blends (w*A + (1-w)*B). Operating at the CSV level — rather
than re-running inference — is safe because the column order encodes the
(t, entity, axis) layout identically across our submitters, and is auditable.

Usage:
    # equal-weight mean of N CSVs:
    python blend_csvs.py --out OUT.csv --inputs a.csv b.csv c.csv
    # weighted 2-way blend:
    python blend_csvs.py --out OUT.csv --inputs A.csv B.csv --weights 0.70 0.30
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--weights', nargs='+', type=float, default=None,
                   help='Per-input weights (must sum-normalize); default = equal mean.')
    args = p.parse_args()

    dfs = [pd.read_csv(f).set_index('id').sort_index() for f in args.inputs]
    ref = dfs[0]
    for f, d in zip(args.inputs, dfs):
        if not d.index.equals(ref.index):
            sys.exit(f'[ERR] id index mismatch in {f}')
        if list(d.columns) != list(ref.columns):
            sys.exit(f'[ERR] column mismatch in {f}')

    if args.weights is None:
        w = [1.0 / len(dfs)] * len(dfs)
    else:
        if len(args.weights) != len(dfs):
            sys.exit('[ERR] #weights != #inputs')
        s = sum(args.weights)
        w = [x / s for x in args.weights]  # normalize defensively

    out = sum(wi * d for wi, d in zip(w, dfs))
    out.to_csv(args.out)
    v = out.values
    print(f'[BLEND] {len(dfs)} inputs, weights={[round(x,3) for x in w]}')
    print(f'[BLEND] wrote {args.out} | rows {out.shape[0]} cols {out.shape[1]} '
          f'| NaNs {int(np.isnan(v).sum())} | min/max {v.min():.1f}/{v.max():.1f}')


if __name__ == '__main__':
    main()
