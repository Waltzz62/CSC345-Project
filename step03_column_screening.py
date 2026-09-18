"""
STEP 3 - Why THESE columns? (inside the two chosen themes)

Every column is measured against Transported TWICE:

  linear   - Pearson / point-biserial, the standard correlation everyone reaches
             for. Assumes the relationship is a straight line.
  binned   - chi-square -> Cramer's V on binned values. Assumes nothing about
             the shape of the relationship.

The two disagree badly on this dataset, and the disagreement is the point:
a linear screen throws away variables that carry a real, strong, non-linear
signal. This script also proves that Pearson on `deck` is not merely weak but
meaningless, by re-running it with the deck labels shuffled.

    python step03_column_screening.py --data train.csv
Outputs: out/03_column_screening.csv, out/03_deck_label_shuffle.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import viz_lib as V

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

df = V.load(a.data)
y = df.Transported.astype(int)
BINS = [-1, 0, 50, 200, 500, 1000, 2000, 5000, np.inf]

NUMERIC = ["lux", "RoomService", "Spa", "VRDeck", "FoodCourt", "ShoppingMall",
           "com", "total_spend", "Age", "num", "gsize"]
CATEGORICAL = ["CryoSleep", "deck", "side", "HomePlanet", "Destination", "VIP"]

V.banner("STEP 3.1  every column, measured two ways")
rows = []
for c in NUMERIC:
    m = df[c].notna()
    r, pl = stats.pointbiserialr(y[m], df.loc[m, c])
    v, pb = V.cramers_v(pd.cut(df.loc[m, c], BINS), y[m])
    rows.append((c, "numeric", abs(r), pl, v, pb))
for c in CATEGORICAL:
    m = df[c].notna()
    v, p = V.cramers_v(df.loc[m, c], y[m])
    rows.append((c, "categorical", v, p, np.nan, np.nan))

t = pd.DataFrame(rows, columns=["column", "type", "linear_strength", "linear_p",
                                "binned_V", "binned_p"])
t["best"] = t[["linear_strength", "binned_V"]].max(axis=1)
t = t.sort_values("best", ascending=False).drop(columns="best")

pd.set_option("display.width", 200)
print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
V.save_table(t, "03_column_screening.csv", float_fmt="%.4g")

V.banner("STEP 3.2  where the two measures disagree")
dis = t[(t.type == "numeric") & (t.linear_p > 0.01) & (t.binned_p < 1e-10)]
for r in dis.itertuples():
    print(f"  {r.column}: linear says r={r.linear_strength:.3f} (p={r.linear_p:.2f}, "
          f"NOT significant) but binned says V={r.binned_V:.3f} (p={r.binned_p:.1e})")
print("\n  A near-zero linear correlation does NOT mean 'no relationship'.")
print("  It means 'no STRAIGHT-LINE relationship'. Print the shape to see it:")
awake = df[(df.CryoSleep == False) & df.ShoppingMall.notna()]
shape = awake.groupby(pd.cut(awake.ShoppingMall, BINS),
                      observed=True).Transported.agg(["mean", "size"])
shape["mean"] = (shape["mean"] * 100).round(1)
print("  (awake passengers only, to match Figure 2)")
print(shape.rename(columns={"mean": "transported_%", "size": "n"}).to_string())
print("  -> down, then up. A U shape. A straight line through it is flat.")

V.banner("STEP 3.3  proof that Pearson on `deck` is meaningless, not just weak")
rng = np.random.default_rng(a.seed)
m = df.deck.notna()
decks = sorted(df.deck.dropna().unique())
out = []
for i in range(10):
    order = list(rng.permutation(decks))
    mp = {d: j for j, d in enumerate(order)}
    r = stats.pearsonr(df.loc[m, "deck"].map(mp), y[m])[0]
    out.append(("".join(order), r))
    print(f"  labelling {''.join(order)} -> Pearson r = {r:+.3f}")
v, _ = V.cramers_v(df.loc[m, "deck"], y[m])
print(f"\n  Pearson r ranges {min(o[1] for o in out):+.3f} to {max(o[1] for o in out):+.3f}"
      f" on IDENTICAL data - it only depends on how we numbered the decks.")
print(f"  Cramer's V = {v:.3f} every single time, because it ignores ordering.")
print("  -> deck is nominal. Pearson is the wrong tool for it, whatever it returns.")
V.save_table(pd.DataFrame(out, columns=["deck_label_order", "pearson_r"]),
             "03_deck_label_shuffle.csv")

print("\nSTEP 3 done.\n")
