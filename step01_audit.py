"""
STEP 1 - Know the data before touching it.

Produces the raw material for the "data summary & data problems" slide:
shape, target balance, missing values, and the three structural problems
that shape everything we do later.

    python step01_audit.py --data train.csv
Outputs: out/01_overview.csv, out/01_missing.csv, out/01_spend_skew.csv
"""

import argparse
import pandas as pd
import viz_lib as V

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
a = ap.parse_args()

df = pd.read_csv(a.data)

V.banner("STEP 1.1  shape and target balance")
print(f"  rows x columns      : {df.shape[0]:,} x {df.shape[1]}")
print(f"  Transported = True  : {df.Transported.mean()*100:.1f}%")
print(f"  Transported = False : {(1-df.Transported.mean())*100:.1f}%")
print("  -> the target is almost perfectly balanced, so 50% is the score to beat")
V.save_table(pd.DataFrame({
    "metric": ["rows", "columns", "transported_true_pct"],
    "value": [df.shape[0], df.shape[1], round(df.Transported.mean()*100, 2)]}),
    "01_overview.csv")

V.banner("STEP 1.2  missing values")
m = pd.DataFrame({"column": df.columns,
                  "missing": df.isna().sum().values,
                  "pct": (df.isna().mean()*100).round(2).values})
print(m.to_string(index=False))
print("  -> ~2% missing spread over EVERY column except PassengerId/Transported.")
print("     No single column is unusable; no row is complete-case safe either.")
V.save_table(m, "01_missing.csv", float_fmt="%.2f")

V.banner("STEP 1.3  structural problem A - Cabin is three fields in one")
print(df[["PassengerId", "Cabin"]].head(5).to_string(index=False))
print("  -> 'B/0/P' = deck B, room 0, Port side. Unusable until split.")

V.banner("STEP 1.4  structural problem B - spending is extremely skewed")
S = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
sk = df[S].describe().T[["mean", "50%", "max"]]
sk["pct_zero"] = (df[S] == 0).mean().round(3) * 100
sk = sk.reset_index().rename(columns={"index": "service", "50%": "median"})
print(sk.to_string(index=False))
print("  -> median is 0 for every service while the max reaches tens of thousands.")
print("     A mean would be meaningless here; we bin instead.")
V.save_table(sk, "01_spend_skew.csv", float_fmt="%.1f")

V.banner("STEP 1.5  structural problem C - one deck is far too small to plot")
df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True)
d = df.deck.value_counts().sort_index()
print(d.to_string())
print("  -> deck T has 5 passengers. A 'rate' out of 5 people is noise,")
print("     so deck T is excluded from Figure 1 (and we say so on the chart).")

print("\nSTEP 1 done.\n")
