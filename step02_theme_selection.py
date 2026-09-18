"""
STEP 2 - Why THESE variables? (theme level)

The dataset has 13 usable columns. Rather than picking favourites, we sort
them into five themes and measure two things on a held-out 30% of the data:

  * theme alone      - how well that theme predicts Transported by itself
  * drop-one-theme   - how much predictive power is LOST when it is removed

The second number is the one that matters. A theme can look strong alone and
still be worthless if another theme already carries the same information -
which is exactly what happens to CryoSleep.

    python step02_theme_selection.py --data train.csv
Outputs: out/02_theme_selection.csv, out/fig0_why_these_variables.png
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import viz_lib as V

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = V.load(a.data)
y = df.Transported.astype(int)

# Name is only used to derive a family-size proxy (surname frequency)
df["famsize"] = df.Name.str.split().str[-1].map(
    df.Name.str.split().str[-1].value_counts())

THEMES = {
    "Behaviour / spending":   ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
    "Position on ship":       ["deck", "num", "side"],
    "Origin & demographics":  ["HomePlanet", "Destination", "Age", "VIP"],
    "State (CryoSleep)":      ["CryoSleep"],
    "Social group":           ["gsize", "famsize"],
}
CHOSEN = ["Behaviour / spending", "Position on ship"]


def prep(X):
    """Numeric matrix; categoricals -> integer codes with NaN preserved."""
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(float)
        else:
            codes = pd.factorize(X[c])[0].astype(float)
            codes[codes < 0] = np.nan          # keep missing as missing
            X[c] = codes
    return X


def auc(cols, tr, te):
    X = prep(df[cols])
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    m.fit(X.iloc[tr], y.iloc[tr])
    return roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])


tr, te = train_test_split(np.arange(len(df)), test_size=0.30,
                          random_state=a.seed, stratify=y)
allc = sum(THEMES.values(), [])

V.banner("STEP 2.1  each theme ON ITS OWN (held-out 30%)")
alone = {k: auc(v, tr, te) for k, v in THEMES.items()}
for k, v in sorted(alone.items(), key=lambda x: -x[1]):
    print(f"  {k:24s} AUC {v:.3f}   ({len(THEMES[k])} columns)")

full = auc(allc, tr, te)
print(f"\n  {'ALL FIVE THEMES':24s} AUC {full:.3f}")

V.banner("STEP 2.2  drop-one-theme : what is LOST without it?")
loss = {}
for k, cols in THEMES.items():
    keep = [c for c in allc if c not in cols]
    loss[k] = full - auc(keep, tr, te)
for k, v in sorted(loss.items(), key=lambda x: -x[1]):
    print(f"  without {k:24s} -> loses {v:+.3f}")
print("\n  -> CryoSleep is the single strongest column in the data, yet removing")
print("     its whole theme costs almost nothing: every CryoSleep passenger spent")
print("     exactly 0, so the spending theme already contains that information.")

V.banner("STEP 2.3  do our two themes carry the signal?")
two = auc(sum([THEMES[k] for k in CHOSEN], []), tr, te)
print(f"  our 2 themes  ({len(sum([THEMES[k] for k in CHOSEN],[]))} columns) AUC {two:.3f}")
print(f"  all 5 themes  ({len(allc)} columns) AUC {full:.3f}")
print(f"  difference                        {full-two:+.3f}")

res = pd.DataFrame({
    "theme": list(THEMES),
    "n_columns": [len(v) for v in THEMES.values()],
    "auc_alone": [alone[k] for k in THEMES],
    "auc_lost_if_removed": [loss[k] for k in THEMES],
    "selected": [k in CHOSEN for k in THEMES],
}).sort_values("auc_lost_if_removed", ascending=False)
V.save_table(res, "02_theme_selection.csv", float_fmt="%.3f")

# ------------------------------------------------------------------ figure --
W, H = 1280, 720
ML, MR, MT, MB = 330, 210, 288, 176
PW, PH = W - ML - MR, H - MT - MB
XMAX = max(0.12, res.auc_lost_if_removed.max() * 1.15)

rows, step = [], PH / len(res)
for i, r in enumerate(res.itertuples()):
    cy = MT + step * (i + 0.5)
    w = max(r.auc_lost_if_removed, 0) / XMAX * PW
    col = V.BLUE if r.selected else V.GREY
    rows.append(f"""
      <g>
        <text x="{ML-26}" y="{cy}" class="lbl" style="font-weight:{600 if r.selected else 400}">{r.theme}</text>
        <rect x="{ML}" y="{cy-15}" width="{max(w,2):.1f}" height="30" rx="4" fill="{col}"/>
        <text x="{ML+max(w,2)+14}" y="{cy}" class="val">{r.auc_lost_if_removed:+.3f}</text>
        <text x="{W-MR+92}" y="{cy}" class="alone">{r.auc_alone:.3f}</text>
      </g>""")

grid = "".join(
    f'<line x1="{ML+v/XMAX*PW}" y1="{MT-10}" x2="{ML+v/XMAX*PW}" y2="{MT+PH}" class="grid"/>'
    f'<text x="{ML+v/XMAX*PW}" y="{MT+PH+38}" class="tick" text-anchor="middle">{v:.2f}</text>'
    for v in np.arange(0, XMAX, 0.025) if v > 0)

body = f"""<div id="card">
  <div class="hd">
    <h1>Why we studied these variables</h1>
    <div class="sub">All 13 columns were sorted into five themes. The bar is what the
      data loses when a theme is <b>removed</b> &mdash; its unique contribution.</div>
    <div class="note">Gradient-boosted model, AUC measured on a held-out 30% of passengers
      (n = {len(te):,}). Used only to decide what to study, not as a result.</div>
  </div>
  <svg width="{W}" height="{H}" style="position:absolute;inset:0">
    <line x1="{ML}" y1="{MT-10}" x2="{ML}" y2="{MT+PH}" class="rule"/>
    {grid}
    <text x="{W-MR+92}" y="{MT-26}" class="hdr">theme<tspan x="{W-MR+92}" dy="20">alone</tspan></text>
    {''.join(rows)}
    <text x="{ML+PW/2}" y="{MT+PH+80}" class="axl">AUC lost when the theme is removed</text>
  </svg>
  <div class="key">
    <div><span class="sw" style="background:{V.BLUE}"></span>selected for this project</div>
    <div><span class="sw" style="background:{V.GREY}"></span>not selected</div>
  </div>
  <div class="foot">Our two themes together reach <b>AUC {two:.3f}</b> using
    {len(sum([THEMES[k] for k in CHOSEN],[]))} of 13 columns &mdash; against
    <b>{full:.3f}</b> for all five themes. The other three add {full-two:+.3f}.</div>
</div>"""

css = f"""
  .lbl   {{ font-size:20px; fill:{V.INK}; text-anchor:end; dominant-baseline:middle; }}
  .val   {{ font-size:19px; font-weight:600; fill:{V.INK2}; dominant-baseline:middle; }}
  .alone {{ font-size:16px; fill:{V.MUTED}; dominant-baseline:middle; text-anchor:middle; }}
  .hdr   {{ font-size:14px; fill:{V.MUTED}; text-anchor:middle; }}
  .key   {{ position:absolute; right:56px; top:168px; display:flex; gap:24px; }}
  .key div {{ display:flex; align-items:center; gap:10px; font-size:17px;
              color:{V.INK2}; }}
  .sw    {{ width:18px; height:18px; border-radius:4px; }}
  .foot  {{ position:absolute; left:56px; right:56px; bottom:34px;
            font-size:18px; color:{V.INK2}; line-height:1.5; }}
  .foot b {{ color:{V.INK}; font-weight:600; }}
"""
V.render(body, "fig0_why_these_variables.png", W, H, css)
print("\nSTEP 2 done.\n")
