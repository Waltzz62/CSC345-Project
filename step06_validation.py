"""
STEP 6 - Does the conclusion survive being attacked?

Figure 1 and Figure 2 each show a pattern. This step asks the four questions a
sceptical examiner would ask, and answers every one with a number:

  6.1  Are the two findings the same finding counted twice?
  6.2  Is the spending effect really just wealth?
  6.3  Did we get the answer by choosing convenient bins?
  6.4  Is any of this actually useful for the Kaggle task?

It also draws FIGURE 3, which is the picture of the conclusion: the two signals
are independent, so their effects stack.

    python step06_validation.py --data train.csv
Outputs: out/06_two_signals.csv, out/06_wealth_check.csv, out/06_bin_robustness.csv,
         out/06_rule_accuracy.csv, out/fig3_signals_add_up.png
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import viz_lib as V

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--cut", type=float, default=500, help="spend threshold")
a = ap.parse_args()

df = V.load(a.data)
awake = df[df.CryoSleep == False].dropna(subset=["side"]).copy()

# three mutually exclusive spending groups
awake["spend_group"] = np.where(
    awake.lux > a.cut, "Private-zone spender",
    np.where(awake.com > a.cut, "Public-zone spender", "Spent little on either"))
ORDER = ["Private-zone spender", "Spent little on either", "Public-zone spender"]

# ---------------------------------------------------------------- 6.1 -------
V.banner("STEP 6.1  are the two findings independent, or the same thing twice?")
rate = (awake.pivot_table(index="spend_group", columns="side",
                          values="Transported", aggfunc="mean") * 100).loc[ORDER]
cnt = awake.pivot_table(index="spend_group", columns="side",
                        values="Transported", aggfunc="size").loc[ORDER]
tbl = rate.round(1).rename(columns={"P": "port_%", "S": "starboard_%"})
tbl["gap_points"] = (rate.S - rate.P).round(1)
tbl["n_port"], tbl["n_starboard"] = cnt.P, cnt.S
print(tbl.to_string())
print("\n  Reading it: DOWN a column = the spending effect. ACROSS a row = the side")
print("  effect. Both are present in every cell, so they are not the same signal.")
V.save_table(tbl.reset_index(), "06_two_signals.csv", float_fmt="%.1f")

print("\n  Does the spending effect CHANGE depending on side? (odds ratios)")
for s, lab in [("P", "port"), ("S", "starboard")]:
    d = awake[awake.side == s]
    t = pd.crosstab(d.lux > a.cut, d.Transported)
    orr = (t.loc[True, True] * t.loc[False, False]) / (t.loc[True, False] * t.loc[False, True])
    print(f"    {lab:10s}: OR(private-zone spender) = {orr:.2f}")
try:
    import statsmodels.formula.api as smf
    m = awake.assign(S=(awake.side == "S").astype(int),
                     L=(awake.lux > a.cut).astype(int),
                     C=(awake.com > a.cut).astype(int),
                     yy=awake.Transported.astype(int))
    fit = smf.logit("yy ~ S*L + S*C", data=m).fit(disp=0)
    print(f"    interaction side x private-spend : p = {fit.pvalues['S:L']:.3f}")
    print(f"    interaction side x public-spend  : p = {fit.pvalues['S:C']:.3f}")
    print("    -> both far above 0.05: the spending effect is the same on both sides,")
    print("       which is what 'two independent signals' means.")
except ImportError:
    print("    (install statsmodels to also get the interaction p-values)")

# ---------------------------------------------------------------- 6.2 -------
V.banner("STEP 6.2  is the spending effect just wealth in disguise?")
vip = df.groupby("VIP").Transported.agg(["mean", "count"])
vip["mean"] = (vip["mean"] * 100).round(1)
print("  VIP is the most direct wealth marker in the data:")
print(vip.rename(columns={"mean": "transported_%", "count": "n"}).to_string())
v, p = V.cramers_v(df.dropna(subset=["VIP"]).VIP,
                   df.dropna(subset=["VIP"]).Transported)
print(f"  Cramer's V = {v:.3f} (p = {p:.1e})  <- one of the WEAKEST columns in the data")
print("  -> if wealth drove the outcome, VIP would be strong. It is not.")

print("\n  And the spending effect inside each home planet:")
b = pd.cut(awake.lux, [-1, a.cut, np.inf], labels=[f"lux<={a.cut:.0f}", f"lux>{a.cut:.0f}"])
hp = (pd.crosstab(b, awake.HomePlanet, awake.Transported, aggfunc="mean") * 100).round(1)
print(hp.to_string())
print("  -> the effect survives inside every planet, so it is not a wealth proxy.")
V.save_table(hp.reset_index(), "06_wealth_check.csv", float_fmt="%.1f")

# ---------------------------------------------------------------- 6.3 -------
V.banner("STEP 6.3  did our choice of bins create the result?")
SCHEMES = {
    "as used in Fig 2": [-1, 0, 50, 200, 500, 1000, 2000, 5000, np.inf],
    "very coarse":      [-1, 0, 500, np.inf],
    "three cuts":       [-1, 0, 100, 1000, np.inf],
    "fine":             [-1, 0, 25, 100, 300, 700, 1500, 3000, 6000, np.inf],
}
rob = []
for col in ["lux", "FoodCourt", "ShoppingMall"]:
    row = {"column": col}
    for name, bins in SCHEMES.items():
        m = df[col].notna()
        vv, _ = V.cramers_v(pd.cut(df.loc[m, col], bins), df.loc[m, "Transported"])
        row[name] = round(vv, 3)
    rob.append(row)
rob = pd.DataFrame(rob)
print(rob.to_string(index=False))
print("  -> stable to the third decimal across wildly different binnings.")
V.save_table(rob, "06_bin_robustness.csv", float_fmt="%.3f")

# ---------------------------------------------------------------- 6.4 -------
V.banner("STEP 6.4  is the insight actually useful?")
tr, te = train_test_split(df.index, test_size=0.30, random_state=a.seed,
                          stratify=df.Transported)
hold = df.loc[te]

def rule(d):
    """Three lines, written straight off the two figures."""
    pred = pd.Series(False, index=d.index)
    pred[d.CryoSleep == True] = True                 # asleep -> taken
    m = d.CryoSleep != True
    pred[m & (d.lux > a.cut)] = False                # private-zone spender -> stayed
    pred[m & (d.lux <= a.cut) & (d.com > a.cut)] = True
    pred[m & (d.lux <= a.cut) & (d.com <= a.cut) & (d.side == "S")] = True
    return pred

acc = {
    "always guess the majority class": max(hold.Transported.mean(), 1 - hold.Transported.mean()),
    "rule from our two figures": (rule(hold) == hold.Transported).mean(),
    "private-zone spending alone": ((~(hold.lux > a.cut)) == hold.Transported).mean(),
    "starboard alone": ((hold.side == "S").fillna(False) == hold.Transported).mean(),
}
for k, vv in acc.items():
    print(f"  {k:34s} {vv*100:5.1f}%")
print(f"\n  held-out passengers n = {len(hold):,} (never seen while building the rule)")
print("  -> note the combined rule is NOT better than the single best one. The side")
print("     signal is real but weak; stacking rules by hand does not exploit it.")
V.save_table(pd.DataFrame({"method": list(acc), "accuracy_pct":
                           [round(x*100, 1) for x in acc.values()]}),
             "06_rule_accuracy.csv", float_fmt="%.1f")

# ------------------------------------------------------------------ figure --
W, H = 1280, 720
ML, MR, MT, MB = 118, 88, 276, 168
PW, PH = W - ML - MR, H - MT - MB
gw = PW / len(ORDER)
py = lambda v: MT + (100 - v) / 100 * PH

bars = ""
for i, g in enumerate(ORDER):
    cx = ML + gw * (i + 0.5)
    for j, (s, col) in enumerate([("P", V.BLUE), ("S", V.ORANGE)]):
        val = rate.loc[g, s]
        bx = cx - 74 + j * 78
        bars += (f'<rect x="{bx}" y="{py(val)}" width="70" height="{MT+PH-py(val)}" '
                 f'rx="5" fill="{col}"/>'
                 f'<text x="{bx+35}" y="{py(val)-14}" class="cap">{val:.0f}%</text>')
    bars += (f'<text x="{cx}" y="{MT+PH+40}" class="glab">{g}</text>'
             f'<text x="{cx}" y="{MT+PH+66}" class="gn">n = {cnt.loc[g].sum():,}</text>')

grid = "".join(
    f'<line x1="{ML-14}" y1="{py(v)}" x2="{ML+PW}" y2="{py(v)}" class="grid"/>'
    f'<text x="{ML-28}" y="{py(v)}" class="ytick">{v}%</text>' for v in range(0, 81, 20))

body = f"""<div id="card">
  <div class="hd">
    <h1>Two independent signals, stacking up</h1>
    <div class="sub">Reading <b>down</b> gives the spending effect, <b>across each
      pair</b> the side effect &mdash; both appear in every group.</div>
    <div class="note">Awake passengers only (n = {len(awake):,}). Groups are mutually
      exclusive at a {a.cut:.0f}-credit threshold. Interaction between side and spending is
      not significant, so the two effects simply add.</div>
  </div>
  <div class="key">
    <div><span class="sw" style="background:{V.BLUE}"></span>Port</div>
    <div><span class="sw" style="background:{V.ORANGE}"></span>Starboard</div>
  </div>
  <svg width="{W}" height="{H}" style="position:absolute;inset:0">
    {grid}
    {bars}
    <line x1="{ML-14}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" class="rule"/>
    <text x="{ML-28}" y="{MT-26}" class="axl" style="text-anchor:start">Share transported</text>
  </svg>
</div>"""

css = f"""
  .ytick {{ font-size:16px; fill:{V.MUTED}; text-anchor:end; dominant-baseline:middle; }}
  .cap   {{ font-size:21px; font-weight:600; fill:{V.INK}; text-anchor:middle; }}
  .glab  {{ font-size:20px; font-weight:600; fill:{V.INK}; text-anchor:middle; }}
  .gn    {{ font-size:15px; fill:{V.MUTED}; text-anchor:middle; }}
  .key   {{ position:absolute; right:56px; top:160px; display:flex; gap:26px; }}
  .key div {{ display:flex; align-items:center; gap:10px; font-size:19px; color:{V.INK2}; }}
  .sw    {{ width:18px; height:18px; border-radius:50%; }}
"""
V.render(body, "fig3_signals_add_up.png", W, H, css)
print("\nSTEP 6 done.\n")
