"""
STEP 5 - FIGURE 2 : the behavioural evidence

"Same money, opposite fate"
Transported rate against how much was spent, for all five onboard services.
The five split into two families that run in OPPOSITE directions, which rules
out "amount of money" as the explanation.

Two deliberate choices, both printed so they can be checked:
  * CryoSleep passengers are dropped. All 3,037 of them spent exactly 0 and
    81.8% were transported, so leaving them in dumps them into the "0" bin and
    manufactures a downward trend that has nothing to do with spending.
  * The three private-zone services are drawn as a band (their min-max envelope)
    with the individual lines inside. They behave almost identically, so five
    separate lines cost legibility for nothing.

    python step05_fig2_spending.py --data train.csv
Outputs: out/05_spend_curves.csv, out/05_cryosleep_effect.csv, out/fig2_spending.png
"""

import argparse
import numpy as np
import pandas as pd
import viz_lib as V

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
a = ap.parse_args()

df = V.load(a.data)
PRIVATE = ["RoomService", "Spa", "VRDeck"]
PUBLIC = ["FoodCourt", "ShoppingMall"]
BINS = [-1, 0, 50, 200, 500, 1000, 2000, 5000, np.inf]
LABELS = ["0", "1–50", "51–200", "201–500", "501–1k", "1k–2k", "2k–5k", ">5k"]

V.banner("STEP 5.1  why CryoSleep must be excluded")
cs = df[df.CryoSleep == True]
print(f"  CryoSleep passengers        : {len(cs):,}")
print(f"  their maximum total spend   : {cs.total_spend.max():.0f}  <- every one of them spent 0")
print(f"  their transported rate      : {cs.Transported.mean()*100:.1f}%")
z_all = df[df.total_spend == 0].Transported.mean() * 100
z_awake = df[(df.total_spend == 0) & (df.CryoSleep == False)].Transported.mean() * 100
print(f"\n  'spent 0' bin WITH CryoSleep : {z_all:.1f}%")
print(f"  'spent 0' bin WITHOUT them   : {z_awake:.1f}%")
print("  -> keeping them inflates the zero bin by ~16 points and fakes a clean")
print("     downward trend. That would be a textbook biased chart.")
V.save_table(pd.DataFrame({
    "group": ["zero-spend bin incl. CryoSleep", "zero-spend bin excl. CryoSleep",
              "CryoSleep passengers"],
    "transported_pct": [z_all, z_awake, cs.Transported.mean()*100],
    "n": [(df.total_spend == 0).sum(),
          ((df.total_spend == 0) & (df.CryoSleep == False)).sum(), len(cs)]}),
    "05_cryosleep_effect.csv", float_fmt="%.1f")

V.banner("STEP 5.2  the five curves (awake passengers only)")
awake = df[df.CryoSleep == False]
curves = {}
for c in PRIVATE + PUBLIC:
    b = pd.cut(awake[c], BINS, labels=LABELS)
    g = awake.groupby(b, observed=False).Transported.agg(["mean", "size"])
    curves[c] = g["mean"].values * 100

out = pd.DataFrame(curves, index=LABELS).round(1)
out.index.name = "spend_bin"
print(out.to_string())
print(f"\n  awake passengers n = {len(awake):,}")
print("  -> three services fall, two rise. Same money, opposite direction.")
V.save_table(out.reset_index(), "05_spend_curves.csv", float_fmt="%.1f")

# ------------------------------------------------------------------ figure --
W, H = 1280, 720
ML, MR, MT, MB = 128, 292, 284, 136
PW, PH = W - ML - MR, H - MT - MB
px = lambda i: ML + i * PW / (len(LABELS) - 1)
py = lambda v: MT + (100 - v) / 100 * PH

lo = np.min([curves[c] for c in PRIVATE], axis=0)
hi = np.max([curves[c] for c in PRIVATE], axis=0)
band = (" ".join(f"{px(i)},{py(v)}" for i, v in enumerate(hi)) + " " +
        " ".join(f"{px(i)},{py(v)}" for i, v in reversed(list(enumerate(lo)))))
thin = "".join(
    f'<polyline class="thin" points="{" ".join(f"{px(i)},{py(v)}" for i,v in enumerate(curves[c]))}"/>'
    for c in PRIVATE)

pub, dots, labels = "", "", ""
for c in PUBLIC:
    yv = curves[c]
    pub += f'<polyline class="pub" points="{" ".join(f"{px(i)},{py(v)}" for i,v in enumerate(yv))}"/>'
    dots += "".join(f'<circle cx="{px(i)}" cy="{py(v)}" r="7" class="pubdot"/>'
                    for i, v in enumerate(yv))
    labels += (f'<text x="{px(7)+26}" y="{py(yv[-1])}" class="lab">{c}</text>'
               f'<text x="{px(7)+26}" y="{py(yv[-1])+26}" class="sublab">'
               f'{yv[-1]:.0f}% at the top spend level</text>')

grid = "".join(
    f'<line x1="{ML-18}" y1="{py(v)}" x2="{ML+PW+18}" y2="{py(v)}" class="grid"/>'
    f'<text x="{ML-34}" y="{py(v)}" class="ytick">{v}%</text>'
    for v in range(0, 101, 25) if v != 50)
xticks = "".join(f'<text x="{px(i)}" y="{MT+PH+44}" class="xtick">{t}</text>'
                 for i, t in enumerate(LABELS))

body = f"""<div id="card">
  <div class="hd">
    <h1>Same money, opposite fate</h1>
    <div class="sub">Spending in the <b>private zone</b> goes with staying aboard.
      The very same spending in the <b>public zone</b> goes with disappearing.</div>
    <div class="note">Awake passengers only (n = {len(awake):,}) &mdash; CryoSleep excluded,
      since all {len(cs):,} of them spent exactly 0. Blue band = the range across the three
      private-zone services.</div>
  </div>
  <svg width="{W}" height="{H}" style="position:absolute;inset:0">
    {grid}
    <line x1="{ML-18}" y1="{py(50)}" x2="{ML+PW+18}" y2="{py(50)}" class="rule"/>
    <text x="{ML-34}" y="{py(50)}" class="ytick">50%</text>
    <text x="{ML-24}" y="{py(50)-9}" class="halftx">50% line</text>
    <polygon class="band" points="{band}"/>
    {thin}{pub}{dots}{labels}
    <text x="{px(7)+26}" y="{py(6)}" class="lab">RoomService, Spa,</text>
    <text x="{px(7)+26}" y="{py(6)+27}" class="lab">VRDeck</text>
    <text x="{px(7)+26}" y="{py(6)+53}" class="sublab">all three land near 0%</text>
    {xticks}
    <text x="{ML+PW/2}" y="{MT+PH+92}" class="axl">How much that passenger spent on the service (credits)</text>
    <text x="{ML-34}" y="{MT-24}" class="axl" style="text-anchor:start">Share transported</text>
  </svg>
</div>"""

css = f"""
  .ytick  {{ font-size:16px; fill:{V.MUTED}; text-anchor:end; dominant-baseline:middle; }}
  .xtick  {{ font-size:16px; fill:{V.MUTED}; text-anchor:middle; }}
  .halftx {{ font-size:15px; fill:{V.MUTED}; }}
  .band   {{ fill:{V.BLUE}; fill-opacity:.13; }}
  .thin   {{ fill:none; stroke:{V.BLUE}; stroke-width:2.6; stroke-linejoin:round;
             stroke-linecap:round; stroke-opacity:.9; }}
  .pub    {{ fill:none; stroke:{V.ORANGE}; stroke-width:3.5; stroke-linejoin:round;
             stroke-linecap:round; }}
  .pubdot {{ fill:{V.ORANGE}; stroke:{V.SURFACE}; stroke-width:2.5; }}
  .lab    {{ font-size:21px; font-weight:600; fill:{V.INK}; dominant-baseline:middle; }}
  .sublab {{ font-size:15px; fill:{V.MUTED}; dominant-baseline:middle; }}
"""
V.render(body, "fig2_spending.png", W, H, css)
print("\nSTEP 5 done.\n")
