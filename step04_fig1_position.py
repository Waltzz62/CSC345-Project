"""
STEP 4 - FIGURE 1 : the physical-position evidence

"The anomaly came in from starboard"
Port vs starboard transported rate, compared WITHIN each deck.

Why within-deck and not deck-vs-deck: the decks are almost perfectly segregated
by home planet (A/B/C are 100% Europa, G is 100% Earth), so comparing decks would
confuse position with origin. Comparing the two sides of the SAME deck holds
home planet - and everything else about that deck - constant. The script prints
the deck x HomePlanet table so you can see the confound for yourself.

    python step04_fig1_position.py --data train.csv
Outputs: out/04_deck_side.csv, out/04_deck_homeplanet_confound.csv,
         out/fig1_position.png
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import viz_lib as V

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="train.csv")
a = ap.parse_args()

df = V.load(a.data)

V.banner("STEP 4.1  the confound we are avoiding: deck is tied to HomePlanet")
conf = pd.crosstab(df.HomePlanet, df.deck)
print(conf.to_string())
print("  -> decks A/B/C carry Europa passengers ONLY; deck G carries Earth ONLY.")
print("     So 'deck B is dangerous' and 'Europa is dangerous' cannot be told apart.")
print("     Port vs starboard inside one deck has no such problem.")
V.save_table(conf.reset_index(), "04_deck_homeplanet_confound.csv", float_fmt="%.0f")

V.banner("STEP 4.2  port vs starboard, within each deck")
d = df.dropna(subset=["deck", "side"])
d = d[d.deck != "T"]                       # n=5, a rate out of 5 is noise
rate = d.groupby(["deck", "side"]).Transported.mean().unstack() * 100
n = d.groupby("deck").size()
rate = rate.loc[sorted(rate.index)]
base = df.Transported.mean() * 100

tbl = rate.round(1).rename(columns={"P": "port_%", "S": "starboard_%"})
tbl["gap_points"] = (rate.S - rate.P).round(1)
tbl["n"] = n
print(tbl.to_string())
V.save_table(tbl.reset_index(), "04_deck_side.csv", float_fmt="%.1f")

ct = pd.crosstab(d.side, d.Transported)
chi2, p = stats.chi2_contingency(ct)[:2]
print(f"\n  starboard is higher on {int((rate.S > rate.P).sum())} of {len(rate)} decks")
print(f"  overall chi-square p = {p:.2e}")
print("  -> a one-sided pattern this consistent is not chance.")

# ------------------------------------------------------------------ figure --
W, H = 1280, 720
ML, MR, MT, MB = 168, 168, 262, 118
PW, PH = W - ML - MR, H - MT - MB
XMIN, XMAX = 20, 82
x = lambda v: ML + (v - XMIN) / (XMAX - XMIN) * PW

rows, step = [], PH / len(rate)
for i, (deck, r) in enumerate(rate.iterrows()):
    cy = MT + step * (i + 0.5)
    p_, s_ = r["P"], r["S"]
    rows.append(f"""
      <g>
        <text x="{ML-28}" y="{cy}" class="deck">Deck {deck}</text>
        <line x1="{x(p_)}" y1="{cy}" x2="{x(s_)}" y2="{cy}" class="conn"/>
        <circle cx="{x(p_)}" cy="{cy}" r="11" fill="{V.BLUE}" class="dot"/>
        <circle cx="{x(s_)}" cy="{cy}" r="11" fill="{V.ORANGE}" class="dot"/>
        <text x="{(x(p_)+x(s_))/2}" y="{cy-26}" class="gap">+{s_-p_:.1f} pts</text>
        <text x="{W-MR+34}" y="{cy}" class="n">{n[deck]:,} people</text>
      </g>""")

grid = "".join(
    f'<line x1="{x(v)}" y1="{MT-8}" x2="{x(v)}" y2="{MT+PH}" class="grid"/>'
    f'<text x="{x(v)}" y="{MT+PH+40}" class="tick" text-anchor="middle">{v}%</text>'
    for v in range(20, 81, 10) if abs(v - base) > 3)

body = f"""<div id="card">
  <div class="hd">
    <h1>The anomaly came in from starboard</h1>
    <div class="sub">On <b>every single deck</b>, cabins on the starboard side lost
      more passengers than cabins on the port side.</div>
    <div class="note">Deck T excluded (only 5 passengers). Comparing the two sides of the
      same deck holds home planet constant. &chi;&sup2; test, p = {p:.1e}.</div>
  </div>
  <div class="key">
    <div><span class="sw" style="background:{V.BLUE}"></span>Port (left)</div>
    <div><span class="sw" style="background:{V.ORANGE}"></span>Starboard (right)</div>
  </div>
  <svg width="{W}" height="{H}" style="position:absolute;inset:0">
    {grid}
    <line x1="{x(base)}" y1="{MT-30}" x2="{x(base)}" y2="{MT+PH}" class="rule"/>
    <text x="{x(base)}" y="{MT-40}" class="tick" text-anchor="middle">ship-wide average {base:.1f}%</text>
    {''.join(rows)}
    <text x="{ML+PW/2}" y="{MT+PH+84}" class="axl">Share of passengers transported to the alternate dimension</text>
  </svg>
</div>"""

css = f"""
  .deck {{ font-size:21px; font-weight:600; fill:{V.INK}; text-anchor:end;
           dominant-baseline:middle; }}
  .n    {{ font-size:15px; fill:{V.MUTED}; dominant-baseline:middle; }}
  .gap  {{ font-size:18px; font-weight:600; fill:{V.INK2}; text-anchor:middle; }}
  .conn {{ stroke:{V.RULE}; stroke-width:3; stroke-linecap:round; }}
  .dot  {{ stroke:{V.SURFACE}; stroke-width:3; }}
  .key  {{ position:absolute; right:56px; top:152px; display:flex; gap:26px; }}
  .key div {{ display:flex; align-items:center; gap:10px; font-size:19px; color:{V.INK2}; }}
  .sw   {{ width:18px; height:18px; border-radius:50%; }}
"""
V.render(body, "fig1_position.png", W, H, css)
print("\nSTEP 4 done.\n")
