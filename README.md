# CSC345 Project Phase 1 — Spaceship Titanic: EDA & Data Visualization

Every number in our presentation is produced by a script in this repository.
Nothing is typed in by hand. Run the pipeline and you get the same figures and
the same tables we show.

Dataset: <https://www.kaggle.com/competitions/spaceship-titanic>

## Setup

```bash
pip install pandas numpy scipy scikit-learn statsmodels playwright
playwright install chromium
```

Put Kaggle's `train.csv`, `test.csv` and `sample_submission.csv` next to the scripts.

## Run

```bash
python step01_audit.py            --data train.csv
python step02_theme_selection.py  --data train.csv
python step03_column_screening.py --data train.csv
python step04_fig1_position.py    --data train.csv
python step05_fig2_spending.py    --data train.csv
python step06_validation.py       --data train.csv
python step07_train_test_check.py --train train.csv --test test.csv
```

Everything lands in `out/` — PNG figures plus a CSV for every table printed to
the console, so each claim has a file behind it.

## The pipeline, and what question each step answers

| Step | Question it answers | Output |
|---|---|---|
| 01 | What is in the data, and what is broken about it? | 3 CSVs |
| 02 | **Why did we study these variables and not others?** | `fig0` + CSV |
| 03 | Which columns inside those themes, and measured how? | 2 CSVs |
| 04 | **Insight 1** — does position on the ship matter? | `fig1` + 2 CSVs |
| 05 | **Insight 2** — does spending behaviour matter? | `fig2` + 2 CSVs |
| 06 | **Does the conclusion survive being attacked?** | `fig3` + 4 CSVs |
| 07 | What are the other two Kaggle files for, and does the finding generalise? | 2 CSVs |

### Step 02 — how we chose what to study

The 13 usable columns are sorted into five themes. Each theme is scored two
ways on a held-out 30% of passengers: how well it predicts alone, and how much
is lost when it is removed. The second number is what we select on, because a
theme can be strong alone and still be redundant.

That is exactly what happens to CryoSleep: it is the single strongest column in
the dataset, yet deleting its whole theme costs 0.003 AUC — every CryoSleep
passenger spent exactly 0, so the spending theme already carries the information.

Our two themes reach AUC 0.891 using 8 of 13 columns, against 0.898 for all five.

### Step 03 — why we do not screen with plain correlation

Two structural problems with the standard approach on this dataset:

* `deck` is **nominal**. Turning A–G into 0–6 invents an order that does not
  exist. Step 03 shuffles the labels ten times: Pearson *r* swings from −0.19 to
  +0.14 on identical data, while Cramér's V returns 0.215 every time.
* `ShoppingMall` has a **U-shaped** relationship with the target — the rate falls
  and then rises. A straight line through a U is flat, so Pearson reports
  *r* = 0.010, p = 0.35 ("not significant"), while a chi-square on bins reports
  V = 0.310, p = 1e−172.

So we report both measures and say which one we act on, rather than quietly
using whichever is more flattering.

### Steps 04–05 — the two figures, and the bias controls in them

* **Figure 1** compares port vs starboard **within each deck**. The decks are
  almost perfectly segregated by home planet (A/B/C are 100% Europa, G is 100%
  Earth), so a deck-vs-deck comparison would confuse position with origin.
  Deck T (n = 5) is excluded and the chart says so.
* **Figure 2** excludes CryoSleep passengers. All 3,037 of them spent exactly 0,
  so including them lifts the zero bin from 61.8% to 78.4% and manufactures a
  trend that is not about spending at all. The chart says so.

### Step 06 — the four attacks

1. *Are the two findings the same finding twice?* No — the spending effect is
   the same size on both sides (interaction p ≈ 0.27), so the effects add.
2. *Is spending just wealth?* No — VIP, the direct wealth marker, is one of the
   weakest columns (V = 0.037), and the effect holds inside every home planet.
3. *Did the bins create the result?* No — Cramér's V is stable to the third
   decimal across four very different binning schemes.
4. *Is it useful?* A three-line rule read straight off the figures scores 75.9%
   on held-out passengers against a 50.3% baseline.

## What we claim, and what we do not

**Supported by the data**

* Starboard cabins lost more passengers than port cabins on all 7 decks (p = 1.5e−21)
* The five services split into two families that move in opposite directions
* The two signals are independent and their effects add
* The outcome is not explained by wealth

**Hypotheses, not findings**

* That the anomaly struck from the starboard side
* That spending is a proxy for which zone a passenger occupied
* That the public-zone venues sat inside the affected region

**Known limitation.** CryoSleep passengers were sealed in their cabins and have
the *highest* transported rate (81.8%), while RoomService users — also in their
cabins — have the *lowest* (16.9%). Our zone reading does not explain this.
Testing it properly would need a deck plan giving the location of each service,
which the dataset does not contain.

## Files

```
viz_lib.py                 design tokens + the HTML/SVG -> PNG renderer
step01_audit.py            data audit
step02_theme_selection.py  theme selection      -> fig0_why_these_variables.png
step03_column_screening.py column screening
step04_fig1_position.py    Insight 1            -> fig1_position.png
step05_fig2_spending.py    Insight 2            -> fig2_spending.png
step06_validation.py       validation           -> fig3_signals_add_up.png
step07_train_test_check.py train/test comparison
out/                       all figures and result tables
```

Figures are laid out as HTML + inline SVG and screenshotted by headless
Chromium at 2× (2560×1440 PNG), which gives typographic control that a plotting
library does not, while still producing a plain image for the slides. The two
chart colours are a contrast- and colour-blindness-checked pair used across all
four figures so they read as one set.
