"""
CSC345 Project Phase 1 - shared helpers
Spaceship Titanic (Kaggle)

Holds the design tokens and the HTML/SVG -> PNG renderer that every figure
script uses, so all figures come out as one visual system.

Rendering works by laying the chart out as HTML + inline SVG and letting
headless Chromium screenshot it at 2x. That buys web-grade typography and
spacing in a static PNG you can drop straight into a slide.
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- tokens ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a84"
GRID, RULE = "#eceae3", "#d6d4cb"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated categorical slots 1 & 2
GREY = "#c3c2b7"                      # for de-emphasised / rejected items
FONT = "Carlito, Calibri, 'Liberation Sans', system-ui, sans-serif"

OUT = pathlib.Path("out")


# ------------------------------------------------------------- utilities ----
def load(path):
    """Read the Kaggle file and add every derived column the project uses."""
    df = pd.read_csv(path)

    # Cabin is a composite field: deck / room number / side (P=port, S=starboard)
    df[["deck", "num", "side"]] = df["Cabin"].str.split("/", expand=True)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")

    # PassengerId is gggg_pp : the first part is the travelling group
    df["group"] = df["PassengerId"].str.split("_").str[0]
    df["gsize"] = df["group"].map(df.groupby("group").size())

    # engineered spend features - the two service families
    df["lux"] = df[["RoomService", "Spa", "VRDeck"]].sum(axis=1)   # private zone
    df["com"] = df[["FoodCourt", "ShoppingMall"]].sum(axis=1)      # public zone
    df["total_spend"] = df[["RoomService", "FoodCourt", "ShoppingMall",
                            "Spa", "VRDeck"]].sum(axis=1)
    return df


def cramers_v(a, b):
    """Association strength for two categorical variables. Unlike Pearson it
    does not assume any ordering, so it is valid for nominal fields like deck."""
    t = pd.crosstab(a, b)
    chi2, p = stats.chi2_contingency(t)[:2]
    return np.sqrt(chi2 / t.values.sum()), p


def save_table(df, name, float_fmt="%.4f"):
    """Write a result table to out/ so there is a file to show, not just a print."""
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False, float_format=float_fmt)
    print(f"  [saved] {path}")
    return path


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- render ----
BASE_CSS = f"""
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:{SURFACE}; font-family:{FONT};
          -webkit-font-smoothing:antialiased; }}
  #card {{ position:relative; background:{SURFACE}; }}
  .hd {{ position:absolute; left:56px; right:56px; top:52px; }}
  h1 {{ font-size:42px; font-weight:700; letter-spacing:-.9px; color:{INK};
        line-height:1.1; }}
  .sub {{ font-size:19px; color:{INK2}; line-height:1.55; margin-top:16px;
          max-width:1180px; }}
  .sub b {{ font-weight:600; color:{INK}; }}
  .note {{ font-size:15px; color:{MUTED}; margin-top:10px; max-width:1180px; }}
  text {{ font-family:{FONT}; }}
  .grid   {{ stroke:{GRID}; stroke-width:1; }}
  .rule   {{ stroke:{RULE}; stroke-width:1.5; }}
  .tick   {{ font-size:16px; fill:{MUTED}; }}
  .axl    {{ font-size:17px; fill:{INK2}; text-anchor:middle; }}
"""


def render(body_html, out_png, w=1280, h=720, extra_css=""):
    """Lay the chart out in Chromium and screenshot #card at 2x."""
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>{BASE_CSS}"
            f"{extra_css}\n#card {{ width:{w}px; height:{h}px; }}"
            f"</style></head><body>{body_html}</body></html>")

    tmp = OUT / ("_" + pathlib.Path(out_png).stem + ".html")
    tmp.write_text(html, encoding="utf-8")

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.goto(tmp.resolve().as_uri())
        pg.wait_for_timeout(300)
        pg.locator("#card").screenshot(path=str(OUT / out_png))
        b.close()
    tmp.unlink()
    print(f"  [saved] {OUT / out_png}  ({w*2}x{h*2}px)")
