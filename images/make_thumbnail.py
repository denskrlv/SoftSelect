"""Generate images/thumbnail.svg, the README figure illustrating the PFSS pipeline.

Run from the repository root:  python images/make_thumbnail.py
"""

from pathlib import Path

W, H = 1600, 620
FONT = "'Latin Modern Roman', 'CMU Serif', 'Times New Roman', Times, serif"

INK = "#1A1A1A"
GREY = "#555555"
RULE = "#BDBDBD"
DARK_BLUE = (8, 81, 156)     # ColorBrewer Blues, darkest
LIGHT_BLUE = (158, 202, 225)  # ColorBrewer Blues, light

PANEL_W, PANEL_TOP, PANEL_BOTTOM = 300, 85, 590
PANEL_X = [60, 453, 847, 1240]

# Illustrative values: feature probabilities (sorted), sampled subsets, results
PROBS = [0.24, 0.19, 0.15, 0.12, 0.11, 0.08, 0.06, 0.05]
SUBSETS = [("1", {0, 1, 2, 4}), ("2", {0, 1, 3, 5}), ("K", {0, 2, 3, 7})]
ROW_Y = [185, 295, 465]  # vertical centres of the subset / ensemble rows
FREQUENCY = [1.0, 0.8, 0.68, 0.6, 0.45, 0.38, 0.22, 0.18]
ACCURACY_BY_K = [(0.30, 9), (0.58, 7), (0.78, 6), (0.88, 5), (0.86, 4)]  # (relative height, error) per k
CLASSIFIERS = ["RF", "GB", "AdaBoost", "LR", "SVM", "LDA"]

TITLES = ["Feature scoring", "Probabilistic sampling", "Ensemble training", "Aggregation"]


def blue(i):
    """Colour for feature i: stronger features are darker."""
    t = i / (len(PROBS) - 1)
    r, g, b = (round(d + (l - d) * t) for d, l in zip(DARK_BLUE, LIGHT_BLUE))
    return f"#{r:02X}{g:02X}{b:02X}"


def text(x, y, s, size=17, fill=INK, anchor="start", italic=False, weight=400, extra=""):
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style}{extra}>{s}</text>')


def sub(base, index, size=12):
    """Italic symbol with a subscript, e.g. sub('f', '1') -> f_1."""
    return f'{base}<tspan baseline-shift="sub" font-size="{size}">{index}</tspan>'


def rect(x, y, w, h, fill="none", stroke=None, width=1):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{stroke_attr}/>'


def line(x1, y1, x2, y2, stroke=INK, width=1.2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def vdots(cx, top):
    return [f'<circle cx="{cx}" cy="{top + d}" r="2.2" fill="{INK}"/>' for d in (0, 12, 24)]


def connector(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    return (f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2 - 3},{y2}" fill="none" '
            f'stroke="{INK}" stroke-width="1.2" marker-end="url(#arrow)"/>')


def scoring_panel(px):
    out = []
    x0, scale, y_axis = px + 70, 800, 478  # 800 px per unit of probability
    for i, p in enumerate(PROBS):
        cy = 125 + i * 44
        out.append(text(px + 56, cy + 6, sub("f", i + 1), anchor="end", italic=True))
        out.append(rect(x0, cy - 11, scale * p, 22, fill=blue(i)))
    out.append(line(x0, 100, x0, y_axis))
    out.append(line(x0, y_axis, x0 + 210, y_axis))
    for tick in (0, 0.1, 0.2):
        x = x0 + scale * tick
        out.append(line(x, y_axis, x, y_axis + 6))
        out.append(text(x, y_axis + 24, f"{tick:g}", size=15, anchor="middle"))
    out.append(text(x0 + 105, y_axis + 50, sub("p", "i"), italic=True, anchor="middle"))
    out.append(text(px + PANEL_W / 2, 568,
                    f'{sub("p", "i")} = {sub("s", "i")} / &#931;<tspan baseline-shift="sub" font-size="12">j</tspan> {sub("s", "j")}',
                    size=19, italic=True, anchor="middle"))
    return out


def sampling_panel(px):
    out = []
    cell, gap = 26, 4
    x0 = px + 54
    for i in range(8):
        out.append(text(x0 + i * (cell + gap) + cell / 2, 130, sub("f", i + 1, size=10), size=14,
                        italic=True, anchor="middle"))
    for (label, chosen), cy in zip(SUBSETS, ROW_Y):
        out.append(text(px + 44, cy + 6, sub("S", label), anchor="end", italic=True))
        for i in range(8):
            x = x0 + i * (cell + gap)
            out.append(rect(x, cy - cell / 2, cell, cell, fill=blue(i) if i in chosen else "#FFFFFF",
                            stroke=INK if i in chosen else RULE))
    out += vdots(x0 + (8 * cell + 7 * gap) / 2, 370)
    out.append(text(px + PANEL_W / 2, 545, "|<tspan font-style=\"italic\">S</tspan><tspan baseline-shift=\"sub\" "
                    "font-size=\"12\" font-style=\"italic\">k</tspan>| = <tspan font-style=\"italic\">k</tspan>, "
                    "drawn &#8733; <tspan font-style=\"italic\">p</tspan>", anchor="middle"))
    out.append(text(px + PANEL_W / 2, 568, "without replacement", anchor="middle"))
    return out


def training_panel(px):
    out = []
    cw, ch, gap = 72, 22, 6
    for (label, _), cy in zip(SUBSETS, ROW_Y):
        top = cy - 42
        out.append(rect(px + 24, top, PANEL_W - 48, 84, fill="#FFFFFF", stroke=INK))
        out.append(text(px + 34, top + 20, sub("E", label), size=16, italic=True))
        x0 = px + (PANEL_W - (3 * cw + 2 * gap)) / 2
        for j, name in enumerate(CLASSIFIERS):
            x = x0 + (j % 3) * (cw + gap)
            y = top + 30 + (j // 3) * (ch + 4)
            out.append(rect(x, y, cw, ch, fill="#F3F6FA", stroke=RULE))
            out.append(text(x + cw / 2, y + 16, name, size=14, anchor="middle"))
    out += vdots(px + PANEL_W / 2, 370)
    out.append(text(px + PANEL_W / 2, 545, "one ensemble per subset", anchor="middle"))
    out.append(text(px + PANEL_W / 2, 568, f'{sub("E", "k")} trained on {sub("S", "k")}', italic=True,
                    anchor="middle"))
    return out


def aggregation_panel(px):
    out = []
    # Top: mean accuracy +/- std against subset size k
    left, right, top, bottom = px + 62, px + 276, 125, 235
    out.append(line(left, top, left, bottom))
    out.append(line(left, bottom, right, bottom))
    out.append(text(px + 34, (top + bottom) / 2, "Accuracy", size=15, anchor="middle",
                    extra=f' transform="rotate(-90 {px + 34} {(top + bottom) / 2})"'))
    points = []
    for j, (acc, err) in enumerate(ACCURACY_BY_K):
        x = left + 18 + j * (right - left - 36) / (len(ACCURACY_BY_K) - 1)
        y = bottom - 12 - acc * (bottom - top - 24)
        points.append((x, y))
        out.append(line(x, y - err, x, y + err, stroke=INK, width=1))
        out.append(line(x - 4, y - err, x + 4, y - err, stroke=INK, width=1))
        out.append(line(x - 4, y + err, x + 4, y + err, stroke=INK, width=1))
    out.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in points) +
               f'" fill="none" stroke="{blue(0)}" stroke-width="2"/>')
    out += [f'<circle cx="{x}" cy="{y}" r="4" fill="{blue(0)}" stroke="#FFFFFF" stroke-width="1"/>'
            for x, y in points]
    for (x, _), label in ((points[0], "<tspan font-style=\"italic\">m</tspan>/2"),
                          (points[-1], "<tspan font-style=\"italic\">m</tspan>")):
        out.append(line(x, bottom, x, bottom + 5))
        out.append(text(x, bottom + 22, label, size=15, anchor="middle"))
    out.append(text((left + right) / 2, bottom + 42, "k", italic=True, anchor="middle"))

    # Bottom: selection frequency of each feature across subsets
    left, top, bottom = px + 62, 315, 505
    out.append(line(left, top, left, bottom))
    out.append(line(left, bottom, px + 276, bottom))
    out.append(text(px + 34, (top + bottom) / 2, "Selection frequency", size=15, anchor="middle",
                    extra=f' transform="rotate(-90 {px + 34} {(top + bottom) / 2})"'))
    for tick in (0, 0.5, 1):
        y = bottom - tick * (bottom - top - 10)
        out.append(line(left - 5, y, left, y))
        out.append(text(left - 9, y + 5, f"{tick:g}", size=14, anchor="end"))
    bw, gap = 18, 7
    x0 = left + 12
    for i, f in enumerate(FREQUENCY):
        x = x0 + i * (bw + gap)
        h = f * (bottom - top - 10)
        out.append(rect(x, bottom - h, bw, h, fill=blue(i)))
        out.append(text(x + bw / 2, bottom + 20, sub("f", i + 1, size=10), size=14, italic=True, anchor="middle"))
    return out


def build():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{FONT}" role="img" aria-label="Figure 1: overview of PFSS. (a) feature scores are '
        f'normalised into selection probabilities; (b) feature subsets are sampled from them; (c) an ensemble '
        f'of six classifiers is trained on each subset; (d) accuracy versus subset size and feature selection '
        f'frequency are reported">',
        '<defs>',
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,1 L10,5 L0,9 z" fill="{INK}"/></marker>',
        '</defs>',
        rect(0, 0, W, H, fill="#FFFFFF"),
    ]

    for px, letter, title in zip(PANEL_X, "abcd", TITLES):
        parts.append(text(px, 60, f'<tspan font-weight="700">({letter})</tspan> {title}', size=21))
        parts.append(rect(px, PANEL_TOP, PANEL_W, PANEL_BOTTOM - PANEL_TOP, stroke=RULE))

    mid = (PANEL_TOP + PANEL_BOTTOM) / 2
    right = [x + PANEL_W for x in PANEL_X]
    parts += [connector(right[0], mid, PANEL_X[1], y) for y in ROW_Y]   # fan out into subsets
    parts += [connector(right[1], y, PANEL_X[2], y) for y in ROW_Y]     # subset -> ensemble
    parts += [connector(right[2], y, PANEL_X[3], mid) for y in ROW_Y]   # fan in to aggregation

    parts += scoring_panel(PANEL_X[0])
    parts += sampling_panel(PANEL_X[1])
    parts += training_panel(PANEL_X[2])
    parts += aggregation_panel(PANEL_X[3])
    parts.append('</svg>')
    return "\n".join(parts)


if __name__ == "__main__":
    out = Path(__file__).with_name("thumbnail.svg")
    out.write_text(build() + "\n", encoding="utf-8")
    print(f"Wrote {out}")
