"""Build the Phase 1 final presentation as an editable .pptx (native charts and tables).

All numbers are read from reports/ieee_report/figures/figure_values.json, which
make_figures.py derives from the committed results/metrics/ artifacts, so the deck
and the manuscript share one source. Run from the repository root:

    python reports/ieee_report/make_figures.py
    python reports/presentation/build_presentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

VALUES = json.loads(Path("reports/ieee_report/figures/figure_values.json").read_text())
OUT = Path("reports/presentation/phase1_final_presentation.pptx")

# Palette: ink navy dominant on title slides; Okabe-Ito accents shared with the paper figures
INK = RGBColor(0x1B, 0x2A, 0x41)
INK_SOFT = RGBColor(0x2C, 0x3E, 0x5A)
TEXT = RGBColor(0x22, 0x2B, 0x36)
MUTED = RGBColor(0x5C, 0x67, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xF1, 0xF4, 0xF8)
GRID = "D9DEE5"
BLUE = RGBColor(0x00, 0x72, 0xB2)
ORANGE = RGBColor(0xE6, 0x9F, 0x00)
GREEN = RGBColor(0x00, 0x9E, 0x73)
VERMIL = RGBColor(0xD5, 0x5E, 0x00)
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"

ENDPOINTS = list(VALUES["rf_combined_endpoints"])
CFG = VALUES["configs"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
TITLE_ONLY = prs.slide_layouts[5]
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, runs, size=16, color=TEXT, bold=False, font=BODY_FONT, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0.05, italic=False, line_spacing=None):
    """Add a text box. `runs` is a str, or a list of paragraphs; a paragraph is a str or
    a list of (text, {overrides}) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, opt in segs:
            r = p.add_run()
            r.text = seg
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=TEXT, space=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.05))
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25))))
        pPr.set("indent", str(-Emu(Inches(0.25))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        segs = item if isinstance(item, list) else [(item, {})]
        for seg, opt in segs:
            r = p.add_run()
            r.text = seg
            r.font.name = BODY_FONT
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", False)
            r.font.color.rgb = opt.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


def badge(slide, x, y, label, fill):
    """Numbered circle: the deck's recurring motif."""
    c = box(slide, x, y, 0.5, 0.5, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD_FONT, Pt(16), True, WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def content_slide(title, notes=None, source=None):
    s = prs.slides.add_slide(TITLE_ONLY)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = 0
    tf.text = title
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    for r in p.runs:
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD_FONT, Pt(34), True, INK
    if source:
        text(s, 0.6, 6.95, 10.5, 0.35, source, size=10, color=MUTED)
    text(s, 11.9, 6.95, 0.8, 0.35, str(len(prs.slides)), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def dark_slide(notes=None):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = INK
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def style_chart(chart, size=12):
    chart.font.name = BODY_FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = TEXT
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor.from_string(GRID)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    va.tick_labels.font.color.rgb = MUTED
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.color.rgb = TEXT
    ca.format.line.color.rgb = RGBColor.from_string("9AA4AF")


def table(slide, x, y, w, rows, col_w, header_fill=INK, size=12, row_h=0.36, bold_rows=(), center_cols=None):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * len(rows)))
    tbl = shp.table
    tbl.first_row = True
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (TINT if i % 2 == 0 else WHITE)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            centered = (j > 0) if center_cols is None else (j in center_cols)
            p.alignment = PP_ALIGN.CENTER if centered else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
            r.font.bold = i == 0 or i in bold_rows
            r.font.color.rgb = WHITE if i == 0 else TEXT
    return tbl


def stat(slide, x, y, w, value, label, color=INK, size=40):
    vh = max(0.8, size * 1.3 / 72 + 0.15)
    text(slide, x, y, w, vh, value, size=size, bold=True, color=color, font=HEAD_FONT)
    text(slide, x, y + vh, w, 0.7, label, size=14, color=MUTED)


rf = VALUES["rf_combined_endpoints"]
ls = VALUES["label_stats"]
n_obs = sum(s["positive_count"] + s["negative_count"] for s in ls.values())
n_missing = sum(s["missing_samples"] for s in ls.values())
miss_pct = 100 * n_missing / (n_obs + n_missing)
prev_sorted = sorted(ls.items(), key=lambda kv: kv[1]["prevalence"])
lo_e, lo = prev_sorted[0][0], 100 * prev_sorted[0][1]["prevalence"]
hi_e, hi = prev_sorted[-1][0], 100 * prev_sorted[-1][1]["prevalence"]
best = CFG["random_forest_combined"]
xgbc = CFG["xgboost_combined"]

# ============================================================ 1. Title
s = dark_slide(notes="Introduce the team and the scope: this is Phase 1 only, classical machine learning. "
                     "Graph neural networks in the project title are Phase 2 and are not reported here.")
text(s, 0.8, 1.2, 11.7, 0.5, "PHASE 1 FINAL PRESENTATION", size=14, bold=True, color=RGBColor(0x8F, 0xD3, 0xBE))
text(s, 0.8, 1.7, 11.7, 2.35, "Molecular Toxicity Prediction with Morgan Fingerprints and Physicochemical Descriptors",
     size=40, bold=True, color=WHITE, font=HEAD_FONT)
text(s, 0.8, 4.2, 11.7, 0.6, "A scaffold-based evaluation of classical machine-learning models on Tox21",
     size=20, color=RGBColor(0xCA, 0xD5, 0xE2), italic=True)
text(s, 0.8, 5.35, 11.7, 0.5,
     "Amogha Raj Sindur  ·  Bulbul Agarwalla  ·  Shourya Pratap  ·  Kanishka Dubey  ·  Aditya Yadav",
     size=16, color=WHITE)
text(s, 0.8, 5.85, 11.7, 0.5, "[Department, Institution]   ·   Supervisor: [Name]", size=14,
     color=RGBColor(0xA9, 0xB6, 0xC6))

# ============================================================ 2. Problem
s = content_slide("Why toxicity prediction is hard",
                  notes="Animal testing is slow, costly, and translates imperfectly to humans (Hartung 2009); "
                        "the NRC 2007 report pushed toward high-throughput in vitro assays, which produced Tox21. "
                        f"Our data: {miss_pct:.2f}% of molecule-endpoint labels are missing, and prevalence runs "
                        f"from {lo:.2f}% to {hi:.2f}%. Random splits leak close analogues into the test set.",
                  source="[1] Hartung 2009 · [2] NRC 2007 · [3] Huang et al. 2016 · values: this project")
cards = [
    ("1", "Costly experiments", "In vivo testing is slow and extrapolates imperfectly to humans; in vitro HTS (Tox21) "
                                "was the response [1, 2, 3].", BLUE),
    ("2", "Sparse, imbalanced labels", f"{miss_pct:.1f}% of molecule–assay labels are missing; actives are only "
                                       f"{lo:.1f}–{hi:.1f}% per endpoint.", VERMIL),
    ("3", "Optimistic evaluation", "Structurally related molecules across train and test inflate scores; the split "
                                   "protocol matters.", GREEN),
]
for i, (n, head, body, col) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 1.9, 3.8, 3.9, fill=TINT)
    badge(s, x + 0.3, 2.2, n, col)
    text(s, x + 0.3, 2.9, 3.2, 0.9, head, size=21, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.3, 3.8, 3.2, 1.8, body, size=17, color=TEXT)

# ============================================================ 3. Objectives
s = content_slide("Research question and objectives",
                  notes="State the single question. Emphasise that test labels are untouched until the end, "
                        "and that we deliberately report what the uncertainty does and does not mean.")
box(s, 0.6, 1.55, 12.1, 1.45, fill=INK)
text(s, 0.95, 1.65, 11.4, 1.25,
     "Under a leakage-controlled scaffold split of Tox21, how do circular fingerprints, physicochemical descriptors "
     "and their combination compare across logistic regression, random forest and XGBoost?",
     size=20, color=WHITE, italic=True, anchor=MSO_ANCHOR.MIDDLE, font=HEAD_FONT)
objs = [
    ("Curate", "Checksum-verified Tox21, RDKit sanitisation, auditable exclusions"),
    ("Split", "Bemis–Murcko scaffolds with zero overlap across partitions"),
    ("Benchmark", "3 models × 3 representations × 12 endpoints"),
    ("Evaluate honestly", "Masked labels, train-only scaling, validation-only thresholds"),
]
for i, (h, b) in enumerate(objs):
    x = 0.6 + (i % 2) * 6.15
    y = 3.35 + (i // 2) * 1.7
    badge(s, x, y + 0.05, str(i + 1), [BLUE, ORANGE, GREEN, VERMIL][i])
    text(s, x + 0.7, y - 0.02, 5.2, 0.5, h, size=20, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.7, y + 0.48, 5.2, 0.9, b, size=16)

# ============================================================ 4. Related work
s = content_slide("What the literature tells us",
                  notes="Key point: published Tox21 numbers come from different test protocols "
                        "(challenge test set, or MoleculeNet random split), so we do not compare numbers directly. "
                        "Classical descriptor models remain competitive with graph networks on average (Jiang 2021).",
                  source="[3] Huang 2016 · [4] Mayr 2016 · [5] Wu 2018 · [6] Svetnik 2003 · [7] Jiang 2021 · "
                         "[8] Sheridan 2013 · [9] Saito 2015")
rows = [
    ["Study", "Finding", "Implication for us"],
    ["Tox21 Challenge [3]", "Winners ROC-AUC 0.81–0.95; AR assays hardest (~0.7)", "Different test set → context only"],
    ["DeepTox [4]", "Multi-task DNN won; RF/SVM close behind", "Strong classical baselines matter"],
    ["MoleculeNet [5]", "Scaffold split is harder; Tox21 benchmarked on random split", "Our numbers are not comparable"],
    ["Svetnik et al. [6]", "Random forest among most accurate QSAR tools", "Motivates RF"],
    ["Jiang et al. [7]", "Descriptor models ≥ GNNs on average (11 datasets)", "Classical baseline before GNNs"],
    ["Sheridan [8]", "Random splits overestimate prospective accuracy", "Use structure-aware split"],
    ["Saito & Rehmsmeier [9]", "PR is more informative than ROC under imbalance", "Report AP with ROC-AUC"],
]
table(s, 0.6, 1.55, 12.1, rows, [2.6, 5.3, 4.2], size=14, row_h=0.6, center_cols=())

# ============================================================ 5. Dataset
s = content_slide("Dataset: Tox21, curated with RDKit",
                  notes="7,831 records in the DeepChem file, SHA-256 verified. Eight organo-aluminium records fail "
                        "RDKit valence sanitisation and are logged, not repaired. Missing labels are never treated "
                        "as inactive.",
                  source="Source: DeepChem tox21.csv.gz (SHA-256 verified) · results/metrics/endpoint_label_diagnostics.json")
stat(s, 0.6, 1.6, 3.6, "7,823", "molecules kept of 7,831\n(8 excluded: Al valence errors)")
stat(s, 0.6, 3.25, 3.6, "12", "assay endpoints\n(7 nuclear receptor, 5 stress response)")
stat(s, 0.6, 4.9, 3.6, f"{miss_pct:.1f}%", "molecule–endpoint labels missing\n(masked, never imputed)", color=VERMIL)
cd = CategoryChartData()
cd.categories = ENDPOINTS
cd.add_series("Active (% of labelled)", [round(100 * ls[e]["prevalence"], 2) for e in ENDPOINTS])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.5), Inches(1.5), Inches(8.2), Inches(5.3), cd)
ch = gf.chart
style_chart(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "Positive prevalence per endpoint (%)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 60
pl.series[0].format.fill.solid()
pl.series[0].format.fill.fore_color.rgb = VERMIL
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.0", False
pl.data_labels.font.size = Pt(11)
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.category_axis.tick_labels.font.size = Pt(11)
ch.value_axis.maximum_scale = 20
ch.value_axis.has_major_gridlines = False
ch.value_axis.visible = False

# ============================================================ 6. Pipeline
s = content_slide("End-to-end methodology",
                  notes="Walk left to right. The only place the validation set is used is threshold selection; "
                        "the test set is touched once, at the end.")
steps = [
    ("Tox21 CSV", "7,831 rows\nSHA-256 check", BLUE),
    ("Curate", "RDKit sanitise\ncanonical SMILES\n7,823 kept", BLUE),
    ("Featurise", "Morgan 2048\n+ 9 descriptors\n= 2057", GREEN),
    ("Scaffold split", "6,258 / 782 / 783\n0 overlap", ORANGE),
    ("Train", "LR · RF · XGB\nper endpoint\nNaN masked", GREEN),
    ("Threshold", "max F1 on\nvalidation only", ORANGE),
    ("Test", "ROC-AUC · AP · F1\nmacro over 12", VERMIL),
]
cw, gap = 1.6, 0.15
for i, (h, b, col) in enumerate(steps):
    x = 0.6 + i * (cw + gap)
    chev = box(s, x, 2.0, cw, 1.05, fill=col, shape=MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON)
    tf = chev.text_frame
    tf.margin_left, tf.margin_right = Inches(0.3), Inches(0.2)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = h
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = BODY_FONT, Pt(13), True, WHITE
    box(s, x + 0.05, 3.3, cw - 0.1, 1.75, fill=TINT)
    text(s, x + 0.1, 3.38, cw - 0.2, 1.6, b, size=13, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.6, 5.45, 12.1, 1.15, fill=WHITE, line=RGBColor(0xC8, 0xD0, 0xDA))
text(s, 0.85, 5.5, 11.6, 1.05,
     [[("Leakage controls:  ", {"bold": True, "color": INK}),
       ("descriptor scaling fitted on training rows only  ·  thresholds chosen on validation only  ·  "
        "test labels used only for the final metrics  ·  all seeds = 42", {})]],
     size=16, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================ 7. Representations
s = content_slide("Three molecular representations",
                  notes="Morgan bits capture local substructures (ECFP4-like, radius 2). Descriptors are global "
                        "properties with known origins: Crippen logP, Ertl TPSA, Lipinski HBD/HBA, Lovering Fsp3. "
                        "Combined concatenates both; only the descriptor block is scaled.",
                  source="[10] Rogers & Hahn 2010 · [11] Wildman & Crippen 1999 · [12] Ertl 2000 · "
                         "[13] Lipinski 1997 · [14] Lovering 2009")
reps = [
    ("2,048", "Morgan fingerprint", BLUE,
     ["Radius 2 (ECFP4-like) [10]", "Binary, hashed circular environments", "Local substructures; collisions possible"]),
    ("9", "RDKit descriptors", ORANGE,
     ["MW, logP [11], TPSA [12], HBD/HBA [13]", "Rot. bonds, rings, heavy atoms, Fsp³ [14]", "Standardised on train only"]),
    ("2,057", "Combined", GREEN,
     ["Concatenation [Morgan ‖ descriptors]", "Local + global information", "Bits unscaled; descriptors scaled"]),
]
for i, (dim, name, col, pts) in enumerate(reps):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 5.0, fill=TINT)
    text(s, x + 0.3, 1.8, 3.2, 0.9, dim, size=44, bold=True, color=col, font=HEAD_FONT)
    text(s, x + 0.3, 2.7, 3.2, 0.4, "dimensions", size=12, color=MUTED)
    text(s, x + 0.3, 3.15, 3.2, 0.55, name, size=20, bold=True, color=INK, font=HEAD_FONT)
    bullets(s, x + 0.3, 3.8, 3.3, 2.6, pts, size=15)

# ============================================================ 8. Scaffold split
sz = VALUES.get("split_sizes", {"train": 6258, "val": 782, "test": 783})
s = content_slide("Scaffold split: testing on unseen chemotypes",
                  notes="Scaffolds via RDKit MurckoScaffoldSmiles. Groups are assigned largest first, so validation "
                        "and test contain only singleton scaffolds: a strict test of generalisation to rare "
                        "chemotypes, but also a different composition from training. All 1,775 acyclic molecules are "
                        "in training. Zero overlap confirmed by independent re-extraction.",
                  source="[15] Bemis & Murcko 1996 · [5] Wu et al. 2018")
parts = [("Train", sz["train"], "760 scaffolds", INK_SOFT), ("Validation", sz["val"], "782 scaffolds", ORANGE),
         ("Test", sz["test"], "783 scaffolds", VERMIL)]
total = sum(p[1] for p in parts)
x = 0.6
for name, n, sc, col in parts:
    w = 12.1 - 2 * 1.85 - 0.1 if name == "Train" else 1.85
    box(s, x, 1.6, w, 1.45, fill=col, radius=0.05)
    text(s, x + 0.2, 1.65, w - 0.3, 1.35,
         [[(name, {"bold": True})], [(f"{n:,}", {"bold": True, "size": 24})], sc],
         size=14, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    x += w + 0.05
text(s, 0.6, 3.15, 12.1, 0.4, "Seed 42 · 80 / 10 / 10 target · scaffold groups assigned largest-first", size=13,
     color=MUTED)
stat(s, 0.6, 4.0, 3.4, "0", "scaffolds shared between\nany two partitions", color=GREEN, size=54)
bullets(s, 4.4, 4.05, 8.3, 2.7, [
    [("Why: ", {"bold": True}), ("random splits put close analogues in train and test, inflating scores [5, 8]", {})],
    [("Bemis–Murcko scaffold ", {"bold": True}), ("= ring systems + linkers, side chains removed [15]", {})],
    [("Caveat: ", {"bold": True}), ("val/test hold only singleton scaffolds; 1,775 acyclic molecules all in train", {})],
], size=17, space=16)

# ============================================================ 9. Models
s = content_slide("Classical models and imbalance handling",
                  notes="Hyperparameters were fixed in advance; no tuning was done. Each endpoint gets its own "
                        "classifier trained only on molecules labelled for that endpoint. Imbalance handled with "
                        "class weights (LR, RF) or scale_pos_weight = negatives/positives in training (XGBoost).",
                  source="[16] Breiman 2001 · [17] Chen & Guestrin 2016 · [18] Pedregosa et al. 2011")
rows = [["Model", "Configuration (fixed, not tuned)", "Imbalance handling"],
        ["Logistic Regression", "L2, C = 1.0, lbfgs, max_iter 1000", "class_weight = balanced"],
        ["Random Forest [16]", "100 trees, max_depth 12, min_samples_split 5", "class_weight = balanced"],
        ["XGBoost [17]", "100 rounds, max_depth 5, learning rate 0.1", "scale_pos_weight = N⁻/N⁺ (train)"]]
table(s, 0.6, 1.65, 12.1, rows, [3.0, 5.5, 3.6], size=16, row_h=0.62, center_cols=())
for i, (h, b) in enumerate([("36 classifiers per model family", "3 representations × 12 endpoints, each trained "
                                                                  "independently"),
                            ("random_state = 42 everywhere", "scikit-learn [18] and XGBoost implementations")]):
    x = 0.6 + i * 6.15
    box(s, x, 4.65, 5.95, 1.75, fill=TINT)
    text(s, x + 0.3, 4.8, 5.4, 0.5, h, size=19, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.3, 5.35, 5.4, 0.9, b, size=16)

# ============================================================ 10. Evaluation protocol
s = content_slide("Evaluation protocol",
                  notes="AP is the step-wise average precision (sklearn average_precision_score), not a trapezoid. "
                        "Its random baseline equals prevalence. The plus-minus value is the SEM across the 12 "
                        "endpoints from one run: it describes assay-to-assay spread, not seed variance, and is not "
                        "a significance test.",
                  source="[9] Saito & Rehmsmeier 2015 · [19] Davis & Goadrich 2006")
items = [
    ("Missing labels", "Masked per endpoint in training and in every metric"),
    ("ROC-AUC", "P(active ranked above inactive); prevalence-invariant"),
    ("Average precision (AP)", "Precision averaged over recall steps; random baseline = prevalence [9, 19]"),
    ("Threshold τ*", "argmax F1 on validation, grid 0.05–0.95, frozen for test"),
]
for i, (h, b) in enumerate(items):
    y = 1.6 + i * 1.0
    badge(s, 0.6, y + 0.1, str(i + 1), [BLUE, GREEN, ORANGE, VERMIL][i])
    text(s, 1.3, y, 3.4, 0.7, h, size=18, bold=True, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 4.3, y, 8.4, 0.7, b, size=16, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.6, 5.7, 12.1, 1.05, fill=INK)
text(s, 0.9, 5.75, 11.6, 0.95,
     [[("Reading “± SEM”:  ", {"bold": True, "color": RGBColor(0x8F, 0xD3, 0xBE)}),
       ("SD/√12 across the 12 endpoints of a single seed-42 run. It is assay-to-assay spread, "
        "not run-to-run uncertainty, and not a significance test.", {})]],
     size=16, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================ 11. Main results (charts)
s = content_slide("Main results: combined features lead for tree models",
                  notes=f"RF with combined features: ROC-AUC {best['roc_mean']:.4f} ± {best['roc_sem']:.4f}, "
                        f"AP {best['ap_mean']:.4f} ± {best['ap_sem']:.4f}. XGBoost combined is close on ROC-AUC "
                        f"({xgbc['roc_mean']:.4f}). Logistic regression on descriptors has decent ROC-AUC but the "
                        "lowest AP. All values recompute exactly from the saved prediction CSVs.",
                  source="Test partition (783 molecules) · results/metrics/model_comparison_table.csv")
for k, (metric, title, lo_, hi_) in enumerate((("roc", "Test macro ROC-AUC", 0.6, 0.8),
                                              ("ap", "Test macro average precision", 0.0, 0.45))):
    cd = CategoryChartData()
    cd.categories = ["Logistic Reg.", "Random Forest", "XGBoost"]
    for fk, fl in (("morgan", "Morgan"), ("descriptors", "Descriptors"), ("combined", "Combined")):
        cd.add_series(fl, [round(CFG[f"{m}_{fk}"][f"{metric}_mean"], 4)
                           for m in ("logistic_regression", "random_forest", "xgboost")])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6 + k * 6.15), Inches(1.45), Inches(5.95),
                            Inches(5.35), cd)
    ch = gf.chart
    style_chart(ch, 12)
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(15)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = lo_, hi_
    ch.value_axis.tick_labels.number_format, ch.value_axis.tick_labels.number_format_is_linked = "0.00", False
    pl = ch.plots[0]
    pl.gap_width, pl.overlap = 70, 0
    for ser, col in zip(pl.series, (BLUE, ORANGE, GREEN)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.000", False
    pl.data_labels.font.size = Pt(10)
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END

# ============================================================ 12. Full table
s = content_slide("All nine configurations",
                  notes="Best value per column is RF + combined. Note the SEMs (about 0.02 for ROC-AUC) are as large "
                        "as the gap between RF and XGBoost on combined features.",
                  source="± = SEM across 12 endpoints, single run (seed 42) · F1 / BA at validation-selected thresholds")
rows = [["Model", "Representation", "ROC-AUC (± SEM)", "AP (± SEM)", "Macro F1", "Balanced acc."]]
order = []
for m, ml in (("logistic_regression", "LR"), ("random_forest", "RF"), ("xgboost", "XGBoost")):
    for fk, fl in (("morgan", "Morgan (2048)"), ("descriptors", "Descriptors (9)"), ("combined", "Combined (2057)")):
        c = CFG[f"{m}_{fk}"]
        rows.append([ml, fl, f"{c['roc_mean']:.4f} ± {c['roc_sem']:.4f}", f"{c['ap_mean']:.4f} ± {c['ap_sem']:.4f}",
                     f"{c['f1_mean']:.4f}", f"{c['bacc_mean']:.4f}"])
table(s, 0.6, 1.55, 12.1, rows, [1.6, 2.6, 2.4, 2.3, 1.5, 1.7], size=15, row_h=0.5, bold_rows=(6,))

# ============================================================ 13. Analysis
wins_m = sum(a > b for a, b in zip(best["endpoint_roc"], CFG["random_forest_morgan"]["endpoint_roc"]))
wins_x = sum(a > b for a, b in zip(best["endpoint_roc"], xgbc["endpoint_roc"]))
lifts = [rf[e]["ap"] / rf[e]["test_prev"] for e in ENDPOINTS]
s = content_slide("What the results show",
                  notes="Robust: combining representations helps tree models on almost every endpoint; AP beats "
                        "the random baseline everywhere. Not robust: RF versus XGBoost is a tie at endpoint level. "
                        "ROC-AUC and AP rank endpoints differently: SR-ARE has the lowest ROC-AUC but mid AP because "
                        "it has the highest test prevalence.",
                  source="Per-endpoint values: results/metrics/model_benchmark_summary.json (RF, combined features)")
stat(s, 0.6, 1.55, 3.7, f"{wins_m}/12", "endpoints where combined beats\nMorgan-only ROC-AUC (RF)", color=GREEN)
stat(s, 0.6, 3.25, 3.7, f"{wins_x}/12", "endpoints RF beats XGBoost\n(combined): effectively a tie", color=ORANGE)
stat(s, 0.6, 4.95, 3.7, f"{min(lifts):.1f}–{max(lifts):.0f}×", "AP over random baseline\n(prevalence), every endpoint",
     color=BLUE)
order = sorted(ENDPOINTS, key=lambda e: rf[e]["roc"])
cd = CategoryChartData()
cd.categories = order
cd.add_series("ROC-AUC", [round(rf[e]["roc"], 3) for e in order])
cd.add_series("AP", [round(rf[e]["ap"], 3) for e in order])
cd.add_series("Prevalence (AP baseline)", [round(rf[e]["test_prev"], 3) for e in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(4.5), Inches(1.45), Inches(8.2), Inches(5.4), cd)
ch = gf.chart
style_chart(ch, 11)
ch.has_title = True
ch.chart_title.text_frame.text = "RF + combined, test set, per endpoint"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1
pl = ch.plots[0]
pl.gap_width, pl.overlap = 40, 0
for ser, col in zip(pl.series, (BLUE, GREEN, RGBColor(0x9A, 0xA4, 0xAF))):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col

# ============================================================ 14. Reproducibility
s = content_slide("Reproducibility check",
                  notes="We regenerated everything from the raw download in the pinned environment; 122 tests pass. "
                        "XGBoost reproduces exactly. Random forest does not: the committed results predate version "
                        "pinning, so the scikit-learn version that produced them is unknown. Ranking is unchanged. "
                        "We report the committed values because every artifact refers to them.",
                  source="reports/experiment_tracking_audit.md §3")
rows = [["Configuration", "Committed ROC-AUC", "Re-run ROC-AUC", "Prediction file"],
        ["XGBoost (all 3 representations)", "0.7114 / 0.7024 / 0.7456", "identical", "byte-identical"],
        ["LR + descriptors", "0.7203", "0.7203", "byte-identical"],
        ["LR + Morgan / combined", "0.6987 / 0.7138", "0.6989 / 0.7139", "differs (≤ 0.0002)"],
        ["RF + combined", "0.7488", "0.7625", "differs"]]
table(s, 0.6, 1.6, 12.1, rows, [3.6, 3.0, 2.6, 2.9], size=15, row_h=0.6, bold_rows=(4,))
for i, (h, b, col) in enumerate([("122 / 122", "tests pass after full regeneration", GREEN),
                                 ("Exact", "metrics recomputed from saved predictions", BLUE),
                                 ("Unrecorded", "scikit-learn version behind committed RF", VERMIL)]):
    x = 0.6 + i * 4.1
    box(s, x, 4.85, 3.8, 1.75, fill=TINT)
    text(s, x + 0.3, 4.95, 3.3, 0.7, h, size=26, bold=True, color=col, font=HEAD_FONT)
    text(s, x + 0.3, 5.65, 3.3, 0.8, b, size=15)

# ============================================================ 15. Limitations
s = content_slide("Limitations and threats to validity",
                  notes="Be explicit: the best model is the best among nine under this one protocol. No significance "
                        "testing, no tuning, no external validation.")
lims = [
    ("Single seed, single split", "no run-to-run variance; no significance tests"),
    ("No hyperparameter tuning", "rankings could change with tuning"),
    ("Version sensitivity", "committed RF/LR not bit-reproducible"),
    ("Split composition", "singleton test scaffolds; some endpoints < 30 actives"),
    ("Data", "noisy, incomplete assays; 8 exclusions; no external set"),
    ("Representations", "hash collisions; no stereochemistry; endpoints modelled independently"),
]
for i, (h, b) in enumerate(lims):
    col_, row_ = i % 2, i // 2
    x, y = 0.6 + col_ * 6.15, 1.6 + row_ * 1.7
    box(s, x, y, 5.95, 1.5, fill=TINT)
    text(s, x + 0.3, y + 0.15, 5.4, 0.5, h, size=19, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.3, y + 0.7, 5.4, 0.7, b, size=16)

# ============================================================ 16. Conclusion
s = dark_slide(notes="Close with the robust findings and the Phase 2 plan. Phase 2 is future work only.")
text(s, 0.8, 0.6, 11.7, 0.9, "Conclusion and future work", size=36, bold=True, color=WHITE, font=HEAD_FONT)
concl = [
    ("Leakage-controlled benchmark", "verified data, zero-overlap split, validation-only thresholds"),
    ("Best of nine: RF + combined", f"ROC-AUC {best['roc_mean']:.4f}, AP {best['ap_mean']:.4f} (single run)"),
    ("Robust finding", "combining fingerprints and descriptors helps tree ensembles on 10–11 of 12 endpoints"),
    ("Not robust", "RF vs XGBoost ordering: within SEM and version-sensitive"),
]
for i, (h, b) in enumerate(concl):
    y = 1.75 + i * 0.95
    badge(s, 0.8, y + 0.05, str(i + 1), [GREEN, BLUE, ORANGE, VERMIL][i])
    text(s, 1.5, y, 11.0, 0.6, [[(h + ":  ", {"bold": True, "color": WHITE}),
                                  (b, {"color": RGBColor(0xCA, 0xD5, 0xE2)})]], size=17, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.8, 5.7, 11.7, 1.1, fill=INK_SOFT)
text(s, 1.1, 5.75, 11.2, 1.0,
     [[("Next (Phase 2, not done here):  ", {"bold": True, "color": RGBColor(0x8F, 0xD3, 0xBE)}),
       ("GCN, GAT, GIN on the same split indices [20–22]; repeated seeds and splits; validation-based tuning; "
        "package versions saved with every result.", {"color": WHITE})]], size=16, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================ 17. References
s = content_slide("Key references")
refs = [
    "[1] T. Hartung, Nature 460, 208–212, 2009.",
    "[2] National Research Council, Toxicity Testing in the 21st Century, 2007.",
    "[3] R. Huang et al., Front. Environ. Sci. 3:85, 2016.",
    "[4] A. Mayr et al., DeepTox, Front. Environ. Sci. 3:80, 2016.",
    "[5] Z. Wu et al., MoleculeNet, Chem. Sci. 9, 513–530, 2018.",
    "[6] V. Svetnik et al., J. Chem. Inf. Comput. Sci. 43, 1947–1958, 2003.",
    "[7] D. Jiang et al., J. Cheminform. 13:12, 2021.",
    "[8] R. P. Sheridan, J. Chem. Inf. Model. 53, 783–790, 2013.",
    "[9] T. Saito, M. Rehmsmeier, PLOS ONE 10, e0118432, 2015.",
    "[10] D. Rogers, M. Hahn, J. Chem. Inf. Model. 50, 742–754, 2010.",
    "[11] S. A. Wildman, G. M. Crippen, J. Chem. Inf. Comput. Sci. 39, 868–873, 1999.",
    "[12] P. Ertl et al., J. Med. Chem. 43, 3714–3717, 2000.",
    "[13] C. A. Lipinski et al., Adv. Drug Deliv. Rev. 23, 3–25, 1997.",
    "[14] F. Lovering et al., J. Med. Chem. 52, 6752–6756, 2009.",
    "[15] G. W. Bemis, M. A. Murcko, J. Med. Chem. 39, 2887–2893, 1996.",
    "[16] L. Breiman, Machine Learning 45, 5–32, 2001.",
    "[17] T. Chen, C. Guestrin, Proc. ACM SIGKDD, 785–794, 2016.",
    "[18] F. Pedregosa et al., JMLR 12, 2825–2830, 2011.",
    "[19] J. Davis, M. Goadrich, Proc. ICML, 233–240, 2006.",
    "[20] T. N. Kipf, M. Welling, ICLR, 2017.",
    "[21] P. Veličković et al., ICLR, 2018.",
    "[22] K. Xu et al., ICLR, 2019.",
]
text(s, 0.6, 1.45, 6.0, 5.4, refs[:11], size=14, line_spacing=1.15)
text(s, 6.75, 1.45, 6.0, 5.4, refs[11:], size=14, line_spacing=1.15)

# ============================================================ 18. Q&A
s = dark_slide(notes="Likely questions: (1) Why scaffold split rather than random? (2) What does ± SEM mean? "
                     "(3) Why does RF not reproduce exactly? (4) Why AP as well as ROC-AUC? (5) Why are 8 molecules "
                     "excluded? (6) How is test-set leakage prevented? Answers: reports/viva_preparation.md.")
text(s, 0.8, 2.6, 11.7, 1.2, "Thank you — questions?", size=44, bold=True, color=WHITE, font=HEAD_FONT,
     align=PP_ALIGN.CENTER)
text(s, 0.8, 3.9, 11.7, 0.6, "Code, data provenance and audit: Molecular-Toxicity-Prediction repository", size=16,
     color=RGBColor(0xCA, 0xD5, 0xE2), align=PP_ALIGN.CENTER)

prs.core_properties.title = "Molecular Toxicity Prediction — Phase 1"
prs.core_properties.author = "Project 8 team"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("Saved", OUT, "slides:", len(prs.slides))
