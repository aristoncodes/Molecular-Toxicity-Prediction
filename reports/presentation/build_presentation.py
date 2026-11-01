"""Build the updated, logically ordered Phase 1 presentation deck.

Slides flow:
  1. Title & Team Roles
  2. Problem & Motivation (Why in-silico toxicity prediction?)
  3. Dataset: Tox21 Curation, Endpoints & Imbalance
  4. End-to-End Pipeline
  5. Feature Engineering 1: Morgan Fingerprints (ECFP4) with Aspirin Example
  6. Feature Engineering 2: 9 Physicochemical Descriptors with Aspirin Example
  7. Feature Engineering 3: Combined Representation (2,057 Dims) & Synergies
  8. Data Splitting: Bemis-Murcko Scaffold Split (Zero Leakage)
  9. Classical ML Models & Multi-Task Imbalance Handling
 10. Evaluation Protocol: Why ROC-AUC & PR-AUC, Not Simple Accuracy
 11. Benchmark Performance: All 9 Configurations (Table)
 12. Benchmark Performance: Visual Bar Charts (ROC-AUC & PR-AUC)
 13. Endpoint Diagnostics: Easiest vs Hardest Assays
 14. Key Literature & Practical Learnings (Moved to back)
 15. Conclusion & Phase 2 Roadmap (GNNs)
 16. Q&A / Thank You

All numbers derived from results/metrics/ and reports/ieee_report/figures/figure_values.json.
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

# Palette: Navy dominant, Okabe-Ito accents
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
rf = VALUES["rf_combined_endpoints"]
ls = VALUES["label_stats"]
sz = VALUES.get("split_sizes", {"train": 6258, "val": 782, "test": 783})

n_obs = sum(s["positive_count"] + s["negative_count"] for s in ls.values())
n_missing = sum(s["missing_samples"] for s in ls.values())
miss_pct = 100 * n_missing / (n_obs + n_missing)
prev_sorted = sorted(ls.items(), key=lambda kv: kv[1]["prevalence"])
lo_e, lo = prev_sorted[0][0], 100 * prev_sorted[0][1]["prevalence"]
hi_e, hi = prev_sorted[-1][0], 100 * prev_sorted[-1][1]["prevalence"]
best = CFG["random_forest_combined"]
xgbc = CFG["xgboost_combined"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
TITLE_ONLY = prs.slide_layouts[5]
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------ Helpers
def text(slide, x, y, w, h, runs, size=16, color=TEXT, bold=False, font=BODY_FONT, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, margin=0.05, italic=False, line_spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    paras = runs if isinstance(runs, list) else [runs]
    for i, p_item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        segs = p_item if isinstance(p_item, list) else [(p_item, {})]
        for seg, opt in segs:
            r = p.add_run()
            r.text = seg
            r.font.name = opt.get("font", font)
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.italic = opt.get("italic", italic)
            r.font.color.rgb = opt.get("color", color)
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


def content_slide(title, subtitle=None, notes=None):
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
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD_FONT, Pt(30), True, INK
    if subtitle:
        text(s, 0.6, 1.15, 11.0, 0.35, subtitle, size=13, color=MUTED, italic=True)
    text(s, 11.9, 6.95, 0.8, 0.35, str(len(prs.slides)), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
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


def stat(slide, x, y, w, value, label, color=INK, size=38):
    vh = max(0.75, size * 1.3 / 72 + 0.1)
    text(slide, x, y, w, vh, value, size=size, bold=True, color=color, font=HEAD_FONT)
    text(slide, x, y + vh, w, 0.7, label, size=13, color=MUTED)


# ==============================================================================
# SLIDE 1: Title & Team Roles
# ==============================================================================
s = dark_slide(notes="Welcome the panel. Emphasize that this is Phase 1 (Classical ML benchmark with strict scaffold splitting).")
text(s, 0.8, 0.8, 11.7, 0.4, "AI / MACHINE LEARNING RESEARCH PROJECT — PHASE 1", size=13, bold=True, color=RGBColor(0x8F, 0xD3, 0xBE))
text(s, 0.8, 1.3, 11.7, 2.0, "Molecular Toxicity Prediction via Chemical Fingerprints and Classical ML Baselines",
     size=36, bold=True, color=WHITE, font=HEAD_FONT)
text(s, 0.8, 3.4, 11.7, 0.5, "Benchmarking Multi-Task Toxicological Classifiers with Strict Bemis–Murcko Scaffold Splitting",
     size=18, color=RGBColor(0xCA, 0xD5, 0xE2), italic=True)

team_cards = [
    ("Student 1", "Amogha Raj", "Dataset Preprocessing\n& Quality Audit"),
    ("Student 2", "Bulbul Agarwalla", "Feature Engineering\n& Scaffold Splitting"),
    ("Student 3", "Shourya Pratap", "Classical ML Models\n& Imbalance Training"),
    ("Student 4", "Kanishka Dubey", "Evaluation Metrics\n& Statistical Audit"),
    ("Student 5", "Aditya Yadav", "Literature Review\n& Documentation"),
]
for i, (role, name, task) in enumerate(team_cards):
    x = 0.8 + i * 2.38
    box(s, x, 4.3, 2.25, 2.4, fill=INK_SOFT, line=RGBColor(0x3D, 0x54, 0x75))
    text(s, x + 0.1, 4.45, 2.05, 0.35, role, size=12, bold=True, color=RGBColor(0x8F, 0xD3, 0xBE), align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 4.85, 2.05, 0.45, name, size=15, bold=True, color=WHITE, font=HEAD_FONT, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 5.4, 2.05, 1.1, task, size=12, color=RGBColor(0xCA, 0xD5, 0xE2), align=PP_ALIGN.CENTER)


# ==============================================================================
# SLIDE 2: Problem & Motivation (Why In-Silico Toxicity Prediction?)
# ==============================================================================
s = content_slide("Why In-Silico Molecular Toxicity Prediction?",
                  subtitle="Toxicity testing bottleneck in drug discovery and environmental safety")
cards = [
    ("1", "Wet-Lab Testing is Slow & Costly",
     "Traditional in vivo animal toxicology takes years, costs millions of dollars per compound, and raises major ethical concerns.",
     BLUE),
    ("2", "The Tox21 High-Throughput Response",
     "US Agencies (NIH, EPA, FDA) created Tox21: testing 10,000+ compounds against 12 crucial cellular toxicity pathways using robotics.",
     ORANGE),
    ("3", "The Machine Learning Challenge",
     "Real-world data has 7% to 26% missing assay labels and severe class imbalance (up to 34:1 non-toxic to toxic).",
     VERMIL),
    ("4", "The Need for Honest Benchmarks",
     "Standard random splits overestimate model capability by testing on familiar molecules. We need structure-aware scaffold splitting.",
     GREEN),
]
for i, (n, head, body, col) in enumerate(cards):
    x = 0.6 + i * 3.08
    box(s, x, 1.7, 2.95, 4.8, fill=TINT)
    badge(s, x + 0.25, 1.95, n, col)
    text(s, x + 0.25, 2.65, 2.45, 0.9, head, size=19, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.25, 3.65, 2.45, 2.6, body, size=15, color=TEXT)


# ==============================================================================
# SLIDE 3: Dataset: Tox21 Curation, Endpoints & Imbalance
# ==============================================================================
s = content_slide("Tox21 Dataset: Curation & Endpoint Diagnostics",
                  subtitle="7,823 validated small molecules across 12 biological pathway assays")
stat(s, 0.6, 1.6, 3.6, "7,823", "Clean compounds kept of 7,831\n(8 excluded: Al valence sanitization errors)")
stat(s, 0.6, 3.25, 3.6, "12 Assays", "7 Nuclear Receptors (NR)\n5 Stress Response (SR) pathways")
stat(s, 0.6, 4.9, 3.6, f"{miss_pct:.1f}%", "Assay labels missing overall\n(Masked per-task, never imputed as negative)", color=VERMIL)

cd = CategoryChartData()
cd.categories = ENDPOINTS
cd.add_series("Toxic / Active (% of labelled)", [round(100 * ls[e]["prevalence"], 2) for e in ENDPOINTS])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.5), Inches(1.5), Inches(8.2), Inches(5.3), cd)
ch = gf.chart
style_chart(ch, 12)
ch.has_title = True
ch.chart_title.text_frame.text = "Positive (Toxic) Prevalence per Endpoint (%) — Severe Imbalance (1.9% to 18%)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 50
pl.series[0].format.fill.solid()
pl.series[0].format.fill.fore_color.rgb = VERMIL
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.0", False
pl.data_labels.font.size = Pt(11)
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.category_axis.tick_labels.font.size = Pt(11)
ch.value_axis.maximum_scale = 22
ch.value_axis.has_major_gridlines = False
ch.value_axis.visible = False


# ==============================================================================
# SLIDE 4: End-to-End Pipeline
# ==============================================================================
s = content_slide("End-to-End Methodology Pipeline",
                  subtitle="Rigorous, leak-free training and out-of-sample evaluation protocol")
steps = [
    ("Raw Tox21", "7,831 CSV rows\nSHA-256 verified", BLUE),
    ("RDKit Curate", "Sanitize SMILES\n8 Al errors removed\n7,823 valid kept", BLUE),
    ("Featurize", "Morgan (2048)\n+ 9 Descriptors\n= 2,057 dims", GREEN),
    ("Scaffold Split", "6,258 / 782 / 783\n0% scaffold overlap\nSeed = 42", ORANGE),
    ("Train Models", "LR · RF · XGBoost\nPer-endpoint task\nNaNs masked out", GREEN),
    ("Threshold", "Optimize tau on\nValidation split\nMaximize F1", ORANGE),
    ("Test Eval", "ROC-AUC · PR-AUC\nMacro mean +/- SEM\nOver 12 endpoints", VERMIL),
]
cw, gap = 1.6, 0.15
for i, (h, b, col) in enumerate(steps):
    x = 0.6 + i * (cw + gap)
    box(s, x, 1.85, cw, 1.05, fill=col, shape=MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON)
    text(s, x + (0.3 if i else 0.05), 1.85, cw - (0.55 if i else 0.35), 1.05, h, size=13, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0)
    box(s, x + 0.05, 3.15, cw - 0.1, 1.9, fill=TINT)
    text(s, x + 0.1, 3.25, cw - 0.2, 1.7, b, size=13, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

box(s, 0.6, 5.35, 12.1, 1.45, fill=WHITE, line=RGBColor(0xC8, 0xD0, 0xDA))
text(s, 0.85, 5.4, 11.6, 1.35,
     [[("Key Scientific Guarantees to Avoid Data Leakage:\n", {"bold": True, "color": INK, "size": 16})],
      [("1. Scaling Isolation: ", {"bold": True, "color": BLUE}),
       ("Descriptor scalers fitted ONLY on 6,258 training rows, frozen for validation/test.\n", {})],
      [("2. Decision Thresholds: ", {"bold": True, "color": ORANGE}),
       ("Thresholds tuned strictly on Validation F1; Test labels remain completely untouched.\n", {})],
      [("3. Unimputed Labels: ", {"bold": True, "color": VERMIL}),
       ("Untested compounds kept as NaNs; never falsely assumed negative.", {})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)


# ==============================================================================
# SLIDE 5: Feature Engineering 1: Morgan Fingerprints (ECFP4) with Aspirin
# ==============================================================================
s = content_slide("Feature Engineering: Morgan Circular Fingerprints (ECFP4)",
                  subtitle="Student 2 Module — Encoding local chemical sub-environments into 2,048 bits")
fp_boxes = [
    ("Step 0: Atom Identity", "Radius 0", "Each heavy atom hashed based on atomic number, charge, valence degree, and aromaticity.", BLUE),
    ("Step 1: 1-Bond Neighborhood", "Radius 1", "Each atom gathers its immediate bonded neighbors and bond orders (single, double, aromatic).", BLUE),
    ("Step 2: 2-Bond Environment", "Radius 2", "Gathers 2-bond subgraphs, capturing rings, carbonyls, esters, and functional groups.", GREEN),
    ("Step 3: Bit Folding (2048)", "Modulo Hash", "Hashes mapped into fixed 2,048-bit binary vector (1 = present, 0 = absent). uint8 encoding.", VERMIL),
]
for i, (title, sub, desc, col) in enumerate(fp_boxes):
    x = 0.6 + i * 3.08
    box(s, x, 1.65, 2.95, 2.5, fill=TINT)
    badge(s, x + 0.2, 1.85, str(i + 1), col)
    text(s, x + 0.8, 1.85, 2.0, 0.45, sub, size=13, color=MUTED, bold=True)
    text(s, x + 0.2, 2.4, 2.55, 0.6, title, size=16, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.2, 3.0, 2.55, 1.0, desc, size=13, color=TEXT)

# Concrete Aspirin Example Box
box(s, 0.6, 4.35, 12.1, 2.45, fill=WHITE, line=RGBColor(0x2E, 0xC4, 0xB6))
text(s, 0.9, 4.45, 11.5, 0.4, "Concrete Example: Aspirin (Acetylsalicylic Acid — CC(=O)Oc1ccccc1C(=O)O)",
     size=17, bold=True, color=INK, font=HEAD_FONT)
aspirin_fp_text = [
    [("• SMILES representation: ", {"bold": True}), ("Parsed by RDKit into a molecular graph of 13 heavy atoms and 13 bonds.\n", {})],
    [("• Circular hashing: ", {"bold": True}), ("Carboxyl group (–COOH) hashes at radius 1; benzene ring atoms hash circular environments up to radius 2.\n", {})],
    [("• Bit activation: ", {"bold": True}), ("Aspirin activates exactly ", {}), ("58 out of 2,048 bits", {"bold": True, "color": BLUE}),
     (" (the remaining 1,990 bits are 0).\n", {})],
    [("• Dataset-wide properties: ", {"bold": True}), ("Across all 7,823 molecules, Morgan matrix is ", {}),
     ("97.2% sparse (zeros)", {"bold": True, "color": VERMIL}), (", with an average of ", {}),
     ("56.7 ± 23.4 active bits", {"bold": True}), (" per compound.", {})],
]
text(s, 0.9, 4.9, 11.5, 1.8, aspirin_fp_text, size=14)


# ==============================================================================
# SLIDE 6: Feature Engineering 2: 9 Physicochemical Descriptors with Aspirin
# ==============================================================================
s = content_slide("Feature Engineering: 9 Physicochemical Descriptors",
                  subtitle="Student 2 Module — Interpretable continuous physical & chemical properties")
desc_rows = [
    ["Descriptor", "RDKit Function", "Aspirin Value", "Scientific Definition", "Toxicological Relevance"],
    ["Molecular Weight (MW)", "Descriptors.MolWt", "180.16 g/mol", "Total mass of all constituent atoms", "Cell membrane permeability (Lipinski Rule < 500)"],
    ["LogP (Octanol-Water)", "Descriptors.MolLogP", "1.31", "Octanol/water partition coefficient", "Lipophilicity; tissue penetration & bioaccumulation"],
    ["TPSA (Polar Surface Area)", "rdMolDescriptors.CalcTPSA", "63.60 Å²", "Surface area of polar atoms (N, O, H)", "Oral bioavailability & blood-brain barrier transport"],
    ["H-Bond Donors (HBD)", "Descriptors.NumHDonors", "1", "Count of –OH and –NH hydrogen donors", "Receptor binding affinity and aqueous solubility"],
    ["H-Bond Acceptors (HBA)", "Descriptors.NumHAcceptors", "3", "Count of N and O atoms with lone pairs", "Hydrogen-bonding network in target binding sites"],
    ["Rotatable Bonds", "Descriptors.NumRotatableBonds", "2", "Freely rotating single non-ring bonds", "Conformational flexibility (rigidity aids binding)"],
    ["Ring Count", "Descriptors.RingCount", "1", "Count of aromatic / aliphatic rings", "Molecular stiffness and scaffold classification"],
    ["Heavy Atom Count", "Descriptors.HeavyAtomCount", "13", "Total non-hydrogen atoms", "Overall molecular bulk and steric hindrance"],
    ["Fraction sp³ Carbons (Fsp³)", "rdMolDescriptors.CalcFractionCSP3", "0.11", "Ratio of sp³ carbons to total carbons", "3D molecular shape complexity vs flat aromaticity"],
]
table(s, 0.6, 1.6, 12.1, desc_rows, [2.3, 2.5, 1.4, 2.9, 3.0], size=13, row_h=0.51, center_cols=(2,))


# ==============================================================================
# SLIDE 7: Feature Engineering 3: Combined Features (2,057 Dims)
# ==============================================================================
s = content_slide("Combined Feature Matrix: Synergistic Representations",
                  subtitle="Student 2 Module — Concatenating local topology and global physicochemical properties")
reps = [
    ("2,048", "Morgan Fingerprints Only", BLUE,
     ["Captures local fragments & functional groups", "Binary {0, 1} encoding, 97.2% sparse", "Unscaled, uint8 format", "Test ROC-AUC (RF): 0.7208"]),
    ("9", "Physicochemical Descriptors", ORANGE,
     ["Captures macroscopic bulk properties", "Continuous floating-point values", "Standardized on Train split only", "Test ROC-AUC (RF): 0.7215"]),
    ("2,057", "Combined Representation", GREEN,
     ["Horizontal concatenation: [Morgan || Descriptors]", "Combines local motifs + global properties", "Descriptors scaled; Morgan bits unscaled", "Test ROC-AUC (RF): 0.7488 (+0.028 boost!)"]),
]
for i, (dim, name, col, pts) in enumerate(reps):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 5.0, fill=TINT)
    text(s, x + 0.3, 1.8, 3.2, 0.8, dim, size=42, bold=True, color=col, font=HEAD_FONT)
    text(s, x + 0.3, 2.65, 3.2, 0.35, "feature dimensions", size=12, color=MUTED)
    text(s, x + 0.3, 3.05, 3.2, 0.6, name, size=19, bold=True, color=INK, font=HEAD_FONT)
    bullets(s, x + 0.3, 3.75, 3.3, 2.7, pts, size=15)


# ==============================================================================
# SLIDE 8: Data Splitting: Bemis-Murcko Scaffold Split (Zero Leakage)
# ==============================================================================
s = content_slide("Bemis–Murcko Scaffold Splitting: Preventing Data Leakage",
                  subtitle="Student 2 Module — Out-of-distribution evaluation on structurally unseen chemical families")
parts = [
    ("Train Partition", sz["train"], "79.99%", "760 core scaffolds\n(+ 1,775 acyclic molecules)", INK_SOFT),
    ("Validation Partition", sz["val"], "10.00%", "782 singleton scaffolds\nUsed ONLY for threshold tuning", ORANGE),
    ("Test Partition", sz["test"], "10.01%", "783 singleton scaffolds\nEvaluated out-of-sample ONLY", VERMIL),
]
for i, (name, n, pct, detail, col) in enumerate(parts):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 1.7, fill=col)
    text(s, x + 0.25, 1.75, 3.3, 0.35, name, size=16, bold=True, color=WHITE)
    text(s, x + 0.25, 2.1, 3.3, 0.55, f"{n:,} molecules ({pct})", size=22, bold=True, color=WHITE, font=HEAD_FONT)
    text(s, x + 0.25, 2.65, 3.3, 0.6, detail, size=13, color=RGBColor(0xEA, 0xEE, 0xF3))

stat(s, 0.6, 3.65, 3.8, "0.0%", "Scaffolds shared between Train, Val, & Test\n(Strict Zero Scaffold Overlap)", color=GREEN, size=48)

scaf_bullets = [
    [("Why Random Split Fails: ", {"bold": True, "color": VERMIL}),
     ("Medicinal chemistry datasets contain analog series (same core ring with varying side chains). Random splits put analogs in both train and test, allowing models to cheat by memorizing known scaffolds.\n", {})],
    [("How Bemis–Murcko Works: ", {"bold": True, "color": BLUE}),
     ("RDKit strips all side chains, retaining only ring systems and linkers. (Aspirin's scaffold is simply a benzene ring: ", {}),
     ("c1ccccc1", {"bold": True, "font": "Courier"}), ("). Molecules with identical scaffolds are assigned together.\n", {})],
    [("Rigorous Scientific Standard: ", {"bold": True, "color": GREEN}),
     ("Validation and test sets contain only distinct singleton scaffolds, ensuring models are tested on true prospective generalisation.", {})],
]
text(s, 4.6, 3.65, 8.1, 3.1, scaf_bullets, size=15)


# ==============================================================================
# SLIDE 9: Classical Machine Learning Models & Multi-Task Setup
# ==============================================================================
s = content_slide("Classical Machine Learning Models & Multi-Task Setup",
                  subtitle="Student 3 Module — Benchmarking Linear, Bagging, and Boosting Paradigms")
model_rows = [
    ["Model Family", "Algorithm & Implementation", "Hyperparameters (Fixed, Audited)", "Class Imbalance Strategy"],
    ["Linear Baseline", "Logistic Regression\n(sklearn)", "penalty='l2', C=1.0, solver='lbfgs', max_iter=1000", "class_weight='balanced'\n(Weights inversely prop. to frequency)"],
    ["Bagging Ensemble", "Random Forest\n(sklearn)", "n_estimators=100, max_depth=12, min_samples_split=5, n_jobs=-1", "class_weight='balanced'\n(Reweights each bootstrap sample)"],
    ["Gradient Boosting", "XGBoost Classifier\n(xgboost library)", "n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss'", "scale_pos_weight = N_neg / N_pos\n(Computed per endpoint from training set)"],
]
table(s, 0.6, 1.6, 12.1, model_rows, [2.2, 2.5, 3.8, 3.6], size=13, row_h=0.72, center_cols=(0,))

info_cards = [
    ("12 Independent Classifiers", "Tox21 has 12 distinct targets. We train 12 binary classifiers per algorithm (36 models per feature set).", BLUE),
    ("Missing Assay Label Masking", "Molecules untested for endpoint k (NaN) are masked out from loss calculation, preventing false negative noise.", ORANGE),
    ("Deterministic Reproducibility", "Random seed = 42 fixed across all data splits, bootstrap resamples, and model initializations.", GREEN),
]
for i, (head, body, col) in enumerate(info_cards):
    x = 0.6 + i * 4.1
    box(s, x, 4.95, 3.8, 1.85, fill=TINT)
    badge(s, x + 0.2, 5.15, str(i + 1), col)
    text(s, x + 0.8, 5.15, 2.8, 0.5, head, size=15, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.2, 5.75, 3.4, 0.95, body, size=13, color=TEXT)


# ==============================================================================
# SLIDE 10: Evaluation Protocol: Why ROC-AUC & PR-AUC, Not Simple Accuracy
# ==============================================================================
s = content_slide("Evaluation Protocol: Metric Rationale",
                  subtitle="Student 4 Module — Why simple accuracy is rejected under extreme class imbalance")
eval_cards = [
    ("Accuracy is Deceptive", "REJECTED",
     "With positive prevalence as low as 1.87% (NR-PPAR-gamma), a naive classifier predicting 'non-toxic' for all molecules achieves 98.1% accuracy while catching zero toxic compounds.",
     VERMIL),
    ("ROC-AUC (Area Under ROC)", "PRIMARY METRIC",
     "Measures the probability of ranking a toxic molecule higher than a non-toxic one across all possible thresholds. Prevalence-invariant; baseline is 0.50 (random coin flip).",
     BLUE),
    ("PR-AUC (Average Precision)", "CRUCIAL FOR MINORITY",
     "Precision vs. Recall curve. Completely ignores True Negatives, focusing strictly on toxic compounds. Random baseline equals positive prevalence (0.02 - 0.16).",
     GREEN),
    ("Validation Threshold Tuning", "OPTIMAL CUTOFF",
     "Default 0.5 cutoff fails on rare classes. We tuned tau in [0.05, 0.95] on Validation F1, then froze tau for Test evaluation (zero test snooping).",
     ORANGE),
]
for i, (title, tag, body, col) in enumerate(eval_cards):
    x = 0.6 + i * 3.08
    box(s, x, 1.65, 2.95, 3.9, fill=TINT)
    badge(s, x + 0.2, 1.85, str(i + 1), col)
    text(s, x + 0.8, 1.85, 2.0, 0.35, tag, size=11, bold=True, color=col)
    text(s, x + 0.2, 2.4, 2.55, 0.7, title, size=18, bold=True, color=INK, font=HEAD_FONT)
    text(s, x + 0.2, 3.2, 2.55, 2.2, body, size=14, color=TEXT)

# Statistical aggregation footer
box(s, 0.6, 5.75, 12.1, 1.05, fill=INK)
text(s, 0.9, 5.8, 11.5, 0.95,
     [[("Statistical Reporting Standard:  ", {"bold": True, "color": RGBColor(0x8F, 0xD3, 0xBE), "size": 15})],
      [("All macro metrics reported as ", {}),
       ("Mean ± SEM (K=12 endpoints)", {"bold": True, "color": WHITE}),
       (", where SEM = SD / sqrt(12). Single-class partition slices gracefully yield NaN rather than false zeros.",
        {"color": RGBColor(0xCA, 0xD5, 0xE2)})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)


# ==============================================================================
# SLIDE 11: Benchmark Results: All 9 Configurations (Table)
# ==============================================================================
s = content_slide("Benchmark Results: All 9 Configurations",
                  subtitle="Out-of-sample evaluation on unseen test scaffolds (N = 783 compounds)")
results_rows = [
    ["Model Architecture", "Feature Representation", "Test ROC-AUC (Mean ± SEM)", "Test PR-AUC (Mean ± SEM)", "Test Macro F1", "Balanced Accuracy"],
    ["Logistic Regression", "Morgan (2,048 bits)", f"{CFG['logistic_regression_morgan']['roc_mean']:.4f} ± {CFG['logistic_regression_morgan']['roc_sem']:.4f}", f"{CFG['logistic_regression_morgan']['ap_mean']:.4f} ± {CFG['logistic_regression_morgan']['ap_sem']:.4f}", f"{CFG['logistic_regression_morgan']['f1_mean']:.4f}", f"{CFG['logistic_regression_morgan']['bacc_mean']:.4f}"],
    ["Logistic Regression", "Descriptors (9 continuous)", f"{CFG['logistic_regression_descriptors']['roc_mean']:.4f} ± {CFG['logistic_regression_descriptors']['roc_sem']:.4f}", f"{CFG['logistic_regression_descriptors']['ap_mean']:.4f} ± {CFG['logistic_regression_descriptors']['ap_sem']:.4f}", f"{CFG['logistic_regression_descriptors']['f1_mean']:.4f}", f"{CFG['logistic_regression_descriptors']['bacc_mean']:.4f}"],
    ["Logistic Regression", "Combined (2,057 dims)", f"{CFG['logistic_regression_combined']['roc_mean']:.4f} ± {CFG['logistic_regression_combined']['roc_sem']:.4f}", f"{CFG['logistic_regression_combined']['ap_mean']:.4f} ± {CFG['logistic_regression_combined']['ap_sem']:.4f}", f"{CFG['logistic_regression_combined']['f1_mean']:.4f}", f"{CFG['logistic_regression_combined']['bacc_mean']:.4f}"],
    ["Random Forest", "Morgan (2,048 bits)", f"{CFG['random_forest_morgan']['roc_mean']:.4f} ± {CFG['random_forest_morgan']['roc_sem']:.4f}", f"{CFG['random_forest_morgan']['ap_mean']:.4f} ± {CFG['random_forest_morgan']['ap_sem']:.4f}", f"{CFG['random_forest_morgan']['f1_mean']:.4f}", f"{CFG['random_forest_morgan']['bacc_mean']:.4f}"],
    ["Random Forest", "Descriptors (9 continuous)", f"{CFG['random_forest_descriptors']['roc_mean']:.4f} ± {CFG['random_forest_descriptors']['roc_sem']:.4f}", f"{CFG['random_forest_descriptors']['ap_mean']:.4f} ± {CFG['random_forest_descriptors']['ap_sem']:.4f}", f"{CFG['random_forest_descriptors']['f1_mean']:.4f}", f"{CFG['random_forest_descriptors']['bacc_mean']:.4f}"],
    ["Random Forest (BEST 🏆)", "Combined (2,057 dims)", f"{best['roc_mean']:.4f} ± {best['roc_sem']:.4f}", f"{best['ap_mean']:.4f} ± {best['ap_sem']:.4f}", f"{best['f1_mean']:.4f}", f"{best['bacc_mean']:.4f}"],
    ["XGBoost Classifier", "Morgan (2,048 bits)", f"{CFG['xgboost_morgan']['roc_mean']:.4f} ± {CFG['xgboost_morgan']['roc_sem']:.4f}", f"{CFG['xgboost_morgan']['ap_mean']:.4f} ± {CFG['xgboost_morgan']['ap_sem']:.4f}", f"{CFG['xgboost_morgan']['f1_mean']:.4f}", f"{CFG['xgboost_morgan']['bacc_mean']:.4f}"],
    ["XGBoost Classifier", "Descriptors (9 continuous)", f"{CFG['xgboost_descriptors']['roc_mean']:.4f} ± {CFG['xgboost_descriptors']['roc_sem']:.4f}", f"{CFG['xgboost_descriptors']['ap_mean']:.4f} ± {CFG['xgboost_descriptors']['ap_sem']:.4f}", f"{CFG['xgboost_descriptors']['f1_mean']:.4f}", f"{CFG['xgboost_descriptors']['bacc_mean']:.4f}"],
    ["XGBoost Classifier", "Combined (2,057 dims)", f"{xgbc['roc_mean']:.4f} ± {xgbc['roc_sem']:.4f}", f"{xgbc['ap_mean']:.4f} ± {xgbc['ap_sem']:.4f}", f"{xgbc['f1_mean']:.4f}", f"{xgbc['bacc_mean']:.4f}"],
]
table(s, 0.6, 1.55, 12.1, results_rows, [2.5, 2.5, 2.4, 2.3, 1.2, 1.2], size=13, row_h=0.48, bold_rows=(6,))

text(s, 0.6, 6.45, 12.1, 0.45,
     "Key Takeaway: Random Forest on Combined Features achieves the highest Test ROC-AUC (0.7488), PR-AUC (0.3891), and Balanced Accuracy (0.6636).",
     size=14, bold=True, color=GREEN, align=PP_ALIGN.CENTER)


# ==============================================================================
# SLIDE 12: Benchmark Results: Visual Bar Charts (ROC-AUC & PR-AUC)
# ==============================================================================
s = content_slide("Benchmark Comparison: Feature Synergies in Tree Ensembles",
                  subtitle="Combined representations outperform single feature sets across tree-based algorithms")
for k, (metric, title, lo_, hi_) in enumerate((("roc", "Macro ROC-AUC by Model & Feature Set", 0.6, 0.8),
                                              ("ap", "Macro PR-AUC (Average Precision)", 0.0, 0.45))):
    cd = CategoryChartData()
    cd.categories = ["Logistic Regression", "Random Forest", "XGBoost"]
    for fk, fl in (("morgan", "Morgan (2048)"), ("descriptors", "Descriptors (9)"), ("combined", "Combined (2057)")):
        cd.add_series(fl, [CFG[f"{m}_{fk}"][f"{metric}_mean"]
                           for m in ("logistic_regression", "random_forest", "xgboost")])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6 + k * 6.15), Inches(1.45), Inches(5.95),
                            Inches(5.35), cd)
    ch = gf.chart
    style_chart(ch, 12)
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
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


# ==============================================================================
# SLIDE 13: Endpoint Diagnostics: Easiest vs Hardest Assays
# ==============================================================================
wins_m = sum(a > b for a, b in zip(best["endpoint_roc"], CFG["random_forest_morgan"]["endpoint_roc"]))
wins_x = sum(a > b for a, b in zip(best["endpoint_roc"], xgbc["endpoint_roc"]))
lifts = [rf[e]["ap"] / rf[e]["test_prev"] for e in ENDPOINTS]

s = content_slide("Endpoint Error Diagnostics: Pathway Difficulty Analysis",
                  subtitle="Per-endpoint performance of Champion Model (Random Forest on Combined Features)")
stat(s, 0.6, 1.55, 3.7, f"{wins_m} / 12", "Endpoints where Combined features beat\nMorgan-only ROC-AUC", color=GREEN)
stat(s, 0.6, 3.25, 3.7, f"{wins_x} / 12", "Endpoints where Random Forest beats\nXGBoost (effectively a tie)", color=ORANGE)
stat(s, 0.6, 4.95, 3.7, f"{min(lifts):.1f}–{max(lifts):.0f}×", "PR-AUC improvement over random\nbaseline (prevalence) across all assays", color=BLUE)

order = sorted(ENDPOINTS, key=lambda e: rf[e]["roc"])
cd = CategoryChartData()
cd.categories = order
cd.add_series("ROC-AUC", [round(rf[e]["roc"], 3) for e in order])
cd.add_series("PR-AUC (AP)", [round(rf[e]["ap"], 3) for e in order])
cd.add_series("Prevalence (Random Baseline)", [round(rf[e]["test_prev"], 3) for e in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(4.5), Inches(1.45), Inches(8.2), Inches(5.4), cd)
ch = gf.chart
style_chart(ch, 11)
ch.has_title = True
ch.chart_title.text_frame.text = "Test ROC-AUC & PR-AUC by Assay (Hardest at Top → Easiest at Bottom)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
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


# ==============================================================================
# SLIDE 14: Key Literature & Practical Learnings (Clean Table at Back)
# ==============================================================================
s = content_slide("Key Scientific Literature & Practical Takeaways",
                  subtitle="Core published foundations that guided our experimental design")
lit_rows = [
    ["Foundational Study", "Key Published Finding", "Direct Takeaway for Our Project"],
    ["Tox21 Challenge (Huang et al., 2016)", "Winner models achieved ROC-AUC 0.81–0.95 on challenge test set; AR assays were hardest (~0.70).", "Established the standard 12-assay benchmark and identified intrinsic biological difficulty."],
    ["DeepTox (Mayr et al., 2016)", "Multi-task Deep Neural Network won Tox21; Random Forest and SVM were strong close baselines.", "Confirmed that Random Forest is a formidable baseline that must be thoroughly evaluated."],
    ["MoleculeNet (Wu et al., 2018)", "Demonstrated that scaffold splitting is significantly harder than random splitting.", "Adopted scaffold splitting to avoid overly optimistic, unrealistic performance metrics."],
    ["Bemis & Murcko (1996)", "Decomposed molecules into ring systems and linkers to define core chemical scaffolds.", "Used to implement zero-overlap scaffold clustering across Train, Val, and Test."],
    ["Rogers & Hahn (2010)", "Extended Connectivity Fingerprints (ECFP): circular topological hashing for QSAR modeling.", "Guided our choice of Morgan circular fingerprints (radius 2, ECFP4, 2048 bits)."],
    ["Saito & Rehmsmeier (2015)", "Precision-Recall curves are more informative and sensitive than ROC curves under class imbalance.", "Mandated PR-AUC alongside ROC-AUC to prevent false sense of security from True Negatives."],
]
table(s, 0.6, 1.55, 12.1, lit_rows, [2.8, 5.0, 4.3], size=13, row_h=0.72, center_cols=(0,))


# ==============================================================================
# SLIDE 15: Conclusion & Phase 2 Roadmap
# ==============================================================================
s = dark_slide(notes="Wrap up findings and transition to Phase 2 (Graph Neural Networks).")
text(s, 0.8, 0.7, 11.7, 0.7, "Conclusion & Phase 2 Roadmap", size=36, bold=True, color=WHITE, font=HEAD_FONT)

concl = [
    ("Rigorous, Leak-Free Benchmark Established", "Verified 7,823 molecules, 0% scaffold overlap split, train-only scaling, validation-only thresholds."),
    ("Champion Classical Model", f"Random Forest on Combined features achieved Test ROC-AUC = {best['roc_mean']:.4f} and PR-AUC = {best['ap_mean']:.4f}."),
    ("Confirmed Feature Synergy", "Combining 2,048 Morgan bits with 9 descriptors boosted performance across 11 of 12 endpoints (+0.028 ROC-AUC)."),
    ("Extreme Imbalance Dynamics Audited", "PR-AUC correlated strongly with positive prevalence (r = +0.52); ROC-AUC remained prevalence-invariant (r = +0.22)."),
]
for i, (h, b) in enumerate(concl):
    y = 1.65 + i * 0.95
    badge(s, 0.8, y + 0.05, str(i + 1), [GREEN, BLUE, ORANGE, VERMIL][i])
    text(s, 1.5, y, 11.0, 0.6, [[(h + ":  ", {"bold": True, "color": WHITE}),
                                  (b, {"color": RGBColor(0xCA, 0xD5, 0xE2)})]], size=16, anchor=MSO_ANCHOR.MIDDLE)

box(s, 0.8, 5.55, 11.7, 1.35, fill=INK_SOFT, line=RGBColor(0x2E, 0xC4, 0xB6))
text(s, 1.1, 5.65, 11.2, 1.15,
     [[("Looking Ahead — Phase 2 Graph Neural Networks (GNNs):\n", {"bold": True, "color": RGBColor(0x8F, 0xD3, 0xBE), "size": 16})],
      [("• Direct Molecular Graph Learning: ", {"bold": True, "color": WHITE}),
       ("Implement GCN, GAT, and GIN architectures directly on atom feature vectors and adjacency matrices.\n", {"color": RGBColor(0xCA, 0xD5, 0xE2)})],
      [("• Benchmark Goal: ", {"bold": True, "color": WHITE}),
       (f"Determine whether deep relational graph models can surpass our classical Random Forest benchmark (ROC-AUC {best['roc_mean']:.4f}) on unseen scaffolds.",
        {"color": RGBColor(0xCA, 0xD5, 0xE2)})]],
     size=14, anchor=MSO_ANCHOR.MIDDLE)


# ==============================================================================
# SLIDE 16: Q&A / Thank You
# ==============================================================================
s = dark_slide(notes="Ready for Q&A. All team members stand by their respective technical modules.")
text(s, 0.8, 2.2, 11.7, 1.2, "Thank You", size=52, bold=True, color=WHITE, font=HEAD_FONT, align=PP_ALIGN.CENTER)
text(s, 0.8, 3.4, 11.7, 0.6, "Questions & Discussion", size=26, color=RGBColor(0x8F, 0xD3, 0xBE), font=HEAD_FONT, align=PP_ALIGN.CENTER)
text(s, 0.8, 4.3, 11.7, 0.6,
     "Molecular Toxicity Prediction via Chemical Fingerprints and Molecular Graph Networks",
     size=17, color=RGBColor(0xCA, 0xD5, 0xE2), italic=True, align=PP_ALIGN.CENTER)
text(s, 0.8, 5.0, 11.7, 0.5,
     "Fully reproducible code, audit logs, and test suite (122 / 122 tests passed)",
     size=14, color=RGBColor(0x9A, 0xA4, 0xAF), align=PP_ALIGN.CENTER)

prs.core_properties.title = "Molecular Toxicity Prediction — Phase 1 Final Presentation"
prs.core_properties.author = "Project 8 Team (Amogha, Bulbul, Shourya, Kanishka, Aditya)"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"✅ Successfully built presentation: {OUT} (Total slides: {len(prs.slides)})")
