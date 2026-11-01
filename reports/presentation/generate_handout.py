"""
Generate a 2-page teacher handout explaining Morgan Fingerprints,
Molecular Descriptors, and Scaffold Splitting using Aspirin as example.
"""
import os, sys
from pathlib import Path

# ── Output paths ────────────────────────────────────────────────────────────
REPORT_DIR = Path(__file__).parent
PDF_PATH   = REPORT_DIR / "Understanding_Morgan_Fingerprints_and_Descriptors.pdf"
IMG_MOL    = REPORT_DIR / "aspirin_mol.png"
IMG_SCAF   = REPORT_DIR / "aspirin_scaffold.png"

# ── RDKit molecule images ────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(aspirin_smiles)
    Draw.MolToFile(mol, str(IMG_MOL), size=(260, 200))

    scaffold_smi = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smi_str = Chem.MolToSmiles(scaffold_smi)
    Draw.MolToFile(scaffold_smi, str(IMG_SCAF), size=(160, 140))
    print(f"Scaffold SMILES: {scaffold_smi_str}")
    print("Molecule images generated.")
except Exception as e:
    print(f"WARNING: RDKit image generation failed: {e}")
    IMG_MOL  = None
    IMG_SCAF = None

# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
pt = 1  # 1 point = 1 unit in ReportLab
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Colour palette ───────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#1B2A41")
TEAL      = colors.HexColor("#2EC4B6")
LIGHT_BG  = colors.HexColor("#F1F4F8")
MID_GREY  = colors.HexColor("#6B7280")
WHITE     = colors.white
BLACK     = colors.black
PALE_TEAL = colors.HexColor("#D6F5F3")

# ── Page-numbering canvas ────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_page_number(self, page_count):
        self.setFont("Helvetica", 8)
        self.setFillColor(MID_GREY)
        text = f"Page {self._pageNumber} of {page_count}  |  Molecular Toxicity Prediction — Student 2 Module"
        self.drawCentredString(letter[0] / 2, 18 * pt, text)


# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

title_style = S("Title2",
    fontSize=18, leading=22, textColor=NAVY,
    fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)

subtitle_style = S("Sub",
    fontSize=10, leading=13, textColor=TEAL,
    fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=6)

section_style = S("Sec",
    fontSize=11, leading=14, textColor=WHITE,
    fontName="Helvetica-Bold", alignment=TA_LEFT)

body_style = S("Body2",
    fontSize=9, leading=13, textColor=BLACK,
    fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=4)

bold_body = S("BoldBody",
    fontSize=9, leading=13, textColor=NAVY,
    fontName="Helvetica-Bold", alignment=TA_LEFT)

small_style = S("Small",
    fontSize=8, leading=11, textColor=MID_GREY,
    fontName="Helvetica-Oblique", alignment=TA_CENTER)


def section_header(text):
    """Navy pill header."""
    data = [[Paragraph(text, section_style)]]
    t = Table(data, colWidths=[7.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


def two_col_table(headers, rows, col_widths=None):
    """Generic striped table."""
    if col_widths is None:
        col_widths = [1.4*inch, 1.6*inch, 1.6*inch, 2.6*inch]

    header_cells = [Paragraph(f"<b>{h}</b>", S("TH",
        fontSize=8, leading=10, textColor=WHITE, fontName="Helvetica-Bold")) for h in headers]
    table_data = [header_cells]
    for row in rows:
        table_data.append([Paragraph(str(c), S("TD",
            fontSize=8, leading=11, textColor=BLACK, fontName="Helvetica")) for c in row])

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT_BG]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── Build story ──────────────────────────────────────────────────────────────
story = []
W = 7.2 * inch   # usable width

# ════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("Understanding Molecular Feature Engineering", title_style))
story.append(Paragraph(
    "Morgan Fingerprints · Physicochemical Descriptors · Scaffold-Based Data Splitting",
    subtitle_style))
story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=8))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Running Example: Aspirin
# ════════════════════════════════════════════════════════════════════════════
story.append(section_header("🧪  Running Example: Aspirin  (CC(=O)Oc1ccccc1C(=O)O)"))
story.append(Spacer(1, 5))

aspirin_intro = (
    "We use <b>Aspirin (Acetylsalicylic acid)</b> as a concrete example throughout this handout. "
    "Aspirin is a well-known pain reliever with MW = 180.16 g/mol. Its SMILES string encodes every "
    "atom and bond, and we convert this text into numbers that a machine-learning model can understand."
)
story.append(Paragraph(aspirin_intro, body_style))
story.append(Spacer(1, 4))

# Molecule image + caption
if IMG_MOL and IMG_MOL.exists():
    mol_img = RLImage(str(IMG_MOL), width=2.0*inch, height=1.54*inch)
    img_row = [[mol_img,
                Paragraph(
                    "<b>SMILES</b> (Simplified Molecular Input Line Entry System) is a text notation "
                    "that encodes a molecule's structure. RDKit parses this string into an internal "
                    "graph of atoms and bonds — the starting point for all feature extraction.<br/><br/>"
                    "Think of a molecule as a <b>graph</b>: atoms are <i>nodes</i>, bonds are <i>edges</i>. "
                    "Every feature method below reads this same graph.",
                    body_style)]]
    img_table = Table(img_row, colWidths=[2.2*inch, 5.0*inch])
    img_table.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(img_table)
else:
    story.append(Paragraph("SMILES: CC(=O)Oc1ccccc1C(=O)O  [molecule image not available]", body_style))

story.append(Spacer(1, 8))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Morgan Fingerprints
# ════════════════════════════════════════════════════════════════════════════
story.append(section_header("1.  Morgan Fingerprints  (Extended Connectivity Fingerprints — ECFP)"))
story.append(Spacer(1, 5))

fp_intro = (
    "A Morgan fingerprint is a <b>fixed-length binary vector</b> (e.g. 2048 bits). "
    "Each bit represents whether a particular circular chemical sub-environment exists in the molecule. "
    "We use <b>radius = 2</b> (ECFP4), meaning each atom 'looks' at neighbours up to 2 bonds away."
)
story.append(Paragraph(fp_intro, body_style))
story.append(Spacer(1, 5))

fp_rows = [
    ["Radius 0\n(atom itself)",
     "Each heavy atom gets a unique integer based on its own properties: atomic number, charge, mass, aromaticity.",
     "C in –COOH gets hash 441"],
    ["Radius 1\n(1-bond away)",
     "Each atom hashes itself together with its immediate neighbours to capture local bonding environment.",
     "–COOH carbon + bonded O, OH"],
    ["Radius 2\n(2-bonds away)",
     "Each atom hashes itself with neighbours up to 2 bonds away — captures ring membership, branching.",
     "Benzene carbon sees the whole ring"],
    ["Bit folding\n(final vector)",
     "All collected hashes are mapped (modulo 2048) into a bit array. Bit = 1 if that sub-environment exists, else 0.",
     "2048-element array, e.g. [...,0,1,0,1,...]"],
]
fp_table = two_col_table(
    ["Step", "What Happens", "Aspirin Example"],
    fp_rows,
    col_widths=[1.1*inch, 3.8*inch, 2.3*inch]
)
story.append(fp_table)
story.append(Spacer(1, 5))
story.append(Paragraph(
    "<b>Why useful?</b>  Similar molecules → similar bit patterns → similar model predictions. "
    "RDKit API: <font face='Courier' size='8'>AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)</font>. "
    "Output shape: <b>(N molecules × 2048 bits)</b>.",
    body_style))

story.append(Spacer(1, 10))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Molecular Descriptors
# ════════════════════════════════════════════════════════════════════════════
story.append(section_header("2.  Physicochemical Descriptors  (9 human-interpretable numbers)"))
story.append(Spacer(1, 5))

desc_intro = (
    "Descriptors are <b>computed physical/chemical properties</b>. Unlike fingerprints they are "
    "continuous numbers with clear scientific meaning — directly usable by toxicologists to "
    "explain a model's decision."
)
story.append(Paragraph(desc_intro, body_style))
story.append(Spacer(1, 5))

desc_rows = [
    ["Molecular Weight",    "180.16 g/mol", "Total mass of all atoms",            "Predicts membrane permeability"],
    ["LogP",                "1.31",          "Water vs. octanol partition coeff.", "Lipophilicity; cell penetration"],
    ["TPSA",                "63.60 Ų",      "Surface area of polar atoms (N,O,H)","Oral bioavailability / absorption"],
    ["H-Bond Donors (HBD)", "1",             "N–H or O–H groups that donate H",   "Solubility, protein binding"],
    ["H-Bond Acceptors (HBA)","3",           "N or O atoms that accept H",        "Solubility, protein binding"],
    ["Rotatable Bonds",     "2",             "Freely rotating non-ring bonds",    "Molecular flexibility"],
    ["Ring Count",          "1",             "Number of rings (benzene = 1)",     "Rigidity, scaffold class"],
    ["Heavy Atom Count",    "13",            "All non-hydrogen atoms",            "Molecular size"],
    ["Fsp³",                "0.11",          "Fraction of sp³ carbons",           "3-D character; drug-likeness"],
]
desc_table = two_col_table(
    ["Descriptor", "Aspirin Value", "Definition", "Toxicity Relevance"],
    desc_rows,
    col_widths=[1.35*inch, 0.85*inch, 2.2*inch, 2.8*inch]
)
story.append(desc_table)
story.append(Spacer(1, 5))
story.append(Paragraph(
    "Output shape: <b>(N molecules × 9 descriptors)</b>. "
    "RDKit functions: <font face='Courier' size='8'>Descriptors.MolWt, Descriptors.MolLogP, "
    "rdMolDescriptors.CalcTPSA, …</font>",
    body_style))

story.append(Spacer(1, 10))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Combined Representation
# ════════════════════════════════════════════════════════════════════════════
story.append(section_header("3.  Combined Feature Matrix  (2048 + 9 = 2057 features)"))
story.append(Spacer(1, 5))

combo_text = (
    "We concatenate both representations horizontally. "
    "The fingerprint bits capture <i>structural patterns</i> (presence of sub-environments), "
    "while descriptors capture <i>global physicochemical properties</i>. "
    "Together they give the ML model complementary information."
)
story.append(Paragraph(combo_text, body_style))
story.append(Spacer(1, 4))

combo_rows = [
    ["Fingerprints only",  "2 048", "Structural sub-environments; captures shape & topology"],
    ["Descriptors only",   "9",     "Human-interpretable chemistry; often preferred by toxicologists"],
    ["Combined (concat.)", "2 057", "Best of both worlds — used as the primary feature set"],
]
combo_table = two_col_table(
    ["Representation", "Feature Dim.", "When to Use"],
    combo_rows,
    col_widths=[1.6*inch, 1.0*inch, 4.6*inch]
)
story.append(combo_table)
story.append(Spacer(1, 10))

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Scaffold Splitting
# ════════════════════════════════════════════════════════════════════════════
story.append(section_header("4.  Bemis-Murcko Scaffold Splitting  (Why random splits are wrong)"))
story.append(Spacer(1, 5))

scaf_intro = (
    "In drug discovery the <b>real test</b> is whether a model works on chemically <i>new</i> molecules. "
    "A random 80/10/10 split often places very similar molecules in both train and test, inflating accuracy. "
    "Scaffold splitting groups molecules by their core ring system (<i>scaffold</i>) so train and test "
    "share <b>no scaffolds</b>."
)
story.append(Paragraph(scaf_intro, body_style))
story.append(Spacer(1, 5))

# Scaffold image + explanation side by side
scaf_content = []

if IMG_SCAF and IMG_SCAF.exists():
    scaf_img = RLImage(str(IMG_SCAF), width=1.4*inch, height=1.22*inch)
    scaf_content.append(scaf_img)
    scaf_content.append(Paragraph(
        "<i>Aspirin's Bemis-Murcko scaffold = benzene ring (c1ccccc1).<br/>"
        "All aspirin-like molecules share this scaffold and go to the <b>same</b> split.</i>",
        small_style))
else:
    scaf_content.append(Paragraph("Scaffold: benzene ring (c1ccccc1) — [image not available]", small_style))

scaf_steps = [
    ["1. Extract scaffold", "RDKit strips all side-chains, keeping only the ring system"],
    ["2. Group molecules",  "All molecules with the same scaffold form one cluster"],
    ["3. Sort clusters",    "Clusters sorted by size (largest first) for reproducibility"],
    ["4. Assign splits",    "Train 80 % / Val 10 % / Test 10 % — NO scaffold in two groups"],
    ["5. Seed = 42",        "Reproducible: same split every run; reported in paper"],
]
scaf_table = two_col_table(
    ["Step", "Action"],
    scaf_steps,
    col_widths=[1.5*inch, 5.7*inch]
)

# Put image left, table right
outer_row = [[
    Table([[c] for c in scaf_content],
          colWidths=[1.6*inch],
          style=[("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]),
    scaf_table
]]
outer = Table(outer_row, colWidths=[1.7*inch, 5.5*inch])
outer.setStyle(TableStyle([
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("LEFTPADDING",(0,0),(-1,-1),0),
    ("RIGHTPADDING",(0,0),(-1,-1),0),
    ("TOPPADDING",(0,0),(-1,-1),0),
    ("BOTTOMPADDING",(0,0),(-1,-1),0),
]))
story.append(outer)
story.append(Spacer(1, 5))
story.append(Paragraph(
    "<b>Result:</b> A rigorous, publication-standard split that tests true generalisation. "
    "Implemented with <font face='Courier' size='8'>deepchem.splits.ScaffoldSplitter</font> + "
    "custom fallback using RDKit <font face='Courier' size='8'>MurckoScaffold.GetScaffoldForMol()</font>.",
    body_style))

story.append(Spacer(1, 6))
story.append(HRFlowable(width=W, thickness=1, color=TEAL, spaceAfter=4))
story.append(Paragraph(
    "Molecular Toxicity Prediction — Phase 1 · Student 2 Module · Feature Engineering & Splitting",
    small_style))

# ── Build PDF ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=letter,
    leftMargin=36, rightMargin=36,
    topMargin=36, bottomMargin=36,
)
doc.build(story, canvasmaker=NumberedCanvas)

print(f"\n✅  Handout PDF generated successfully at:\n   {PDF_PATH.resolve()}\n")

# Cleanup temp images
for p in [IMG_MOL, IMG_SCAF]:
    if p and p.exists():
        p.unlink()
        print(f"   Cleaned up temp image: {p.name}")
