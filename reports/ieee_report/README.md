# IEEE Phase 1 Manuscript

| File | Purpose |
| :--- | :--- |
| `phase1_final_manuscript.tex` | Final manuscript (`IEEEtran`, conference mode) |
| `references.bib` | Verified bibliography (29 entries, all cited) |
| `make_figures.py` | Regenerates every figure and `figures/figure_values.json` from `results/metrics/` |
| `figures/` | Vector (PDF) and 300-dpi PNG figures used by the paper |

The earlier template `reports/ieee_phase1_manuscript.tex` is kept unchanged for reference. It predates the model results and should not be submitted.

## Build

The PDF has **not** been compiled yet: no TeX distribution was available when this was prepared.

**Overleaf (easiest):** upload `phase1_final_manuscript.tex`, `references.bib` and the `figures/` folder, then compile with pdfLaTeX. `IEEEtran.cls` is built into Overleaf.

**Local (TeX Live / MacTeX):**

```bash
cd reports/ieee_report
pdflatex phase1_final_manuscript
bibtex   phase1_final_manuscript
pdflatex phase1_final_manuscript
pdflatex phase1_final_manuscript
```

## Regenerating figures

```bash
python reports/ieee_report/make_figures.py            # from the repository root
python reports/presentation/build_presentation.py     # rebuilds the deck from the same values
```

## Before submission

- Fill in the affiliation, e-mail and acknowledgement placeholders.
- Confirm the target venue's page limit and template rules. Standard IEEE conference format was assumed.
- Check the spelling of author names. The README spells the first author "Sindur"; the git log has "Sandur".
