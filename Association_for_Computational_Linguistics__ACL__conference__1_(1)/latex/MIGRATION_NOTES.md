# migration notes

## template details
- ACL template file: `ACL/latex/acl_latex.tex`
- style file: `acl.sty` (loaded via `\usepackage[review]{acl}`)
- documentclass: `\documentclass[11pt]{article}`
- bibliographystyle: `acl_natbib` — matches paper/main.tex; explicit
  `\bibliographystyle{acl_natbib}` retained (acl.sty also sets this
  automatically, but being explicit causes no harm)

## changes made during migration

### preamble
- used template `\documentclass[11pt]{article}` (not paper/main.tex version,
  which was identical in this case)
- replaced `\usepackage[hyperref]{acl}` (paper) with `\usepackage[review]{acl}`
  (template) — `review` mode is correct for anonymous workshop submission;
  change to `final` for camera-ready after acceptance
- added from template (not in paper/main.tex): `inconsolata`
- retained from paper/main.tex (not in template): `amsmath`, `booktabs`,
  `multirow`, `xcolor` — required by table and math content
- float spacing retained: ✓ (template does not set these values)

### template-specific commands
- `\aclfinalcopy`: absent in this template (new ACL style uses the `review`
  option instead; change `\usepackage[review]{acl}` to
  `\usepackage[final]{acl}` for camera-ready)
- `\aclpaperid`: absent in this template (not required)
- `\setlength\titlebox`: kept at template default (commented out in template)

### figure paths
- updated all four `\includegraphics` paths from `../figures/` to `./`
  (figures copied into submission/ alongside main.tex)
- verified: all four PDFs present in submission/
  - fig1_score_deltas.pdf ✓
  - fig2_variance_decomp.pdf ✓
  - fig3_rank_instability.pdf ✓
  - fig4_coverage_heatmap.pdf ✓

### bibliography style
- template uses: `acl_natbib` (via acl_natbib.bst, no explicit
  `\bibliographystyle` in template — set automatically by acl.sty)
- paper used: `acl_natbib` (explicit `\bibliographystyle{acl_natbib}`)
- action taken: kept explicit `\bibliographystyle{acl_natbib}` — matches
  template style, explicit call is harmless

## compile result
- pdflatex not available in build environment
- static checks run instead:

| check | result |
|-------|--------|
| figure paths (no `../figures/` prefix) | ✓ all four on lines 428, 440, 452, 463 |
| refs.bib present in submission/ | ✓ |
| acl.sty present in submission/ | ✓ |
| acl_natbib.bst present in submission/ | ✓ |
| brace depth | 0 ✓ |
| missing cite keys | none ✓ (11 keys, all resolved) |

## submission checklist
- [ ] compile locally: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- [ ] confirm page count ≤ 4 content pages in compiled PDF
- [ ] confirm all four figures render (not blank) in compiled PDF
- [ ] change `\usepackage[review]{acl}` → `\usepackage[final]{acl}` for camera-ready
- [ ] upload contents of submission/ folder to Overleaf or submission system
      (main.tex, refs.bib, acl.sty, acl_natbib.bst, fig1–fig4 PDFs)
