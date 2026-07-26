# TAMP-Nav Project Homepage

Anonymous research artifact homepage for NeurIPS 2026 submission.

## Quick Links

- **Live page:** `docs/index.html`
- **Style guide:** `docs/STYLE_GUIDE.md` — Read this before making changes
- **Assets:** `docs/assets/` (site.css, site.js)
- **Images:** `docs/img/` (all figures)

## Local Preview

```bash
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

## Structure

```
docs/
├── index.html              # Main page (31 KB)
├── STYLE_GUIDE.md          # Style guide for maintainers
├── README.md               # This file
├── assets/
│   ├── site.css            # All styles (38 KB)
│   └── site.js             # Interactions (2 KB)
└── img/                    # Figures (6 files, ~8 MB total)
```

## Recent Updates (2026-07-26)

Homepage refactored to address NeurIPS 2026 reviewer feedback:

1. **System architecture transparency** — Added note box in abstract explaining VLM vs. full-system modalities
2. **Sensor modality clarification** — Explicit statement in results section
3. **Component attribution** — Points to manuscript ablations
4. **Real-world evaluation details** — Expanded Figure 6 caption with hardware stack details
5. **Dataset & reward dependencies** — Added key dependencies paragraph in artifact section
6. **Video placeholder** — Visible box for deployment videos (to be added post-acceptance)

All changes documented in `STYLE_GUIDE.md`.

## Design Principles

- **Academic paper conventions** — Numbered sections/figures/tables, centered abstract, serif typography
- **Neutral palette** — White background, earth-tone accents (#b96420, #2b1d14)
- **Accessibility-first** — WCAG AA contrast, semantic HTML, keyboard navigation
- **No marketing patterns** — No purple gradients, green CTAs, or SaaS aesthetics

See `STYLE_GUIDE.md` for complete design system documentation.

## Deployment

This is a static site. To deploy:

1. Commit all changes to git
2. Push to repository (GitHub Pages, static hosting, etc.)
3. No build step required

## Notes for Future Updates

- **Adding videos:** Replace `.video-placeholder` content with actual `<video>` elements
- **Updating numbers:** Cross-check against `paper/TAMP-Nav/en.tex` manuscript
- **New figures/tables:** Follow templates in `STYLE_GUIDE.md`, increment numbers sequentially
- **Style changes:** Read the style guide first to maintain consistency

## Validation

Before pushing:

```bash
# Check HTML validity
tidy -q -e docs/index.html

# Check image optimization
du -sh docs/img/*

# Test responsive breakpoints
# Open in browser, test at 1200px, 768px, 480px widths
```

---

**Maintainer:** Anonymous (double-blind review)  
**Last updated:** 2026-07-26
