# TAMP-Nav Project Homepage

Project homepage for TAMP-Nav.

## Quick Links

- **Live page:** `index.html`
- **Assets:** `assets/` (site.css, site.js)
- **Media:** `img/` (figures and deployment videos)

## Local Preview

```bash
cd docs
python3 -m http.server 8000
# Open http://localhost:8000
```

## Structure

```
docs/
├── index.html              # Main page
├── README.md               # This file
├── assets/
│   ├── site.css            # All styles (vanilla CSS, no build step)
│   └── site.js             # Lightbox + BibTeX copy + scroll-spy
└── img/
    ├── *.png               # Figures 1–6
    └── *.mp4               # Deployment videos 1–6 (failure case last)
```

## Page Conventions

- **Academic paper layout** — numbered sections, `Figure N.` / `Table N.` / `Video N.` labels, centered serif abstract, resource-link row, BibTeX block
- **Neutral palette** — white background, light gray alternating bands, earth-tone accents; the failure-case video uses the rust accent (`--rust`)
- **Videos** — `.video-grid` (3 columns → 2 → 1 responsive, max-width 900px) of small `.video-button` preview tiles (muted, no controls, aspect ratio 1280/1380, centered play badge); clicking opens the shared lightbox overlay with a `controls` player (in-page enlarge, not native fullscreen); keep success examples first and the failure case (`.failure-case`) last
- **Accessibility** — semantic HTML, ARIA labels, keyboard navigation, WCAG AA contrast
- **No marketing patterns** — no gradients, CTAs, or SaaS styling

## Maintenance Notes

- **Numbers:** cross-check every metric against `paper/TAMP-Nav/en.tex` before editing
- **New figures/tables/videos:** follow the existing markup patterns and renumber sequentially
- **Adding videos:** append `<figure class="deployment-video">` blocks with a `.video-button` tile (`data-lightbox-video` + `data-caption`) inside `.video-grid`; keep the failure case last

## Validation

Before pushing:

```bash
# Check HTML validity
tidy -q -e index.html

# Check media sizes (GitHub blocks single files over 100 MB)
du -sh img/*

# Test responsive breakpoints
# Open in browser, test at 1440px, 980px, 720px widths
```
