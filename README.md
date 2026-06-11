# flow-gallery

A growing collection of **interactive, GPU-accelerated simulations of capillary
and interfacial flow phenomena**. Every effect runs in real time in the browser
(WebGL2) and ships with a clean **NumPy spectral reference solver** — the same
physics, once for speed and once for clarity.

**▶ Gallery:** https://dmitrylobuznov.github.io/flow-gallery/

| Effect | Status | Demo |
|---|---|---|
| [**Cahn–Hilliard**](cahn-hilliard/) — phase separation / spinodal decomposition | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/cahn-hilliard/) |
| [**Capillary waves**](capillary-waves/) — gravity–capillary surface ripples | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/capillary-waves/) |
| Thin-film dewetting — rupture &amp; retraction | 🛠 planned | — |
| Rayleigh–Plateau — jet break-up into droplets | 🛠 planned | — |
| Marangoni flow — surface-tension-gradient convection | 🛠 planned | — |

<p align="center">
  <a href="cahn-hilliard/">
    <img src="cahn-hilliard/assets/spinodal.gif" width="32%" alt="Cahn–Hilliard spinodal decomposition" />
    <img src="cahn-hilliard/assets/droplets.gif" width="32%" alt="droplet morphology" />
    <img src="cahn-hilliard/assets/coarsening.gif" width="32%" alt="coarsening" />
  </a>
</p>

## Design

Each effect is a self-contained module under its own folder:

```
flow-gallery/
├── index.html            ← gallery landing page
├── cahn-hilliard/        ← module: WebGL demo + Python reference + GIFs
│   ├── index.html
│   ├── js/main.js        ← GPU solver (ping-pong FBOs, fragment shaders)
│   ├── python/           ← spectral reference solver + GIF generator
│   └── assets/
└── ...
```

The **browser demo** trades a little numerical accuracy for real-time speed
(explicit finite differences on the GPU); the **Python reference** is the
source of truth for the physics (semi-implicit Fourier-spectral schemes) and
generates the animations. See each module's own README for the maths.

## Run locally

```bash
python -m http.server 8000   # then open http://localhost:8000
```

For a module's reference solver, see e.g. [`cahn-hilliard/`](cahn-hilliard/).

## License

[MIT](LICENSE) © 2026 Dmitry Lobuznov
