# flow-gallery

A collection of **interactive, GPU-accelerated simulations of capillary and
interfacial flow phenomena**. Every effect runs in real time in the browser
(WebGL2) and ships with a clean **NumPy spectral reference solver** — the same
physics, once for speed and once for clarity.

**▶ Gallery:** https://dmitrylobuznov.github.io/flow-gallery/

<p align="center">
  <img src="cahn-hilliard/assets/spinodal.gif" width="19%" alt="Cahn–Hilliard" />
  <img src="capillary-waves/assets/ripple.gif" width="19%" alt="capillary waves" />
  <img src="dewetting/assets/rupture.gif" width="19%" alt="dewetting" />
  <img src="rayleigh-plateau/assets/breakup.gif" width="19%" alt="Rayleigh–Plateau" />
  <img src="marangoni/assets/cells.gif" width="19%" alt="Marangoni cells" />
</p>

| Effect | Status | Demo |
|---|---|---|
| [**Cahn–Hilliard**](cahn-hilliard/) — phase separation / spinodal decomposition | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/cahn-hilliard/) |
| [**Capillary waves**](capillary-waves/) — gravity–capillary surface ripples | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/capillary-waves/) |
| [**Thin-film dewetting**](dewetting/) — rupture &amp; retraction into droplets | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/dewetting/) |
| [**Rayleigh–Plateau**](rayleigh-plateau/) — jet break-up into droplets | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/rayleigh-plateau/) |
| [**Marangoni convection**](marangoni/) — surface-tension-gradient convection cells | ✅ live | [open ↗](https://dmitrylobuznov.github.io/flow-gallery/marangoni/) |

## Design

Each effect is a self-contained module under its own folder:

```
flow-gallery/
├── index.html            ← gallery landing page
├── cahn-hilliard/        ← module: WebGL demo + Python reference + GIFs
│   ├── index.html
│   ├── js/main.js        ← GPU solver (fragment shaders, ping-pong FBOs)
│   ├── python/           ← reference solver + GIF generator (uv + Typer)
│   └── assets/
├── capillary-waves/      ← gravity–capillary surface waves
├── dewetting/            ← thin-film rupture into droplets
├── rayleigh-plateau/     ← jet break-up (1D slender-jet)
└── marangoni/            ← convection cells (Swift–Hohenberg)
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
