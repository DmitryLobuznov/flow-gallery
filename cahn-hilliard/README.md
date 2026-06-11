# Cahn–Hilliard · Phase Separation on the GPU

> Part of [**flow-gallery**](../) — a collection of interactive capillary-effect simulations.

Real-time, interactive simulation of the **Cahn–Hilliard equation** — the
canonical model of *spinodal decomposition* and capillary phase separation.
A binary mixture spontaneously unmixes into two phases; surface tension then
coarsens the pattern, sweeping out the familiar bicontinuous "labyrinth"
and droplet morphologies.

The live demo runs entirely on the GPU in your browser (WebGL2). A
[Python reference solver](python/cahn_hilliard.py) implements the same physics
with a semi-implicit spectral scheme and generates the animations below.

**▶ Live demo:** https://dmitrylobuznov.github.io/flow-gallery/cahn-hilliard/

<p align="center">
  <img src="assets/spinodal.gif" width="32%" alt="spinodal decomposition" />
  <img src="assets/droplets.gif" width="32%" alt="droplet morphology" />
  <img src="assets/coarsening.gif" width="32%" alt="coarsening" />
</p>

## The physics

The state is a conserved order parameter $c(\mathbf{x},t)$ — the local
composition, ranging between the two pure phases $c \approx \pm 1$. Its
free energy combines a double-well bulk term and a gradient (interfacial)
penalty:

$$
F[c] = \int \left[  f(c) + \frac{\kappa}{2} \lvert\nabla c\rvert^2  \right] d\mathbf{x},
\qquad
f(c) = \frac{1}{4}\left(c^2 - 1\right)^2 .
$$

Conserved (model-B) gradient flow of this energy gives the Cahn–Hilliard
equation:

$$
\frac{\partial c}{\partial t} = M \nabla^2 \mu,
\qquad
\mu = \frac{\delta F}{\delta c} = f'(c) - \kappa \nabla^2 c = c^3 - c - \kappa \nabla^2 c .
$$

- $\mu$ is the **chemical potential**; transport is driven by its Laplacian, so total composition $\int c d\mathbf{x}$ is **conserved**.
- $\kappa$ sets the **interfacial width** $\sim\sqrt{\kappa}$ and hence the surface tension — the capillary ingredient.
- $M$ is the **mobility**, rescaling time.
- The mean composition selects the morphology: **0 → interpenetrating labyrinth**, a nonzero offset → **isolated droplets** of the minority phase.

After the initial spinodal instability, the system enters **curvature-driven
coarsening**, with the characteristic length growing as $L(t)\sim t^{1/3}$
(the Lifshitz–Slyozov law).

## How it's solved

| | Live demo (`js/`) | Reference (`python/`) |
|---|---|---|
| Method | Explicit finite differences | Semi-implicit (IMEX) Fourier spectral |
| Laplacian | Isotropic 9-point stencil | Exact in Fourier space |
| Stability | Sub-stepped explicit Euler, dt auto-capped | Stabilized semi-implicit (Eyre-type), large $\Delta t$ |
| Boundaries | Periodic (`GL_REPEAT`) | Periodic (FFT) |
| Runs on | GPU fragment shaders, ping-pong FBOs | NumPy/CPU |

The spectral scheme treats the stiff fourth-order operator implicitly,
stabilised so it stays well-behaved at large time steps:

$$
\hat{c}^{ n+1} =
\frac{\hat{c}^{ n} - \Delta t  M  k^2  \widehat{c^3 - (1+a)c}}
     {1 + \Delta t  M  a  k^2 + \Delta t  M  \kappa  k^4}.
$$

## Interactive controls

- **Composition** — mean $c$; sweep from labyrinth (0) to droplets.
- **Interface width κ**, **Mobility M**, **Time step dt**, **Steps / frame**.
- **Colormap** — Viridis, Magma, Inferno, Ink, Gold/Teal.
- **Paint** — drag on the canvas to inject phase (right-drag for the opposite phase).
- **⟨space⟩** play/pause · **⟨r⟩** reset.

## Run it

**Live demo** — just open `index.html`, or serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000
```

**Python reference / GIF generation** — managed with [uv](https://docs.astral.sh/uv/).
`uv run` reads `pyproject.toml`, builds the environment, and runs — no manual
venv or `pip install` needed. The commands below regenerate the three GIFs above:

```bash
cd python
uv run python cahn_hilliard.py --mean 0.0  --cmap viridis --n 256 --steps 1600           --gif ../assets/spinodal.gif    # labyrinth
uv run python cahn_hilliard.py --mean 0.32 --cmap magma   --n 256 --steps 1600           --gif ../assets/droplets.gif    # droplets
uv run python cahn_hilliard.py --mean 0.0  --cmap inferno --n 256 --steps 3000 --every 30 --gif ../assets/coarsening.gif  # coarsening
```

Run `uv run python cahn_hilliard.py --help` for all parameters (κ, mobility, dt, …).

## References

- J. W. Cahn & J. E. Hilliard, *Free energy of a nonuniform system. I. Interfacial free energy*, J. Chem. Phys. **28**, 258 (1958).
- D. J. Eyre, *Unconditionally gradient stable time marching the Cahn–Hilliard equation*, MRS Proc. **529** (1998).
- I. M. Lifshitz & V. V. Slyozov, *The kinetics of precipitation from supersaturated solid solutions* (1961).

## License

[MIT](LICENSE) © 2026 Dmitry Lobuznov
