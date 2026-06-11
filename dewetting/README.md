# Thin-Film Dewetting · Rupture on the GPU

> Part of [**flow-gallery**](../) — a collection of interactive capillary-effect simulations.

A thin liquid film on a substrate is **unstable**: surface tension and
intermolecular (disjoining) forces conspire to rupture it. Holes nucleate, grow,
and merge; the liquid retracts into an array of **droplets** sitting on an
ultrathin precursor layer. This is *spinodal dewetting* — the same reason a
film of water beads up on a waxed surface.

The live demo runs entirely on the GPU (WebGL2). A
[Python reference solver](python/dewetting.py) integrates the same lubrication
equation with a semi-implicit Fourier scheme and generates the animations below.

**▶ Live demo:** https://dmitrylobuznov.github.io/flow-gallery/dewetting/

<p align="center">
  <img src="assets/rupture.gif" width="32%" alt="film rupturing into droplets" />
  <img src="assets/droplets.gif" width="32%" alt="droplet array" />
  <img src="assets/holes.gif" width="32%" alt="dewetting holes" />
</p>

## The physics

The film thickness $h(x,y)$ evolves by the **lubrication (thin-film) equation**,
a conservative gradient flow:

$$
\frac{\partial h}{\partial t} = -\nabla\cdot\Big[  h^3 \nabla\big(\gamma\nabla^2 h + \Pi(h)\big) \Big].
$$

- $h^3$ — the **lubrication mobility**. It is *degenerate* (vanishes as $h\to 0$), which is what gives dewetting its distinctive look — droplets joined by thin necks, retracting rims — quite unlike constant-mobility phase separation.
- $\gamma\nabla^2 h$ — **capillary pressure**: surface tension smooths curvature.
- $\Pi(h)$ — **disjoining pressure** from the wetting energy. We use a bounded double-well potential with minima at a precursor thickness $h_p$ and a droplet height $h_d$,

$$
W(h) = \kappa (h-h_p)^2 (h-h_d)^2, \qquad \Pi(h) = -W'(h).
$$

A flat film whose mean thickness sits on the central hump ($W''<0$) is
**spinodally unstable**. Linearising about $\bar h$ gives the growth rate

$$
s(k) = -\bar h^3 k^2\big(\gamma k^2 - \Pi'(\bar h)\big),
$$

so a band of long-wavelength modes ($\gamma k^2 < \Pi'(\bar h)$) grows, fastest
at $k^2 = \Pi'(\bar h)/2\gamma$ — that sets the characteristic spacing of the
droplets. Total volume $\int h$ is **conserved** throughout.

## How it's solved

| | Live demo (`js/`) | Reference (`python/`) |
|---|---|---|
| Method | Explicit conservative finite volume | Semi-implicit (IMEX) Fourier spectral |
| Mobility | Face-averaged $h^3$ (mass-conserving) | $h^3$, dealiased (2/3 rule) |
| Stiff term | dt auto-capped to the explicit limit | implicit biharmonic + convex-splitting stabiliser |
| Positivity | mobility clamped at $h\ge 0$ | mobility clamped at $h\ge 0$ |
| Boundaries | Periodic | Periodic (FFT) |

The fourth-order operator is stiff, so the browser demo takes many sub-steps per
frame (the explicit time step is tiny); the Python reference treats the
biharmonic implicitly and steps ~100× larger. Both conserve volume and hold the
film in $[h_p, h_d]$ through rupture.

## Interactive controls

- **Mean thickness h̄** — how much liquid; near the precursor → sparse droplets. Changing it resets the film.
- **Wetting strength κ** — higher → finer, faster dewetting (shorter droplet spacing).
- **Surface tension γ** — smooths rupture fronts; too low and they sharpen.
- **Steps / frame** — sub-steps of the (stiff, small-dt) explicit solver.
- **Look** — Copper · Gold · Ice · Viridis.
- **Paint** — drag to dig a hole (nucleate dewetting), right-drag to add liquid.
- **⟨space⟩** play/pause · **⟨r⟩** reset.

## Run it

**Live demo** — open `index.html`, or serve the folder:

```bash
python -m http.server 8000   # then open http://localhost:8000
```

**Python reference / GIF generation** — managed with [uv](https://docs.astral.sh/uv/):

```bash
cd python
uv run python dewetting.py --mean 0.8  --strength 0.35 --gamma 2.5 --cmap copper --seed 1 --gif ../assets/rupture.gif
uv run python dewetting.py --mean 0.8  --strength 0.22 --gamma 3.0 --cmap gold   --seed 2 --gif ../assets/droplets.gif
uv run python dewetting.py --mean 0.75 --strength 0.35 --gamma 2.5 --cmap ice    --seed 4 --gif ../assets/holes.gif
```

Run `uv run python dewetting.py --help` for all parameters (precursor, drop height, dt, …).

## References

- A. Vrij, *Possible mechanism for the spontaneous rupture of thin free liquid films*, Discuss. Faraday Soc. **42**, 23 (1966).
- U. Thiele, M. Mecke et al., on thin-film / disjoining-pressure dewetting models.
- R. V. Craster & O. K. Matar, *Dynamics and stability of thin liquid films*, Rev. Mod. Phys. **81**, 1131 (2009).

## License

[MIT](../LICENSE) © 2026 Dmitry Lobuznov
