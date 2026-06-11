"""
Marangoni–Bénard convection cells — Swift–Hohenberg reference.

Heat a thin liquid layer from below. Surface tension falls with temperature, so
a warm spot pulls liquid outward along the surface while cool liquid sinks —
and above a critical Marangoni number the layer spontaneously organises into a
lattice of **convection cells** (famously hexagonal — the pattern you see in a
drying paint film or a pan of hot oil).

Right at onset this pattern formation is governed by the **Swift–Hohenberg
equation** — the canonical amplitude model for Rayleigh–Bénard / Marangoni
convection (Cross & Hohenberg, Rev. Mod. Phys. 1993). For an order parameter
u(x,y) (the convection amplitude / local surface temperature):

    ∂ₜu = r·u − (1 + ∇²)² u + g·u² − u³.

    • r              — the drive (a proxy for Marangoni number above onset).
    • (1+∇²)²        — selects a finite cell size: the band of wavenumbers near
                       |k| = 1 grows, so a pattern of wavelength 2π emerges.
    • g·u² − u³      — saturation. The cubic caps the amplitude; the quadratic g
                       breaks up/down symmetry and favours HEXAGONS — the
                       hallmark Marangoni cell (vs. rolls when g = 0).

The stiff linear operator is integrated exactly in Fourier space (semi-implicit
ETD-style step); the nonlinearity is evaluated pseudo-spectrally. The pattern is
unconditionally bounded, so the solver is robust at any drive.

Usage
-----
    uv run python marangoni.py --gif ../assets/cells.gif
    uv run python marangoni.py --g 0.0 --gif ../assets/rolls.gif
    uv run python marangoni.py --r 0.6 --cells 16 --cmap magma --gif ../assets/hex.gif
"""

from pathlib import Path
from typing import Optional

import numpy as np
import typer


class Marangoni:
    """Swift–Hohenberg model of Marangoni–Bénard convection cells."""

    def __init__(
        self,
        n: int = 256,
        cells: float = 12.0,
        r: float = 0.4,
        g: float = 1.0,
        dt: float = 0.1,
        seed: int = 0,
    ):
        self.n = n
        self.r, self.g, self.dt = r, g, dt

        # Physical domain sized so the natural wavelength 2π fits `cells` times.
        L = cells * 2.0 * np.pi
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        k2 = kx**2 + ky**2

        # Linear operator L̂ = r − (1 − k²)²  (note ∇² → −k²).
        self.Lhat = r - (1.0 - k2) ** 2
        # Semi-implicit factor: treat the stiff linear part implicitly.
        self.denom = 1.0 - dt * self.Lhat

        rng = np.random.default_rng(seed)
        self.u = 0.05 * rng.standard_normal((n, n))

    def step(self):
        u = self.u
        nonlin = self.g * u**2 - u**3
        u_hat = np.fft.fft2(u)
        n_hat = np.fft.fft2(nonlin)
        u_hat = (u_hat + self.dt * n_hat) / self.denom
        self.u = np.real(np.fft.ifft2(u_hat))

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.u

    @property
    def amplitude(self):
        return float(self.u.std())


def _shade(u, cmap_name, scale):
    import matplotlib

    t = np.clip(0.5 + 0.5 * u / (scale + 1e-9), 0.0, 1.0)
    base = {"inferno": "inferno", "magma": "magma", "thermal": "RdBu_r", "viridis": "viridis"}.get(cmap_name, "inferno")
    rgb = matplotlib.colormaps[base](t)[..., :3]
    return (rgb * 255).astype(np.uint8)


def render_gif(out_path, n, cells, r, g, dt, steps, every, cmap, fps, seed):
    import imageio.v2 as imageio

    sim = Marangoni(n=n, cells=cells, r=r, g=g, dt=dt, seed=seed)
    frames, scale = [], None
    for s in range(steps + 1):
        if s % every == 0:
            if scale is None and s > steps // 6:
                scale = 2.5 * sim.u.std() + 1e-6
            frames.append(_shade(sim.u, cmap, scale if scale else (2.5 * sim.u.std() + 1e-6)))
        sim.step()
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"wrote {out_path}  ·  {len(frames)} frames  ·  {n}×{n}  ·  r={r}  ·  final amplitude {sim.amplitude:.3f}")


def main(
    n: int = typer.Option(256, help="grid size"),
    cells: float = typer.Option(12.0, help="number of convection cells across the box"),
    r: float = typer.Option(0.4, help="drive r (Marangoni number above onset)"),
    g: float = typer.Option(1.0, help="quadratic term: hexagon strength (0 → rolls)"),
    dt: float = typer.Option(0.1, help="time step"),
    steps: int = typer.Option(2000, help="time steps"),
    every: int = typer.Option(15, help="capture a frame every N steps"),
    cmap: str = typer.Option("inferno", help="look: inferno | magma | thermal | viridis"),
    fps: int = typer.Option(28, help="gif frame rate"),
    seed: int = typer.Option(0, help="RNG seed"),
    gif: Optional[Path] = typer.Option(None, help="output gif path (if omitted, just runs)"),
) -> None:
    """Marangoni–Bénard convection-cell reference solver / GIF maker."""
    if gif is not None:
        render_gif(str(gif), n, cells, r, g, dt, steps, every, cmap, fps, seed)
    else:
        sim = Marangoni(n=n, cells=cells, r=r, g=g, dt=dt, seed=seed)
        sim.run(steps)
        print(f"ran {steps} steps · r={r} · amplitude {sim.amplitude:.3f}")


if __name__ == "__main__":
    typer.run(main)
