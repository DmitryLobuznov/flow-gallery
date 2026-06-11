"""
Cahn–Hilliard phase separation — reference implementation.

Solves the dimensionless Cahn–Hilliard equation

    ∂c/∂t = M ∇²μ,      μ = f'(c) − κ ∇²c,      f'(c) = c³ − c

on a periodic square domain with a semi-implicit (IMEX) Fourier
spectral scheme. The double-well bulk free energy f(c) = ¼(c²−1)²
drives separation into two phases (c ≈ ±1); the gradient term κ|∇c|²
penalises interfaces, giving them a finite capillary width ~√κ.

The stiff fourth-order operator is treated implicitly in Fourier
space and stabilised with an Eyre-type convex-splitting constant `a`,
so the scheme stays stable at large time steps:

    ĉⁿ⁺¹ = (ĉⁿ − Δt·M·k²·  ̂[c³−(1+a)c]) / (1 + Δt·M·a·k² + Δt·M·κ·k⁴)

This is the source-of-truth physics; the WebGL demo mirrors it with a
faster explicit finite-difference scheme.

Usage
-----
    python cahn_hilliard.py --gif ../assets/spinodal.gif
    python cahn_hilliard.py --mean 0.3 --gif ../assets/droplets.gif
    python cahn_hilliard.py --n 512 --steps 4000 --cmap magma
"""

from pathlib import Path
from typing import Optional

import numpy as np
import typer


def laplacian_symbol(n: int, dx: float) -> np.ndarray:
    """k² (the −symbol of ∇²) on an n×n periodic grid via FFT frequencies."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx**2 + ky**2  # |k|²


class CahnHilliard:
    def __init__(
        self,
        n: int = 256,
        dx: float = 1.0,
        dt: float = 0.5,
        kappa: float = 1.0,
        mobility: float = 1.0,
        mean: float = 0.0,
        noise: float = 0.1,
        stabiliser: float = 2.0,
        seed: int | None = 0,
    ):
        self.n = n
        self.dt = dt
        self.kappa = kappa
        self.M = mobility

        rng = np.random.default_rng(seed)
        self.c = mean + noise * (rng.random((n, n)) - 0.5) * 2.0

        k2 = laplacian_symbol(n, dx)
        k4 = k2**2
        a = stabiliser
        # Precompute the implicit denominator (constant in time).
        self._denom = 1.0 + dt * self.M * (a * k2 + kappa * k4)
        self._explicit_k2 = dt * self.M * k2
        self._a = a

    def step(self) -> None:
        c = self.c
        # Nonlinear + shifted-linear part handled explicitly.
        g = c**3 - (1.0 + self._a) * c
        c_hat = np.fft.fft2(c)
        g_hat = np.fft.fft2(g)
        c_hat = (c_hat - self._explicit_k2 * g_hat) / self._denom
        self.c = np.real(np.fft.ifft2(c_hat))

    def run(self, steps: int) -> np.ndarray:
        for _ in range(steps):
            self.step()
        return self.c

    @property
    def total(self) -> float:
        """Total composition — conserved by Cahn–Hilliard dynamics."""
        return float(self.c.mean())


def _to_rgb(field: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map c∈[−1,1] to an 8-bit RGB frame using a Matplotlib colormap."""
    import matplotlib

    t = np.clip(0.5 * (field + 1.0), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap_name](t)
    return (rgba[..., :3] * 255).astype(np.uint8)


def render_gif(
    out_path: str,
    n: int,
    steps: int,
    every: int,
    dt: float,
    kappa: float,
    mobility: float,
    mean: float,
    cmap: str,
    fps: int,
    seed: int | None,
) -> None:
    import imageio.v2 as imageio

    sim = CahnHilliard(n=n, dt=dt, kappa=kappa, mobility=mobility, mean=mean, seed=seed)
    frames = []
    for s in range(steps + 1):
        if s % every == 0:
            frames.append(_to_rgb(sim.c, cmap))
        sim.step()
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(
        f"wrote {out_path}  ·  {len(frames)} frames  ·  "
        f"{n}×{n}  ·  conserved mean drift "
        f"{abs(sim.total - mean):.2e}"
    )


def main(
    n: int = typer.Option(256, help="grid size"),
    steps: int = typer.Option(2000, help="time steps"),
    every: int = typer.Option(20, help="capture a frame every N steps"),
    dt: float = typer.Option(0.5, help="time step"),
    kappa: float = typer.Option(1.0, help="gradient energy coefficient"),
    mobility: float = typer.Option(1.0, help="mobility M"),
    mean: float = typer.Option(0.0, help="mean composition (0=labyrinth, ±=droplets)"),
    cmap: str = typer.Option("viridis", help="matplotlib colormap"),
    fps: int = typer.Option(25, help="gif frame rate"),
    seed: int = typer.Option(0, help="RNG seed"),
    gif: Optional[Path] = typer.Option(None, help="output gif path (if omitted, just runs)"),
) -> None:
    """Cahn–Hilliard reference solver / GIF maker."""
    if gif is not None:
        render_gif(str(gif), n, steps, every, dt, kappa, mobility, mean, cmap, fps, seed)
    else:
        sim = CahnHilliard(n=n, dt=dt, kappa=kappa, mobility=mobility, mean=mean, seed=seed)
        sim.run(steps)
        print(f"ran {steps} steps · mean drift {abs(sim.total - mean):.2e}")


if __name__ == "__main__":
    typer.run(main)
