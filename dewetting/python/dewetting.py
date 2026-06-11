"""
Thin-film dewetting — lubrication reference solver.

A thin liquid film of thickness h(x,y) on a solid substrate evolves under the
lubrication (thin-film) equation

    ∂h/∂t = −∇·[ h³ ∇( γ∇²h + Π(h) ) ]

with lubrication mobility h³ and surface tension γ. The two physical
ingredients:

    • γ∇²h  — capillary pressure: surface tension smooths curvature.
    • Π(h)  — disjoining pressure from the wetting energy. We use a bounded
              double-well wetting potential with minima at a thin precursor h_p
              and a droplet height h_d,

                  W(h) = κ (h − h_p)² (h − h_d)²,     Π(h) = −W'(h),

              a standard thin-film dewetting model (Thiele, Mecke, …). A flat
              film whose mean thickness sits on the central hump (W'' < 0) is
              **spinodally unstable**: thickness ripples in a band of wavenumbers
              grow, the film ruptures, and the liquid retracts into droplets of
              height ≈ h_d sitting on an ultrathin precursor ≈ h_p — *spinodal
              dewetting*. Keeping Π bounded (no 1/hⁿ singularity) holds h in
              [h_p, h_d], so the spectral scheme stays robust through rupture.

Linear stability about a mean thickness h̄ gives the growth rate

    s(k) = −h̄³ k² ( γk² − Π'(h̄) ),

so modes with γk² < Π'(h̄) grow (requires Π'(h̄) > 0, i.e. W''(h̄) < 0),
fastest at k² = Π'(h̄)/(2γ) — the characteristic dewetting wavelength.

The stiff 4th-order operator is integrated with a linearly-stabilised
semi-implicit Fourier scheme (constant-mobility implicit biharmonic, nonlinear
remainder explicit), so it stays stable at large steps. Mass ∫h is conserved.

Usage
-----
    uv run python dewetting.py --gif ../assets/rupture.gif
    uv run python dewetting.py --mean 1.0 --cmap gold --gif ../assets/droplets.gif
"""

from pathlib import Path
from typing import Optional

import numpy as np
import typer


def k_grids(n: int, dx: float):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    return kx, ky, k2


class Dewetting:
    def __init__(
        self,
        n: int = 256,
        dx: float = 1.0,
        mean: float = 0.85,
        precursor: float = 0.5,
        drop: float = 1.8,
        strength: float = 0.35,
        gamma: float = 2.5,
        dt: float = 0.15,
        noise: float = 0.02,
        seed: int = 0,
    ):
        self.n = n
        self.dt = dt
        self.hp, self.hd, self.kappa_w, self.gamma = precursor, drop, strength, gamma
        self.kx, self.ky, self.k2 = k_grids(n, dx)
        self.k4 = self.k2**2

        rng = np.random.default_rng(seed)
        self.h = mean + noise * (rng.standard_normal((n, n)))

        # 2/3-rule dealiasing mask for the spectral nonlinear products — keeps
        # sharp rupture fronts from aliasing into instability.
        kmax = np.abs(2.0 * np.pi * np.fft.fftfreq(n, d=dx)).max()
        self.mask = (np.abs(self.kx) <= (2.0 / 3.0) * kmax) & (np.abs(self.ky) <= (2.0 / 3.0) * kmax)

        # Semi-implicit convex splitting. Treat the surface-tension biharmonic
        # M0∇⁴h (M0 = h̄³) implicitly, plus an artificial M0·c·k² damping that is
        # added to the denominator AND subtracted in the explicit term — it does
        # not change the steady state, but tames the stiff disjoining k² flux so
        # the scheme is stable at large dt. c covers max|Π'| over [h_p, h_d].
        # Use the WORST-CASE mobility (tall droplets, h≈h_d) so the implicit
        # damping covers the largest local surface-tension flux, not just the
        # mean-thickness one — otherwise growing drops outrun the stabiliser.
        self.M0 = drop**3
        hs = np.linspace(precursor, drop, 64)
        self.c_stab = float(np.abs(self._dPi(hs)).max())
        self.denom = 1.0 + dt * self.M0 * (gamma * self.k4 + self.c_stab * self.k2)

    def _Pi(self, h):
        """Disjoining pressure Π(h) = −W'(h), W = κ(h−h_p)²(h−h_d)²."""
        hp, hd, k = self.hp, self.hd, self.kappa_w
        return -2.0 * k * (h - hp) * (h - hd) * (2.0 * h - hp - hd)

    def _dPi(self, h):
        """Π'(h) = −W''(h); >0 on the unstable central hump."""
        hp, hd, k = self.hp, self.hd, self.kappa_w
        # W'' = 2κ[(h−hd)² + 4(h−hp)(h−hd) + (h−hp)²]
        wpp = 2.0 * k * ((h - hd) ** 2 + 4.0 * (h - hp) * (h - hd) + (h - hp) ** 2)
        return -wpp

    def step(self) -> None:
        h = self.h
        k2, kx, ky = self.k2, self.kx, self.ky
        h_hat = np.fft.fft2(h)

        # Pressure p = γ∇²h + Π(h)  (capillary smoothing + nonlinear disjoining).
        lap_h = np.real(np.fft.ifft2(-k2 * h_hat))
        p = self.gamma * lap_h + self._Pi(h)
        p_hat = np.fft.fft2(p)

        # Flux divergence ∇·(h³ ∇p) computed spectrally.
        dpx = np.real(np.fft.ifft2(1j * kx * p_hat))
        dpy = np.real(np.fft.ifft2(1j * ky * p_hat))
        # Mobility h³ — clamp the argument to ≥0 so spectral Gibbs ripples that
        # dip h slightly negative can't flip the mobility sign (which would turn
        # the diffusion backward and blow up). Physically h never reaches 0: the
        # precursor well halts thinning well above it.
        m = np.maximum(h, 0.0) ** 3
        flux_div_hat = 1j * kx * np.fft.fft2(m * dpx) + 1j * ky * np.fft.fft2(m * dpy)
        flux_div_hat *= self.mask  # 2/3-rule dealiasing of the nonlinear flux

        # Thin-film equation: ∂h/∂t = −∇·(h³∇p) = −flux_div. The implicit terms
        # M0(k⁴ + c·k²) are added back explicitly so only the artificial damping
        # is treated implicitly (convex splitting — steady state unchanged).
        rhs_hat = -flux_div_hat + self.M0 * (self.gamma * self.k4 + self.c_stab * self.k2) * h_hat
        h_hat_new = (h_hat + self.dt * rhs_hat) / self.denom
        self.h = np.real(np.fft.ifft2(h_hat_new))

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
        return self.h

    @property
    def mass(self) -> float:
        return float(self.h.mean())


def _shade(h: np.ndarray, cmap_name: str, hmax: float) -> np.ndarray:
    import matplotlib

    gx, gy = np.gradient(h)
    lx, ly, lz = 0.4, 0.5, 0.75
    nz = 1.0 / np.sqrt(gx**2 + gy**2 + 1.0)
    diff = np.clip((-gx * lx - gy * ly + lz) * nz, 0.0, 1.0)
    t = np.clip(h / (hmax + 1e-9), 0.0, 1.0)  # thickness → colour
    base = {"copper": "copper", "gold": "afmhot", "ice": "GnBu", "viridis": "viridis"}.get(cmap_name, "copper")
    rgb = matplotlib.colormaps[base](t)[..., :3]
    rgb = rgb * (0.55 + 0.45 * diff[..., None])  # lit
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def render_gif(out_path, n, steps, every, mean, precursor, drop, strength, gamma, dt, cmap, fps, seed):
    import imageio.v2 as imageio

    sim = Dewetting(n=n, mean=mean, precursor=precursor, drop=drop, strength=strength, gamma=gamma, dt=dt, seed=seed)
    frames, hmax = [], drop
    for s in range(steps + 1):
        if s % every == 0:
            frames.append(_shade(sim.h, cmap, hmax))
        sim.step()
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(
        f"wrote {out_path}  ·  {len(frames)} frames  ·  {n}×{n}  ·  "
        f"h∈[{sim.h.min():.3f},{sim.h.max():.3f}]  ·  mass drift {abs(sim.mass - mean):.2e}"
    )


def main(
    n: int = typer.Option(256, help="grid size"),
    steps: int = typer.Option(9000, help="time steps"),
    every: int = typer.Option(90, help="capture a frame every N steps"),
    mean: float = typer.Option(0.8, help="mean film thickness h̄ (near precursor → sparse drops)"),
    precursor: float = typer.Option(0.5, help="precursor thickness h_p (thin well)"),
    drop: float = typer.Option(1.8, help="droplet height h_d (thick well)"),
    strength: float = typer.Option(0.35, help="wetting-energy strength κ"),
    gamma: float = typer.Option(2.5, help="surface tension γ (smooths rupture fronts)"),
    dt: float = typer.Option(0.15, help="time step"),
    cmap: str = typer.Option("copper", help="look: copper | gold | ice | viridis"),
    fps: int = typer.Option(25, help="gif frame rate"),
    seed: int = typer.Option(0, help="RNG seed"),
    gif: Optional[Path] = typer.Option(None, help="output gif path (if omitted, just runs)"),
) -> None:
    """Thin-film dewetting reference solver / GIF maker."""
    if gif is not None:
        render_gif(str(gif), n, steps, every, mean, precursor, drop, strength, gamma, dt, cmap, fps, seed)
    else:
        sim = Dewetting(n=n, mean=mean, precursor=precursor, drop=drop, strength=strength, gamma=gamma, dt=dt, seed=seed)
        sim.run(steps)
        print(f"ran {steps} steps · h∈[{sim.h.min():.3f},{sim.h.max():.3f}] · mass drift {abs(sim.mass - mean):.2e}")


if __name__ == "__main__":
    typer.run(main)
