"""
Rayleigh–Plateau instability — slender-jet reference solver.

A liquid column of radius R₀ is unstable: surface tension lowers its area by
pinching it into a chain of droplets. A sinusoidal radius perturbation of
wavelength λ grows whenever λ > 2πR₀ (circumference), and the *fastest*-growing
mode is λ ≈ 9.02 R₀ (wavenumber kR₀ ≈ 0.697) — that sets the droplet spacing.

We solve the one-dimensional slender-jet (lubrication) equations of Eggers &
Dupont for the radius h(z,t) and axial velocity v(z,t) on a periodic column
(surface tension and density scaled to 1):

    ∂ₜ(h²) = −∂_z(h² v)                                   (volume conservation)
    ∂ₜ v   = −v ∂_z v − ∂_z κ + (3ν/h²) ∂_z(h² ∂_z v)     (momentum)

with the full mean curvature of the free surface

    κ = 1/(h√(1+h_z²)) − h_zz/(1+h_z²)^{3/2}.

The −∂_z κ term is the Laplace pressure gradient that drives the instability;
the viscous term (Ohnesorge number ν) regularises pinch-off. Integrated
explicitly with RK2 until the necks pinch into droplets.

Usage
-----
    uv run python rayleigh_plateau.py --gif ../assets/breakup.gif
    uv run python rayleigh_plateau.py --modes 4 --visc 0.1 --gif ../assets/satellites.gif
"""

from pathlib import Path
from typing import Optional

import numpy as np
import typer


class RayleighPlateau:
    def __init__(
        self,
        nz: int = 400,
        length: float = 36.0,
        r0: float = 1.0,
        modes: int = 4,
        amp: float = 0.04,
        visc: float = 0.15,
        dt: float = 2.0e-3,
        seed: int = 0,
        hfloor: float = 0.06,
    ):
        self.nz = nz
        self.dz = length / nz
        self.L = length
        self.visc = visc
        self.dt = dt
        self.hfloor = hfloor

        z = np.arange(nz) * self.dz
        # `modes` full waves across the column → wavelength λ = L/modes. Unstable
        # when λ > 2πr0; the seed picks an unstable, near-fastest mode.
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi)
        self.h = r0 * (1.0 + amp * np.cos(2 * np.pi * modes * z / length + phase))
        self.v = np.zeros(nz)
        self.r0 = r0

    # periodic central differences
    def _dz(self, f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * self.dz)

    def _dzz(self, f):
        return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / self.dz**2

    def _curv(self, h):
        hz = self._dz(h)
        hzz = self._dzz(h)
        denom = np.sqrt(1.0 + hz**2)
        return 1.0 / (h * denom) - hzz / denom**3

    def _rhs(self, h, v):
        h = np.maximum(h, self.hfloor)
        h2 = h * h
        kappa = self._curv(h)
        # volume: ∂ₜh² = −∂_z(h² v)  →  ∂ₜh = −∂_z(h² v)/(2h)
        dh2 = -self._dz(h2 * v)
        dh = dh2 / (2.0 * h)
        # momentum: advection + Laplace-pressure gradient + viscous diffusion
        visc_term = (3.0 * self.visc / h2) * self._dz(h2 * self._dz(v))
        dv = -v * self._dz(v) - self._dz(kappa) + visc_term
        return dh, dv

    def step(self):
        # RK2 (midpoint)
        dh1, dv1 = self._rhs(self.h, self.v)
        hm = self.h + 0.5 * self.dt * dh1
        vm = self.v + 0.5 * self.dt * dv1
        dh2, dv2 = self._rhs(hm, vm)
        self.h = np.maximum(self.h + self.dt * dh2, self.hfloor)
        self.v = self.v + self.dt * dv2

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.h

    @property
    def min_radius(self):
        return float(self.h.min())

    @property
    def volume(self):
        return float((self.h**2).sum() * self.dz)


# --- rendering: revolve the radius profile into a lit 3D-looking tube ---------

def _render(h: np.ndarray, r0: float, height: int = 220) -> np.ndarray:
    """Side view of the axisymmetric jet: a shaded tube of local radius h(z)."""
    nz = h.size
    img = np.zeros((height, nz, 3), dtype=np.float32)
    yc = height / 2.0
    rmax = r0 * 1.8
    ys = (np.arange(height) - yc)[:, None]  # (height,1)
    R = (h / rmax * (height * 0.46))[None, :]  # pixel radius per column
    inside = np.abs(ys) <= R
    # Surface normal of a cylinder cross-section → Lambert + specular shading.
    with np.errstate(invalid="ignore", divide="ignore"):
        sn = np.clip(ys / np.maximum(R, 1e-6), -1.0, 1.0)
    cosang = np.sqrt(np.clip(1.0 - sn**2, 0.0, 1.0))  # facing camera
    light = np.clip(0.35 + 0.75 * cosang - 0.25 * sn, 0.0, 1.0)
    spec = np.clip(cosang, 0, 1) ** 18
    col = np.stack([0.30 + 0.7 * light, 0.55 + 0.45 * light, 0.95 * np.ones_like(light)], axis=-1)
    col = np.clip(col + spec[..., None] * 0.8, 0, 1)
    img[inside] = col[inside]
    img *= 1.0  # background stays black
    return (img * 255).astype(np.uint8)


def render_gif(out_path, nz, length, r0, modes, amp, visc, dt, steps, every, fps, seed, pinch=0.12):
    import imageio.v2 as imageio

    sim = RayleighPlateau(nz=nz, length=length, r0=r0, modes=modes, amp=amp, visc=visc, dt=dt, seed=seed)
    frames, V0, pinched_at = [], length * r0**2, None
    for s in range(steps + 1):
        if s % every == 0:
            frames.append(_render(sim.h, r0))
        # Stop at pinch-off: the 1D model can't represent the neck snapping, so
        # the droplets are "done" once the thinnest neck reaches `pinch`.
        if sim.min_radius < pinch:
            frames.append(_render(sim.h, r0))
            pinched_at = s
            break
        sim.step()
    # hold the final pinched state so the loop lingers on the droplets
    for _ in range(fps):
        frames.append(frames[-1])
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(
        f"wrote {out_path}  ·  {len(frames)} frames  ·  drops={modes}  ·  "
        f"pinch@step {pinched_at}  ·  min radius {sim.min_radius:.3f}  ·  "
        f"volume drift {abs(sim.volume - V0) / V0:.2e}"
    )


def main(
    nz: int = typer.Option(400, help="grid points along the jet"),
    length: float = typer.Option(36.0, help="column length L (in units of R₀)"),
    r0: float = typer.Option(1.0, help="unperturbed radius R₀"),
    modes: int = typer.Option(4, help="number of droplets to form (waves across L)"),
    amp: float = typer.Option(0.04, help="initial perturbation amplitude"),
    visc: float = typer.Option(0.15, help="viscosity ν (Ohnesorge regularisation)"),
    dt: float = typer.Option(2.0e-3, help="time step"),
    steps: int = typer.Option(60000, help="time steps"),
    every: int = typer.Option(500, help="capture a frame every N steps"),
    fps: int = typer.Option(30, help="gif frame rate"),
    seed: int = typer.Option(0, help="RNG seed (perturbation phase)"),
    gif: Optional[Path] = typer.Option(None, help="output gif path (if omitted, just runs)"),
) -> None:
    """Rayleigh–Plateau jet break-up reference solver / GIF maker."""
    if gif is not None:
        render_gif(str(gif), nz, length, r0, modes, amp, visc, dt, steps, every, fps, seed)
    else:
        sim = RayleighPlateau(nz=nz, length=length, r0=r0, modes=modes, amp=amp, visc=visc, dt=dt, seed=seed)
        sim.run(steps)
        print(f"ran {steps} steps · min radius {sim.min_radius:.3f} · volume drift "
              f"{abs(sim.volume - (length * r0**2)) / (length * r0**2):.2e}")


if __name__ == "__main__":
    typer.run(main)
