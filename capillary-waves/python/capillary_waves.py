"""
Gravity–capillary surface waves — spectral reference.

Linear surface waves on a pond of depth H obey the dispersion relation

    ω(k)² = (g·k + (σ/ρ)·k³) · tanh(k·H)

where k = |k| is the wavenumber, g gravity and σ/ρ the surface tension per
unit density. The two terms are the two restoring forces:

    • g·k          — gravity. Dominates at LONG wavelengths (small k).
    • (σ/ρ)·k³     — surface tension. Dominates at SHORT wavelengths and makes
                     their group velocity *grow* with k — so the tiniest ripples
                     race outward ahead of the longer swell. That outward-racing
                     fine structure is the visual signature of capillarity.

Because the dispersion is non-local (a fractional power of k), it is solved
exactly in Fourier space. For an initial surface h₀ released from rest, every
Fourier mode is an undamped/­damped oscillator, so

    ĥ(k, t) = ĥ₀(k) · cos(ω(k) t) · e^{−γ t}

and the surface at any time is a single inverse FFT — no time-step error. This
is the source of truth; the WebGL demo mirrors it with a fast local PDE proxy.

Usage
-----
    uv run python capillary_waves.py --gif ../assets/ripple.gif
    uv run python capillary_waves.py --sigma 0.0 --gif ../assets/gravity.gif
    uv run python capillary_waves.py --drops 5 --cmap ocean --gif ../assets/rain.gif
"""

from pathlib import Path
from typing import Optional

import numpy as np
import typer


def wavenumbers(n: int, dx: float):
    """Return (|k|, kx, ky) on an n×n periodic grid via FFT frequencies."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return np.sqrt(kx**2 + ky**2), kx, ky


class CapillaryWaves:
    """Exact spectral propagator for linear gravity–capillary waves."""

    def __init__(
        self,
        n: int = 256,
        dx: float = 1.0,
        g: float = 0.30,
        sigma: float = 0.40,
        depth: float = 8.0,
        damping: float = 0.04,
        drops: int = 1,
        seed: int = 0,
    ):
        self.n = n
        kk, _, _ = wavenumbers(n, dx)
        # ω(k); the k=0 mode has ω=0 (a flat offset that never oscillates).
        self.omega = np.sqrt((g * kk + sigma * kk**3) * np.tanh(kk * depth))
        self.damping = damping

        rng = np.random.default_rng(seed)
        h0 = np.zeros((n, n))
        xs = np.linspace(0, n, n, endpoint=False)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        r = max(2.0, n / 90.0)  # drop radius in cells
        for _ in range(max(1, drops)):
            cx, cy = rng.uniform(0, n, size=2)
            # periodic Gaussian dimple
            ddx = np.minimum(np.abs(gx - cx), n - np.abs(gx - cx))
            ddy = np.minimum(np.abs(gy - cy), n - np.abs(gy - cy))
            h0 += np.exp(-(ddx**2 + ddy**2) / (2.0 * r**2))
        self.h0_hat = np.fft.fft2(h0)

    def height(self, t: float) -> np.ndarray:
        """Surface height at absolute time t (released from rest at t=0)."""
        h_hat = self.h0_hat * np.cos(self.omega * t) * np.exp(-self.damping * t)
        return np.real(np.fft.ifft2(h_hat))


# --- colormaps for the GIF ---------------------------------------------------

def _shade(field: np.ndarray, cmap_name: str, gain: float) -> np.ndarray:
    """Map the height field to an 8-bit RGB frame with lit, embossed shading.

    `gain` rescales the (tiny) surface slopes to a visible range; it is computed
    once per animation so brightness stays steady while ripples disperse.
    """
    import matplotlib

    gx, gy = np.gradient(field)
    gx *= gain
    gy *= gain
    lx, ly, lz = 0.5, 0.6, 0.8
    nz = 1.0 / np.sqrt(gx**2 + gy**2 + 1.0)
    diff = np.clip((-gx * lx - gy * ly + lz) * nz, 0.0, 1.0)
    spec = diff**40

    if cmap_name == "schlieren":
        t = np.clip(0.5 + 0.5 * gain * field, 0.0, 1.0)
        rgb = matplotlib.colormaps["RdBu_r"](t)[..., :3]
    else:
        base = {"water": "GnBu_r", "ocean": "ocean", "mercury": "bone"}.get(cmap_name, "GnBu_r")
        rgb = matplotlib.colormaps[base](0.30 + 0.55 * diff)[..., :3]
        rgb = np.clip(rgb + spec[..., None] * 0.7, 0.0, 1.0)  # specular glints
    return (rgb * 255).astype(np.uint8)


def render_gif(
    out_path: str, n: int, frames: int, dt: float, g: float, sigma: float,
    depth: float, damping: float, drops: int, cmap: str, fps: int, seed: int,
) -> None:
    import imageio.v2 as imageio

    sim = CapillaryWaves(n=n, g=g, sigma=sigma, depth=depth, damping=damping, drops=drops, seed=seed)
    fields = [sim.height(i * dt) for i in range(frames)]
    # One gain for the whole animation: target a typical slope magnitude of ~0.9
    # from the 95th-percentile slope seen across all frames (steady brightness).
    slopes = np.concatenate([np.hypot(*np.gradient(f)).ravel() for f in fields[1:]])
    ref = np.percentile(slopes, 95) + 1e-9
    gain = 0.9 / ref
    imgs = [_shade(f, cmap, gain) for f in fields]
    imageio.mimsave(out_path, imgs, fps=fps, loop=0)
    print(f"wrote {out_path}  ·  {frames} frames  ·  {n}×{n}  ·  ω_max={sim.omega.max():.3f}  ·  gain={gain:.1f}")


def main(
    n: int = typer.Option(256, help="grid size"),
    frames: int = typer.Option(120, help="number of frames"),
    dt: float = typer.Option(0.6, help="simulated time between frames"),
    g: float = typer.Option(0.30, help="gravity coefficient"),
    sigma: float = typer.Option(0.40, help="surface tension σ/ρ"),
    depth: float = typer.Option(8.0, help="water depth H"),
    damping: float = typer.Option(0.04, help="temporal damping γ"),
    drops: int = typer.Option(1, help="number of initial drops"),
    cmap: str = typer.Option("water", help="look: water | ocean | mercury | schlieren"),
    fps: int = typer.Option(30, help="gif frame rate"),
    seed: int = typer.Option(0, help="RNG seed for drop placement"),
    gif: Optional[Path] = typer.Option(None, help="output gif path (if omitted, just runs)"),
) -> None:
    """Gravity–capillary surface-wave reference solver / GIF maker."""
    if gif is not None:
        render_gif(str(gif), n, frames, dt, g, sigma, depth, damping, drops, cmap, fps, seed)
    else:
        sim = CapillaryWaves(n=n, g=g, sigma=sigma, depth=depth, damping=damping, drops=drops, seed=seed)
        h = sim.height(frames * dt)
        print(f"propagated to t={frames * dt:.1f} · ω_max={sim.omega.max():.3f} · rms={np.sqrt((h**2).mean()):.3e}")


if __name__ == "__main__":
    typer.run(main)
