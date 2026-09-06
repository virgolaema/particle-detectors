"""Cross-cube gamma/alpha coincidence in a fragmented 3x3x1 tile.

The borated-PVT tile (3 x 3 x 1 cm) is imagined cut into a matrix of nine
1 cm^3 cubes (3 in x, 3 in y, one layer in z). A single 10B(n,alpha)7Li*
thermal capture produces, at one point:

  * the alpha + 7Li recoil, range < 10 um in PVT  -> ALL its energy stays in
    the cube where the capture happened ("alpha cube");
  * the 478 keV de-excitation gamma (94% branch), emitted isotropically, with
    a mean free path of ~10 cm in PVT -- much larger than the tile -- so it
    usually escapes, but sometimes Compton-scatters in a *different* cube.

This module answers: with the tile fragmented into nine cubes, what is the
probability that the gamma deposits energy in a cube other than the alpha cube
(a genuine two-cube coincidence)? That is the number that decides whether a
segmented tile could tag captures by alpha-here / gamma-there coincidence.

Physics choices, and why:
  * The 478 keV attenuation is taken from Klein-Nishina directly, NOT from the
    lib_photon NIST table, which is a factor ~4 too low at 0.5 MeV (it gives
    mu/rho ~ 0.023 cm^2/g where KN and the tabulated water value both put it at
    ~0.096 cm^2/g). At 478 keV in low-Z plastic Compton is >99% of the total
    cross section, so KN is an excellent stand-in and self-consistent for the
    down-scattered tracking as well.
  * Photoelectric absorption is negligible in C/H down to ~20 keV, below which
    the residual photon energy is simply deposited locally. Rayleigh scatter
    (no energy deposit, small angle) is ignored.
  * Compton electrons have range << 1 cm, so their energy is deposited at the
    interaction point (in that point's cube).

Run as a script to print the numbers and write the two report figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M_E = 510.99895  # keV

# ---- tile geometry: 3 x 3 x 1 cm, nine 1 cm cubes -------------------------
NX, NY, NZ = 3, 3, 1
LX, LY, LZ = float(NX), float(NY), float(NZ)  # cm
# tile spans x,y in [-1.5, 1.5], z in [0, 1]
X0, Y0, Z0 = -LX / 2, -LY / 2, 0.0

E_GAMMA = 478.0  # keV, 7Li* -> 7Li
E_CUT = 20.0     # keV; below this deposit locally (photo-absorbed)

# PVT ~ CH_1.1, density 1.023 g/cm^3
_RHO = 1.023
_ZA = (12 / 13.1) * 6 / 12.011 + (1.1 / 13.1) * 1 / 1.008
_NE = _ZA * 6.02214076e23 * _RHO  # electrons / cm^3
_RE = 2.8179403262e-13            # cm


def kn_total_xsec(E):
    """Klein-Nishina total cross section per electron [cm^2], E in keV."""
    a = E / M_E
    pref = 2 * np.pi * _RE**2
    t1 = (1 + a) / a**2 * (2 * (1 + a) / (1 + 2 * a) - np.log1p(2 * a) / a)
    t2 = np.log1p(2 * a) / (2 * a) - (1 + 3 * a) / (1 + 2 * a) ** 2
    return pref * (t1 + t2)


def mu_of(E):
    """Linear attenuation coefficient [1/cm] at photon energy E [keV]."""
    return kn_total_xsec(E) * _NE


def _sample_compton_costheta(E, rng):
    """Sample cos(theta) from the KN differential cross section (array E)."""
    a = E / M_E
    out = np.empty_like(E)
    todo = np.arange(E.size)
    while todo.size:
        aa = a[todo]
        c = rng.uniform(-1.0, 1.0, todo.size)
        eps = 1.0 / (1.0 + aa * (1.0 - c))          # E'/E
        # dsigma/dcos ~ eps^2 (eps + 1/eps - (1-c^2)); envelope value at eps=1
        f = eps**2 * (eps + 1.0 / eps - (1.0 - c * c))
        fmax = 1.0 + 1.0 / (1.0 + 2.0 * aa) ** 3    # safe upper bound >= max f
        acc = rng.uniform(0.0, 1.0, todo.size) * fmax < f
        out[todo[acc]] = c[acc]
        todo = todo[~acc]
    return out


def _cube_index(x, y):
    """Cube id 0..8 for interaction at (x,y); NZ==1 so z is not indexed."""
    ix = np.clip(((x - X0) // 1.0).astype(int), 0, NX - 1)
    iy = np.clip(((y - Y0) // 1.0).astype(int), 0, NY - 1)
    return ix + NX * iy


@dataclass
class Result:
    n: int
    p_escape: float              # gamma leaves tile with no interaction
    p_same_only: float           # deposits only in the alpha cube
    p_cross_any: float           # any deposit in a cube != alpha cube
    p_cross_50: float            # >50 keV in some cube != alpha cube
    mean_edep_tile: float        # keV deposited in the whole tile per gamma
    mean_edep_cross: float       # keV deposited outside the alpha cube per gamma
    edep_map_center: np.ndarray  # 3x3 mean keV per cube, captures in centre cube
    pmap_center: np.ndarray      # 3x3 P(>50 keV) per cube, captures in centre cube


def simulate(n=400_000, uniform=True, seed=0) -> Result:
    """MC the gamma transport for `n` captures; alpha stays in its own cube.

    `uniform=True` samples the capture point uniformly in the tile. Set False
    for a front-biased (exp along z) capture profile sensitivity check.
    """
    rng = np.random.default_rng(seed)

    # capture positions (alpha cube = cube of this point)
    x = rng.uniform(X0, X0 + LX, n)
    y = rng.uniform(Y0, Y0 + LY, n)
    if uniform:
        z = rng.uniform(Z0, Z0 + LZ, n)
    else:
        # exp toward the front face z=1 (source side), lambda 0.5 cm
        z = (Z0 + LZ) - rng.exponential(0.5, n)
        z = np.clip(z, Z0, Z0 + LZ)
    alpha_cube = _cube_index(x, y)

    # per-cube deposited energy, and a "center-cube capture" accumulator
    edep = np.zeros((n, NX * NY))
    is_center = alpha_cube == _cube_index(np.array([0.0]), np.array([0.0]))[0]

    # isotropic gamma direction
    ct = rng.uniform(-1.0, 1.0, n)
    st = np.sqrt(1.0 - ct * ct)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    dx, dy, dz = st * np.cos(ph), st * np.sin(ph), ct
    E = np.full(n, E_GAMMA)
    alive = np.ones(n, bool)
    px, py, pz = x.copy(), y.copy(), z.copy()

    for _ in range(200):  # generous scatter cap; almost all finish in <5
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        Ei = E[idx]
        step = rng.exponential(1.0 / mu_of(Ei))
        nx = px[idx] + dx[idx] * step
        ny = py[idx] + dy[idx] * step
        nz = pz[idx] + dz[idx] * step
        inside = (
            (nx >= X0) & (nx <= X0 + LX)
            & (ny >= Y0) & (ny <= Y0 + LY)
            & (nz >= Z0) & (nz <= Z0 + LZ)
        )
        left = idx[~inside]
        alive[left] = False  # escaped the tile

        stay = idx[inside]
        if stay.size == 0:
            continue
        px[stay], py[stay], pz[stay] = nx[inside], ny[inside], nz[inside]
        Es = E[stay]
        # Compton scatter
        c = _sample_compton_costheta(Es, rng)
        Eprime = Es / (1.0 + (Es / M_E) * (1.0 - c))
        Te = Es - Eprime  # electron energy, deposited locally
        cube = _cube_index(px[stay], py[stay])
        np.add.at(edep, (stay, cube), Te)
        E[stay] = Eprime

        # new direction (rotate by theta, random azimuth) -- isotropic-in-azimuth
        # is enough for a spatial-coincidence question; use fresh isotropic pick
        # about the scatter cone.
        theta = np.arccos(np.clip(c, -1, 1))
        _rotate(dx, dy, dz, stay, theta, rng)

        # low-energy photons: absorb locally and stop
        done = E[stay] < E_CUT
        if done.any():
            dd = stay[done]
            cd = _cube_index(px[dd], py[dd])
            np.add.at(edep, (dd, cd), E[dd])
            E[dd] = 0.0
            alive[dd] = False

    # ---- classify ---------------------------------------------------------
    tot = edep.sum(1)
    in_alpha = edep[np.arange(n), alpha_cube]
    out_alpha = tot - in_alpha
    other_max = tot.copy()
    e2 = edep.copy()
    e2[np.arange(n), alpha_cube] = 0.0
    other_max = e2.max(1)

    escaped = tot <= 0
    same_only = (~escaped) & (out_alpha <= 1e-6)
    cross_any = out_alpha > 1e-6
    cross_50 = other_max > 50.0

    # center-cube capture maps (3x3)
    ci = np.where(is_center)[0]
    emap = edep[ci].mean(0).reshape(NY, NX)
    pmap = (edep[ci] > 50.0).mean(0).reshape(NY, NX)

    return Result(
        n=n,
        p_escape=float(escaped.mean()),
        p_same_only=float(same_only.mean()),
        p_cross_any=float(cross_any.mean()),
        p_cross_50=float(cross_50.mean()),
        mean_edep_tile=float(tot.mean()),
        mean_edep_cross=float(out_alpha.mean()),
        edep_map_center=emap,
        pmap_center=pmap,
    )


def _rotate(dx, dy, dz, idx, theta, rng):
    """Rotate directions idx by polar angle theta about their current axis,
    with a uniformly random azimuth. In-place update of dx,dy,dz."""
    ux, uy, uz = dx[idx], dy[idx], dz[idx]
    phi = rng.uniform(0.0, 2 * np.pi, idx.size)
    st, cth = np.sin(theta)[:, None], np.cos(theta)[:, None]
    # e1 = normalize(u x a), with a = z-hat unless u ~ z-hat then a = x-hat
    axv = np.stack([np.where(np.abs(uz) < 0.9, 0.0, 1.0),
                    np.zeros_like(uz),
                    np.where(np.abs(uz) < 0.9, 1.0, 0.0)], 1)
    u = np.stack([ux, uy, uz], 1)
    e1 = np.cross(u, axv)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    new = (cth * u
           + st * np.cos(phi)[:, None] * e1
           + st * np.sin(phi)[:, None] * e2)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    dx[idx], dy[idx], dz[idx] = new[:, 0], new[:, 1], new[:, 2]


def _figures(res: Result, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) outcome breakdown
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cats = ["gamma escapes\n(no deposit)", "deposit only in\nalpha cube",
            "deposit in\nanother cube"]
    vals = [res.p_escape, res.p_same_only, res.p_cross_any]
    cols = ["#9aa0a6", "#2E86AB", "#C0392B"]
    b = ax.bar(cats, [100 * v for v in vals], color=cols)
    ax.bar_label(b, fmt="%.1f%%", padding=2)
    ax.set_ylabel("probability [%]")
    ax.set_title("478 keV capture-gamma vs alpha, 3x3x1 tile (nine 1 cm cubes)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(f"{outdir}/cube_coincidence_outcomes.png", dpi=150)
    plt.close(fig)

    # (2) leakage map for a capture in the central cube
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(100 * res.pmap_center, origin="lower", cmap="magma",
                   extent=[0, 3, 0, 3])
    for iy in range(3):
        for ix in range(3):
            v = 100 * res.pmap_center[iy, ix]
            ax.text(ix + 0.5, iy + 0.5, f"{v:.2f}%", ha="center", va="center",
                    color="white" if v < 6 else "black", fontsize=10)
    ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 1, 2, 3])
    ax.set_title("P(gamma deposits >50 keV) per cube\ncapture (alpha) in centre cube")
    fig.colorbar(im, label="probability [%]")
    fig.tight_layout()
    fig.savefig(f"{outdir}/cube_coincidence_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    r = simulate(n=400_000, uniform=True, seed=1)
    print(f"mu(478 keV)      = {mu_of(478.0):.4f} /cm   (mfp {1/mu_of(478.0):.2f} cm)")
    print(f"captures         = {r.n}")
    print(f"P(gamma escapes) = {100*r.p_escape:.2f} %")
    print(f"P(same cube only)= {100*r.p_same_only:.2f} %")
    print(f"P(cross-cube any)= {100*r.p_cross_any:.2f} %   <-- gamma-here/alpha-there")
    print(f"P(cross-cube>50k)= {100*r.p_cross_50:.2f} %")
    print(f"<Edep tile>      = {r.mean_edep_tile:.1f} keV/gamma")
    print(f"<Edep off-cube>  = {r.mean_edep_cross:.1f} keV/gamma")
    # front-biased sensitivity
    rf = simulate(n=200_000, uniform=False, seed=2)
    print(f"[front-biased]   P(cross-cube any)= {100*rf.p_cross_any:.2f} %  "
          f"P(cross>50k)= {100*rf.p_cross_50:.2f} %")
    _figures(r, here)
    print("figures: cube_coincidence_outcomes.png, cube_coincidence_map.png")
