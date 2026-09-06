"""Hydrogen radiative capture in the C8H8 scintillator, and the energy at which
captures vs elastic scatters actually happen.

Three products, all written as report figures:

  1. `pns_hcapture_xsec.png` -- the microscopic cross sections vs neutron
     energy: H(n,gamma) radiative capture (1/v, 332 mb at 2200 m/s) against the
     H and C elastic-scattering cross sections that dominate the interaction
     budget. This is the "cross section as a function of energy" plot.

  2. `pns_tof_xsec_overlay.png` -- the arrival time-of-flight spectrum at the
     tile (from the MC), with the H-capture cross section overlaid on a second
     axis. Because ToF maps monotonically to energy (t ~ 1/sqrt(E)), the
     overlay shows that the capture cross section is large exactly in the late
     (slow, thermal) part of the ToF distribution.

  3. `pns_interactions_by_energy.png` -- the spectrum convolved with each
     process' macroscopic cross section: the *interaction rate* per log-energy
     bin for radiative capture and for elastic scatter, as two histograms on
     one plot. This is "where in energy the two processes actually happen",
     folding the arriving flux against sigma(E).

The scintillator here is treated as pure C8H8 (styrene monomer, rho =
1.05 g/cm3): 8 H and 8 C per unit, so n_H = n_C. Boron capture is the
experiment's signal and is treated in section 4 of the note; this script is
about the plastic's own neutron channels (elastic thermalisation and the
2223 keV H(n,gamma) background line).

The arriving neutron spectrum (E_surv per sample, with the surface rate
weighting) is taken from the notebook MC. It is read from `_mc_state.pkl` if
present (written by `_dump_state.py`); otherwise the notebook is replayed here.
"""

from __future__ import annotations

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from lib_neutrons import get_endf_cross_sections, tof_constant_energy  # noqa: E402

N_A = 6.02214076e23

# ---- C8H8 scintillator -----------------------------------------------------
RHO_SCINT = 1.05          # g/cm^3
M_C8H8 = 8 * 12.011 + 8 * 1.008   # g/mol, = 104.15
N_UNIT = RHO_SCINT * N_A / M_C8H8  # C8H8 units / cm^3
N_H = 8 * N_UNIT          # H nuclei / cm^3
N_C = 8 * N_UNIT          # C nuclei / cm^3
T_SCINT = 1.0             # cm

# ---- H(n,gamma) radiative capture: 1/v, 332 mb at 2200 m/s -----------------
SIGMA_H_CAP_TH = 0.332    # barn at E_th = 25.3 meV (v = 2200 m/s)
E_TH_EV = 0.0253


def sigma_h_capture_b(E_eV):
    """H-1 radiative capture cross section [barn] vs energy, pure 1/v."""
    return SIGMA_H_CAP_TH * np.sqrt(E_TH_EV / np.asarray(E_eV, float))


def _load_state():
    pkl = os.path.join(HERE, "_mc_state.pkl")
    if os.path.exists(pkl):
        return pickle.load(open(pkl, "rb"))
    # fall back to a live replay (slow)
    import json
    import contextlib
    import io
    nb = json.load(open(os.path.join(HERE, "pns.ipynb")))
    g = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(HERE)
    with contextlib.redirect_stdout(io.StringIO()):
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if s.strip():
                exec(compile(s, f"<{c.get('id')}>", "exec"), g)
            if c.get("id") == "1bc91a56":
                break
    os.chdir(cwd)
    return g


# ===========================================================================
# figure 1: cross sections vs energy
# ===========================================================================
def fig_cross_sections():
    E = np.logspace(-3, 7, 2000)  # eV, 1 meV -> 10 MeV
    sHcap = sigma_h_capture_b(E)
    sHel, sCel = get_endf_cross_sections(E)  # barn (bound-H at thermal)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(E, sHel, color="#2E86AB", lw=2, label="H elastic scatter (ENDF)")
    ax.loglog(E, sCel, color="#7f8c8d", lw=2, label="C elastic scatter (ENDF)")
    ax.loglog(E, sHcap, color="#C0392B", lw=2.4,
              label=r"H($n,\gamma$) capture ($1/v$)")
    ax.axvline(E_TH_EV, color="seagreen", ls=":", lw=1.2)
    ax.text(E_TH_EV * 1.3, 2e-4, "thermal\n25.3 meV", color="seagreen",
            fontsize=8, va="bottom")
    ax.set_xlabel("neutron energy  [eV]")
    ax.set_ylabel(r"microscopic cross section  $\sigma$  [barn]")
    ax.set_title("Neutron cross sections in C$_8$H$_8$ scintillator")
    ax.set_ylim(1e-4, 1e3)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pns_hcapture_xsec.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# helpers for the spectrum-weighted figures
# ===========================================================================
def _sample_arrays(st):
    """Yield (key, E_eV, weight) per sample from the MC state.

    weight is the per-neutron surface-rate weight so a histogram of E with
    these weights is the arriving flux in n/s at 100% duty.
    """
    RES, SAMPLES = st["RES"], st["SAMPLES"]
    for key in SAMPLES:
        r = RES[key]
        E = np.asarray(r["E_surv"], float)
        w = np.full(E.size, r["rate_surface"] / E.size)
        yield key, r, E, w


def fig_tof_overlay(st):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (key, r, E, w) in zip(axes, _sample_arrays(st)):
        tof_us = tof_constant_energy(E, r["distance_cm"]) * 1e6
        bins = np.logspace(np.log10(max(tof_us.min(), 1e-3)),
                           np.log10(tof_us.max()), 60)
        ax.hist(tof_us, bins=bins, weights=w, color="#34495e", alpha=0.75,
                label="arriving neutrons")
        ax.set_xscale("log")
        ax.set_xlabel(r"time of flight  [$\mu$s]")
        ax.set_title(f"Sample {key} — {r['distance_cm']:.0f} cm")
        # overlay H-capture cross section vs the energy that maps to each ToF
        ax2 = ax.twinx()
        tt = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
        # invert tof_constant_energy: E from tof (non-relativistic)
        # tof[s] = distance[cm]*1e-2 / v ; v=sqrt(2E/m); E in eV
        M_N, EV_J = 1.675e-27, 1.602e-19
        v = (r["distance_cm"] * 1e-2) / (tt * 1e-6)
        E_of_t = 0.5 * M_N * v**2 / EV_J
        ax2.loglog(tt, sigma_h_capture_b(E_of_t), color="#C0392B", lw=2.2)
        ax2.set_ylabel(r"H($n,\gamma$) $\sigma$  [barn]", color="#C0392B")
        ax2.tick_params(axis="y", colors="#C0392B")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("arriving flux  [n/s per bin]")
    fig.suptitle("Arrival ToF spectrum with H-capture cross section overlaid\n"
                 "(late ToF = slow = thermal, where capture is large)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pns_tof_xsec_overlay.png"), dpi=140)
    plt.close(fig)


def fig_interactions_by_energy(st):
    """Convolve the arriving spectrum with each process' macroscopic cross
    section: interaction rate per log-energy bin for capture and scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    chances = {}
    for ax, (key, r, E, w) in zip(axes, _sample_arrays(st)):
        sHcap = sigma_h_capture_b(E) * 1e-24
        sHel, sCel = get_endf_cross_sections(E)
        Sig_cap = N_H * sHcap                       # H(n,gamma)
        Sig_scat = N_H * sHel * 1e-24 + N_C * sCel * 1e-24
        p_cap = 1.0 - np.exp(-Sig_cap * T_SCINT)    # single-pass
        p_scat = 1.0 - np.exp(-Sig_scat * T_SCINT)
        bins = np.logspace(-3, 7, 60)
        ax.hist(E, bins=bins, weights=w * p_scat, histtype="stepfilled",
                color="#2E86AB", alpha=0.55, label="elastic scatter")
        ax.hist(E, bins=bins, weights=w * p_cap, histtype="stepfilled",
                color="#C0392B", alpha=0.65, label=r"H($n,\gamma$) capture")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2e2)
        ax.set_xlabel("neutron energy  [eV]")
        ax.set_title(f"Sample {key} — {r['distance_cm']:.0f} cm")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")
        R_cap = float((w * p_cap).sum())
        R_scat = float((w * p_scat).sum())
        R_surf = float(w.sum())
        chances[key] = dict(R_surf=R_surf, R_cap=R_cap, R_scat=R_scat,
                            chance_cap=R_cap / R_surf, chance_scat=R_scat / R_surf)
    axes[0].set_ylabel("single-pass interaction rate  [n/s per bin]")
    fig.suptitle("Interactions in 1 cm C$_8$H$_8$ by energy: arriving spectrum "
                 r"$\times$ single-pass probability, per process",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pns_interactions_by_energy.png"), dpi=140)
    plt.close(fig)
    return chances


if __name__ == "__main__":
    print(f"C8H8: n_H = n_C = {N_H:.3e} /cm^3   (rho {RHO_SCINT} g/cm^3)")
    fig_cross_sections()
    print("wrote pns_hcapture_xsec.png")
    st = _load_state()
    fig_tof_overlay(st)
    print("wrote pns_tof_xsec_overlay.png")
    ch = fig_interactions_by_energy(st)
    print("wrote pns_interactions_by_energy.png")
    print("\nSingle-pass chance per arriving neutron (1 cm C8H8):")
    for k, d in ch.items():
        print(f"  Sample {k}:  H(n,g) capture = {100*d['chance_cap']:.3f} %   "
              f"elastic scatter = {100*d['chance_scat']:.2f} %   "
              f"(R_cap={d['R_cap']:.3f} Hz, R_scat={d['R_scat']:.1f} Hz)")
