"""Sample B report plots, on a FIXED version of the shield MC.

Fix vs pns.ipynb cell 2 (see NOTES.md):
  * H radiative capture was a flat ABSORPTION_PROB = 2% per H-scatter at every
    energy.  H(n,gamma) is 1/v (0.332 b at 25.3 meV), so that per-collision
    probability is right only AT thermal and ~1e-4 during slowdown; with ~18
    slowdown collisions the flat hack killed O(30%) of neutrons mid-moderation,
    inflating "H capture" by an order of magnitude and suppressing both the
    transmission and the B-10 capture fraction.  Here H is a proper 1/v
    absorber in every H-bearing layer, competing with scattering in Sigma_tot.
  * Fe(n,gamma) captures in the steel slab were lumped into "H capture";
    they are now their own fate category.
  * C(n,gamma) (3.5 mb thermal) remains neglected; the +-50 keV source spread
    remains unused (negligible against moderation).

Produces (Sample B only):
  sampleB_spectrum.png          - energy spectrum of neutrons reaching the tile
  sampleB_fates.png             - fate fractions, old MC vs fixed MC
  sampleB_tof_interactions.png  - TOF to the tile + tile capture/scatter
                                  probabilities + flux x probability overlays
"""
import os, pickle, sys, time as time_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib_neutrons import get_endf_cross_sections

# ── parameters (verbatim from pns.ipynb cell 0) ──────────────────────────────
E0_MEV, E0_eV = 2.45, 2.45e6
PE_THICKNESS_CM, BHDPE_DENSITY = 14.0, 1.0
BORON_MASS_FRAC, B10_ENRICHMENT = 0.05, 0.199
WOOD_THICKNESS_CM, WOOD_DENSITY = 1.0, 0.6
WOOD_H_FRAC, WOOD_C_FRAC, WOOD_O_FRAC = 0.06, 0.50, 0.44
TOTAL_SHIELD_CM = PE_THICKNESS_CM + WOOD_THICKNESS_CM
STEEL_CM, STEEL_DENSITY, A_FE = 0.5, 7.85, 55.85
SIGMA_FE_ABS_TH = 2.56          # barn, Fe(n,gamma) thermal
SIGMA_B10_THERMAL_BARNS = 3840.0
SIGMA_H_GAMMA_TH = 0.332        # barn, H(n,gamma) thermal  << THE FIX
E_THERMAL_EV = 0.0253
CONE_HALF_ANGLE_DEG = 20.0
N_NEUTRONS = 100_000
DIST_B_CM = TOTAL_SHIELD_CM + 15.0          # 30 cm source -> tile B
N_A = 6.022e23

n_B10_bhdpe = BHDPE_DENSITY * BORON_MASS_FRAC * B10_ENRICHMENT / 10.0 * N_A
n_H_bhdpe = BHDPE_DENSITY * 0.143 / 1.0 * N_A
n_C_bhdpe = BHDPE_DENSITY * 0.807 / 12.0 * N_A
n_H_wood = WOOD_DENSITY * WOOD_H_FRAC / 1.0 * N_A
n_Ceq_wood = (WOOD_DENSITY * WOOD_C_FRAC / 12.0 + WOOD_DENSITY * WOOD_O_FRAC / 16.0) * N_A
n_Fe = STEEL_DENSITY / A_FE * N_A

# tile (from pns.ipynb cell 5 / recoil cell, via _mc_state.pkl)
SCINT_THICKNESS_CM = 1.0
n_B10_scint = 1.05 * 0.02 * 0.199 / 10.0 * N_A          # 2% natural boron in PVT
n_H_scint, n_C_scint = 5.2514e22, 4.7262e22

# ── cross-section tables (identical construction to the notebook) ────────────
_FE_E_TBL = np.array([1e-2, 1e0, 1e2, 1e3, 1e4, 3e4, 5e4, 1e5, 3e5, 5e5, 1e6, 2.45e6, 5e6])
_FE_S_TBL = np.array([11.4, 11.3, 11.2, 11.1, 11.0, 10.0, 8.0, 6.5, 4.2, 3.8, 3.2, 2.9, 2.5])
_E_tbl = np.logspace(np.log10(0.025), np.log10(E0_eV * 1.5), 2000)
_sH_tbl, _sC_tbl = get_endf_cross_sections(_E_tbl)
_sFe_tbl = 10 ** np.interp(np.log10(_E_tbl), np.log10(_FE_E_TBL), np.log10(_FE_S_TBL))
_logE_tbl = np.log(_E_tbl)


def make_layer(name, thickness, z0, n_H, n_heavy, sigma_heavy_tbl, A_heavy, absorbers):
    """absorbers: list of (tag, n, sigma_thermal_barns) 1/v absorbers."""
    Ss = (n_H * _sH_tbl + n_heavy * sigma_heavy_tbl) * 1e-24
    fH = np.divide(n_H * _sH_tbl * 1e-24, Ss, out=np.zeros_like(Ss), where=Ss > 0)
    return dict(name=name, z0=z0, z1=z0 + thickness, A_heavy=A_heavy,
                absorbers=absorbers, Ss=Ss, fH=fH)


LAYERS = [make_layer('B-HDPE', PE_THICKNESS_CM, 0.0, n_H_bhdpe, n_C_bhdpe, _sC_tbl, 12.0,
                     [('B10', n_B10_bhdpe, SIGMA_B10_THERMAL_BARNS),
                      ('H',   n_H_bhdpe,   SIGMA_H_GAMMA_TH)])]
LAYERS.append(make_layer('wood', WOOD_THICKNESS_CM, LAYERS[-1]['z1'], n_H_wood,
                         n_Ceq_wood, _sC_tbl, 12.0,
                         [('H', n_H_wood, SIGMA_H_GAMMA_TH)]))
LAYERS.append(make_layer('steel', STEEL_CM, LAYERS[-1]['z1'], 0.0, n_Fe, _sFe_tbl, A_FE,
                         [('Fe', n_Fe, SIGMA_FE_ABS_TH)]))
Z_EXIT = LAYERS[-1]['z1']
AIR_CM = DIST_B_CM - Z_EXIT


def neutron_velocity(E_eV):
    return 1.3831e6 * np.sqrt(np.maximum(E_eV, 1e-6))


def update_mu(mu, cos_theta):
    sin_theta = np.sqrt(max(1.0 - cos_theta ** 2, 0.0))
    sin_mu = np.sqrt(max(1.0 - mu ** 2, 0.0))
    if sin_mu < 1e-10:
        return cos_theta if mu > 0 else -cos_theta
    cos_phi = 2.0 * np.random.random() - 1.0
    return mu * cos_theta + sin_mu * sin_theta * cos_phi


def layer_of(pos):
    for l in LAYERS:
        if pos < l['z1'] - 1e-12:
            return l
    return LAYERS[-1]


def run_mc(N=N_NEUTRONS):
    MU_MIN = np.cos(np.radians(CONE_HALF_ANGLE_DEG))
    EPS = 1e-9
    E_final = np.zeros(N)
    t_arr = np.zeros(N)
    fate = np.empty(N, dtype='U4')          # 'tran','back','B10','H','Fe'
    t0 = time_module.time()
    for i in range(N):
        E, pos, t = E0_eV, 0.0, 0.0
        mu = MU_MIN + (1.0 - MU_MIN) * np.random.random()
        while True:
            lay = layer_of(pos)
            logE = np.log(max(E, 0.025))
            Sigma_s = float(np.interp(logE, _logE_tbl, lay['Ss']))
            inv_sqrtE_term = np.sqrt(E_THERMAL_EV / max(E, E_THERMAL_EV))
            Sabs = [(tag, n * s_th * inv_sqrtE_term * 1e-24) for tag, n, s_th in lay['absorbers']]
            Sigma_a = sum(s for _, s in Sabs)
            Sigma_t = Sigma_s + Sigma_a
            step = -np.log(np.random.random()) / Sigma_t

            dist_face = ((lay['z1'] - pos) / mu if mu > 0
                         else (pos - lay['z0']) / (-mu) if mu < 0 else 1e30)
            if step >= dist_face:
                t += dist_face / neutron_velocity(E)
                pos += dist_face * mu + EPS * np.sign(mu)
                if pos >= Z_EXIT:
                    fate[i] = 'tran'; break
                if pos <= 0.0:
                    fate[i] = 'back'; break
                continue

            t += step / neutron_velocity(E)
            pos += step * mu

            if np.random.random() * Sigma_t < Sigma_a:        # absorbed: pick nucleus
                r = np.random.random() * Sigma_a
                for tag, s in Sabs:
                    r -= s
                    if r <= 0:
                        fate[i] = tag; break
                else:
                    fate[i] = Sabs[-1][0]
                break

            if np.random.random() < float(np.interp(logE, _logE_tbl, lay['fH'])):
                c = np.sqrt(np.random.random())               # H, A=1
                E = max(E * c * c, 0.025)
                mu = update_mu(mu, c)
            else:                                             # C or Fe, isotropic CM
                A = lay['A_heavy']
                cos_CM = 2.0 * np.random.random() - 1.0
                denom = 1.0 + 2.0 * A * cos_CM + A ** 2
                E = max(E * denom / (A + 1.0) ** 2, 0.025)
                mu = update_mu(mu, (1.0 + A * cos_CM) / np.sqrt(denom))
        E_final[i] = E
        t_arr[i] = t
    print(f"MC done: {N:,} neutrons in {time_module.time() - t0:.1f}s")
    return E_final, t_arr, fate


np.random.seed(20260724)
E_final, t_arr, fate = run_mc()
tran = fate == 'tran'
E_surv, t_shield = E_final[tran], t_arr[tran]
tof_us = (t_shield + AIR_CM / neutron_velocity(E_surv)) * 1e6   # to the tile face

counts = {k: int(np.sum(fate == k)) for k in ('tran', 'back', 'B10', 'H', 'Fe')}
print("fixed-MC fates [%]:", {k: round(100 * v / N_NEUTRONS, 2) for k, v in counts.items()})

# analytic thermal branching cross-check in B-HDPE
br = n_B10_bhdpe * SIGMA_B10_THERMAL_BARNS / (n_B10_bhdpe * SIGMA_B10_THERMAL_BARNS
                                              + n_H_bhdpe * SIGMA_H_GAMMA_TH)
print(f"analytic thermal branching B10:(B10+H) in B-HDPE = {br * 100:.1f}%  "
      f"(MC: {100 * counts['B10'] / max(counts['B10'] + counts['H'], 1):.1f}% "
      f"of shield captures)")

BLUE, GREY = 'steelblue', '0.55'

# ── figure 1: spectrum reaching the tile ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.logspace(np.log10(0.01), np.log10(4e6), 90)
ax.hist(E_surv, bins=bins, color=BLUE, alpha=0.75,
        label=f'transmitted  ({counts["tran"]:,}/{N_NEUTRONS:,} = {100 * counts["tran"] / N_NEUTRONS:.1f}%)')
for hi in (1.0, 1e5):
    ax.axvline(hi, color='k', lw=0.5, alpha=0.3)
frac = lambda lo, hi: 100 * np.mean((E_surv >= lo) & (E_surv < hi))
ax.set_title(f"Neutron spectrum reaching Sample B tile (30 cm, behind 0.5 cm steel)\n"
             f"thermal {frac(0, 1):.1f}%  |  epithermal {frac(1, 1e5):.1f}%  |  fast {frac(1e5, 4e6):.1f}%",
             fontsize=11)
ax.set_xscale('log')
ax.set_xlabel('Neutron energy  [eV]', fontsize=11)
ax.set_ylabel(f'Neutrons / bin  (of {N_NEUTRONS:,} emitted into 20° cone)', fontsize=11)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('sampleB_spectrum.png', dpi=120)

# ── figure 2: fates ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
cats = ['Transmitted', 'Backscattered', 'B-10 capture', 'H capture', 'Fe capture\n(steel)']
vals = [100 * counts[k] / N_NEUTRONS for k in ('tran', 'back', 'B10', 'H', 'Fe')]
bars = ax.bar(np.arange(len(cats)), vals, 0.6, color=BLUE)
for rect, v in zip(bars, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f'{v:.2f}' if v < 1 else f'{v:.1f}',
            ha='center', fontsize=10)
ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel('Fraction of emitted neutrons  [%]', fontsize=11)
ax.set_title(f'Neutron fates — Sample B stack (14 cm B-HDPE + 1 cm wood + 0.5 cm steel)\n'
             f'{N_NEUTRONS:,} neutrons at 2.45 MeV into the 20° cone', fontsize=11)
ax.grid(True, axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig('sampleB_fates.png', dpi=120)

# ── figure 3: TOF + tile FIRST-interaction probabilities ─────────────────────
# First-interaction branching: capture and scatter compete inside Sigma_tot,
# p_i = (Sigma_i / Sigma_tot) * (1 - exp(-Sigma_tot L)).  Computing each as an
# independent 1-exp(-Sigma_i L) double-counts (their sum exceeds the
# probability of interacting at all).
# "Visible scatter": kinematic fraction of scatters whose recoil DEPOSIT
# exceeds E_DEP_THR (raw keV, no quenching).  H recoil is uniform in [0, E_n]
# -> visible fraction (1 - thr/E_n); C recoil uniform in [0, 0.284 E_n].
sH_s, sC_s = get_endf_cross_sections(np.maximum(E_surv, 0.025))
sig_b10 = SIGMA_B10_THERMAL_BARNS * np.sqrt(E_THERMAL_EV / np.maximum(E_surv, E_THERMAL_EV))
Sig_c = n_B10_scint * sig_b10 * 1e-24
Sig_sH = n_H_scint * sH_s * 1e-24
Sig_sC = n_C_scint * sC_s * 1e-24
Sig_t = Sig_c + Sig_sH + Sig_sC
p_int = 1.0 - np.exp(-Sig_t * SCINT_THICKNESS_CM)
p_cap = Sig_c / Sig_t * p_int                       # first interaction = B-10 capture
p_sca = (Sig_sH + Sig_sC) / Sig_t * p_int           # first interaction = elastic scatter
E_DEP_THR_EV = 5e4                                  # 50 keV recoil deposit
f_H = np.clip(1.0 - E_DEP_THR_EV / np.maximum(E_surv, 1e-3), 0.0, 1.0)
f_C = np.clip(1.0 - E_DEP_THR_EV / (0.284 * np.maximum(E_surv, 1e-3)), 0.0, 1.0)
p_vis = (Sig_sH * f_H + Sig_sC * f_C) / Sig_t * p_int

GREEN_C, RED_S, RED_S2 = 'seagreen', 'crimson', 'lightcoral'
tb = np.logspace(np.log10(max(tof_us.min() * 0.8, 5e-3)), np.log10(tof_us.max() * 1.3), 70)
ctr = np.sqrt(tb[:-1] * tb[1:])
h_all = np.histogram(tof_us, bins=tb)[0]
h_cap = np.histogram(tof_us, bins=tb, weights=p_cap)[0]
h_sca = np.histogram(tof_us, bins=tb, weights=p_sca)[0]
h_vis = np.histogram(tof_us, bins=tb, weights=p_vis)[0]


def tof_figure(fname, with_any_scatter):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.hist(tof_us, bins=tb, color=BLUE, alpha=0.30, label='arrivals at tile (flux)')
    if with_any_scatter:
        ax.step(ctr, h_sca, where='mid', color=RED_S2, lw=1.8,
                label='flux × p(scatter first)  — any energy')
    ax.step(ctr, h_vis, where='mid', color=RED_S, lw=1.8,
            label=f'flux × p(scatter first)  — recoil deposit >{E_DEP_THR_EV/1e3:.0f} keV')
    ax.step(ctr, h_cap, where='mid', color=GREEN_C, lw=1.8,
            label='flux × p(B-10 capture first)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.5, None)
    ax.set_xlabel('Time of flight, generation → Sample B tile  [µs]', fontsize=11)
    ax.set_ylabel('Neutrons / bin', fontsize=11)
    ax.grid(True, which='both', alpha=0.25)

    ax2 = ax.twinx()
    ok = h_all > 3
    ax2.plot(ctr[ok], (h_cap / np.maximum(h_all, 1))[ok], '--', color=GREEN_C, lw=1.2,
             label='p(capture first)  (right)')
    h_p = h_sca if with_any_scatter else h_vis
    lbl = 'p(scatter first)  (right)' if with_any_scatter else 'p(visible scatter first)  (right)'
    ax2.plot(ctr[ok], (h_p / np.maximum(h_all, 1))[ok], '--', color=RED_S, lw=1.2, label=lbl)
    ax2.set_ylabel('first-interaction probability in 1 cm tile', fontsize=11)
    ax2.set_ylim(0, 1.02)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=8.5, loc='upper left')
    ax.set_title('TOF to Sample B  +  tile first-interaction probabilities  (curves overlaid, not stacked)\n'
                 'solid: flux and flux × probability (left, log)   dashed: probability vs arrival time (right)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)


tof_figure('sampleB_tof_interactions.png', with_any_scatter=True)
tof_figure('sampleB_tof_interactions_simple.png', with_any_scatter=False)
print(f"expected tile events per {N_NEUTRONS:,} emitted: capture {h_cap.sum():.0f} | "
      f"scatter(any E) {h_sca.sum():.0f} | scatter(deposit>{E_DEP_THR_EV/1e3:.0f} keV) {h_vis.sum():.0f}")
late = tof_us > 10.0
print(f"late arrivals (TOF>10 us): capture {p_cap[late].sum():.0f} vs visible scatter "
      f"{p_vis[late].sum():.0f}")

print(f"TOF: median {np.median(tof_us):.3g} µs | mean {np.mean(tof_us):.3g} µs | "
      f"1-99% {np.percentile(tof_us, 1):.3g}-{np.percentile(tof_us, 99):.3g} µs")
print(f"tile: <p_cap> {np.mean(p_cap) * 100:.2f}%  <p_scat> {np.mean(p_sca) * 100:.2f}%")

pickle.dump(dict(E_surv=E_surv, t_shield=t_shield, tof_us=tof_us, fate_counts=counts,
                 p_cap=p_cap, p_sca=p_sca, N=N_NEUTRONS),
            open('_mc_state_fixedB.pkl', 'wb'))
print("Saved → sampleB_spectrum.png, sampleB_fates.png, sampleB_tof_interactions.png, _mc_state_fixedB.pkl")
