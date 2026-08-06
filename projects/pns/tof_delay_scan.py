"""Purity / efficiency of the B-10 capture signal vs trigger delay (ToF cut).

Re-runs the PNS notebook MC (cells 0-8 of pns.ipynb), then scans a trigger
delay T: only events with time-of-flight > T are accepted.

    eff(T)    = R_capture(T) / R_capture(0)
    purity(T) = R_capture(T) / [R_capture(T) + R_visible-scatter(T)]

A scatter counts as background only if the incident neutron kinetic energy is
above a given threshold (the recoil it can produce is at most E_n).

Writes pns_delay_scan.png and prints a working-point table.
"""
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  SUPERSEDED MODEL — kept for reference, do not use for new results.
#
# This script is built on two things the MC no longer does (see NOTES.md,
# 2026-08-06):
#   * time of flight as a free flight from the shield exit, and
#   * the single-pass tile capture probability p(E), evaluated instantly.
# The delayed signal is not a time of flight. It is a die-away belonging to the
# tile (lambda ~ 2.8 us), so a "trigger delay" scan against ToF is not the right
# framing. Redo this against the die-away if you need a delayed-gate study.
# ─────────────────────────────────────────────────────────────────────────────
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

# ── replay the notebook up to the scintillator-rate cell ─────────────────────
# Replay by cell id, not by index: cells have been inserted since this was
# written and a hard-coded range silently replays the wrong ones.
_REPLAY_THROUGH = '1bc91a56'          # scintillator interaction-rate cell
nb = json.load(open('pns.ipynb'))
g = {'__name__': '__main__'}
for _c in nb['cells']:
    if _c['cell_type'] != 'code':
        continue
    src = ''.join(_c['source'])
    if src.strip():
        exec(compile(src, f"<cell {_c.get('id')}>", 'exec'), g)
    if _c.get('id') == _REPLAY_THROUGH:
        break

np = g['np']
from lib_neutrons import get_endf_cross_sections, tof_constant_energy

RES, SAMPLES = g['RES'], g['SAMPLES']
t_cm = g['SCINT_THICKNESS_CM']

# ── per-neutron weights and ToF ──────────────────────────────────────────────
for key in SAMPLES:
    r = RES[key]
    E = r['E_surv']
    sH, sC = get_endf_cross_sections(E)
    Sig_H = g['n_H_scint'] * sH * 1e-24
    Sig_s = Sig_H + g['n_C_scint'] * sC * 1e-24
    r['w_scatter'] = 1.0 - np.exp(-Sig_s * t_cm)          # any elastic scatter
    r['fH'] = Sig_H / Sig_s                                # H fraction of scatters
    r['w_capture'] = r['p_interact']
    r['tof_us'] = tof_constant_energy(E, r['distance_cm']) * 1e6

# neutron-kinetic-energy threshold for a scatter to be counted [keV]
THRESHOLDS_KEV = [0.0, 10.0, 100.0, 500.0]
LS = {0.0: '--', 10.0: '-.', 100.0: (0, (3, 1, 1, 1)), 500.0: ':'}

delays_us = np.logspace(-3, 2.3, 350)

def rates(r, T_us, thr_keV):
    """Capture and counted-scatter rate [Hz @100% duty] for ToF > T."""
    m = r['tof_us'] > T_us
    n = len(r['tof_us'])
    Rc = r['rate_surface'] * r['w_capture'][m].sum() / n
    if thr_keV > 0:
        m = m & (r['E_surv'] >= thr_keV * 1e3)
    Rs = r['rate_surface'] * r['w_scatter'][m].sum() / n
    return Rc, Rs

# ── figure: one panel per sample ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

for ax, (key, s) in zip(axes, SAMPLES.items()):
    r = RES[key]
    Rc0 = rates(r, 0.0, 0.0)[0]
    eff = np.array([rates(r, T, 0.0)[0] for T in delays_us]) / Rc0
    ax.semilogx(delays_us, 100 * eff, color='k', lw=2.6, label='capture efficiency')

    for thr in THRESHOLDS_KEV:
        pur = []
        for T in delays_us:
            Rc, Rs = rates(r, T, thr)
            pur.append(Rc / (Rc + Rs) if (Rc + Rs) > 0 else np.nan)
        lab = ('purity, all scatters' if thr == 0
               else f'purity, only $E_n$ > {thr:g} keV')
        ax.semilogx(delays_us, 100 * np.array(pur), color=s['color'], lw=1.9,
                    ls=LS[thr], label=lab)

    t_th = tof_constant_energy(g['E_THERMAL_EV'], r['distance_cm']) * 1e6
    ax.axvline(t_th, color='seagreen', ls=':', lw=1.2)
    ax.text(t_th * 0.93, 50, 'thermal ToF', rotation=90, fontsize=8,
            color='seagreen', ha='right', va='center')

    ax.set_title(f"Sample {key} — {r['distance_cm']:.0f} cm"
                 f"{', 0.5 cm steel' if s['steel_cm'] else ', air only'}"
                 f"   (capture {Rc0:.1f} Hz at 100% duty)", fontsize=11)
    ax.set_xlabel('Trigger delay (ToF cut)  [µs]', fontsize=12)
    ax.set_xlim(1e-3, 2e2)
    ax.set_ylim(0, 105)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='center left')

axes[0].set_ylabel('[%]', fontsize=12)

fig.suptitle('B-10 capture vs elastic-scatter background as a function of trigger delay\n'
             'a delay is an energy ceiling: $E_{max} = \\frac{1}{2}m(d/T)^2$  —  scatters '
             'counted only above a threshold on the neutron kinetic energy',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('pns_delay_scan.png', dpi=130)
print('\nSaved → pns_delay_scan.png')

# ── working-point table ──────────────────────────────────────────────────────
M_N_KG, EV_J = 1.675e-27, 1.602e-19
def E_of_delay(T_us, d_cm):
    return 0.5 * M_N_KG * ((d_cm * 1e-2) / (T_us * 1e-6)) ** 2 / EV_J

for key, s in SAMPLES.items():
    r = RES[key]
    Rc0 = rates(r, 0.0, 0.0)[0]
    print(f"\n=== Sample {key} ({r['distance_cm']:.0f} cm) — rates at 100% duty")
    head = f"{'delay[µs]':<10}{'E_max':>11}{'cap[Hz]':>9}{'eff[%]':>8}"
    for thr in THRESHOLDS_KEV:
        head += f"{('scat>' + (f'{thr:g}keV' if thr else 'all')):>13}{'pur[%]':>8}"
    print(head); print('-' * len(head))
    for T in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        Rc = rates(r, T, 0.0)[0]
        Emax = f'{E_of_delay(T, r["distance_cm"]):.3g} eV' if T > 0 else '-'
        row = f'{T:<10.3g}{Emax:>11}{Rc:>9.2f}{100 * Rc / Rc0:>8.1f}'
        for thr in THRESHOLDS_KEV:
            Rs = rates(r, T, thr)[1]
            row += f'{Rs:>13.2f}{100 * Rc / (Rc + Rs):>8.1f}'
        print(row)

print('\n(multiply rates by the duty cycle eta for pulsed running)')
