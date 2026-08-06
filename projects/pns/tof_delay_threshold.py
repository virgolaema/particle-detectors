"""Same delay scan, but asking which scatters are actually *visible*.

An elastic scatter only produces a signal if the recoil is above the detector
threshold.  For H the recoil energy is uniform in [0, E_n], mean E_n/2, and the
scintillation light of a proton is quenched (Birks): a 100 keV proton gives
roughly 10 keVee.  Sub-keV epithermal/thermal neutrons therefore scatter often
but deposit nothing detectable.

Prints capture/scatter rates and purity for a few neutron-energy thresholds
applied to the scattering neutron.
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

for key in SAMPLES:
    r = RES[key]
    E = r['E_surv']
    sH, sC = get_endf_cross_sections(E)
    Sig_s = (g['n_H_scint'] * sH + g['n_C_scint'] * sC) * 1e-24
    r['w_scatter'] = 1.0 - np.exp(-Sig_s * t_cm)
    r['w_capture'] = r['p_interact']
    r['tof_us'] = tof_constant_energy(E, r['distance_cm']) * 1e6

# energy corresponding to a given delay (free flight, non-relativistic)
M_N_KG, EV_J = 1.675e-27, 1.602e-19
def E_of_delay(T_us, d_cm):
    v = (d_cm * 1e-2) / (T_us * 1e-6)
    return 0.5 * M_N_KG * v**2 / EV_J

DELAYS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
THRESH = [0.0, 1e4, 1e5, 5e5]     # neutron-energy threshold for a visible recoil

for key, s in SAMPLES.items():
    r = RES[key]
    n = len(r['tof_us'])
    print(f"\n=== Sample {key}  ({r['distance_cm']:.0f} cm) "
          f"— rates at 100% duty, purity = cap/(cap+visible scatter)")
    head = f"{'delay[µs]':<10}{'E(delay)':>10}{'cap[Hz]':>9}"
    for th in THRESH:
        lab = 'all' if th == 0 else f'>{th/1e3:g}keV'
        head += f"{'scat ' + lab:>14}{'pur[%]':>8}"
    print(head)
    print('-' * len(head))
    for T in DELAYS:
        m = r['tof_us'] > T
        Rc = r['rate_surface'] * r['w_capture'][m].sum() / n
        Ed = E_of_delay(T, r['distance_cm']) if T > 0 else np.inf
        row = f"{T:<10.3g}{(f'{Ed:.3g} eV' if T > 0 else '-'):>10}{Rc:>9.2f}"
        for th in THRESH:
            mm = m & (r['E_surv'] >= th)
            Rs = r['rate_surface'] * r['w_scatter'][mm].sum() / n
            row += f"{Rs:>14.2f}{100 * Rc / (Rc + Rs):>8.1f}"
        print(row)
