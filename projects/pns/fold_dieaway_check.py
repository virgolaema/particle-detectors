"""Reconcile the TOF-capture picture with the MEASURED post-gate die-away.

Observable = capture time after gate rise, folded at the 50 us period:
  emission (uniform in 1 us gate) + shield transit (MC clock) + ballistic air
  drift (gap / v_exit) + in-tile thermal dwell (exponential, lambda ~ 2.8 us,
  the tile's own die-away from notebook cell 8).

The knob is the shield->tile DISTANCE: a tile at the shield face sees a prompt
supply and the observed lambda is the tile dwell (~3-4 us, as measured); a
15 cm standoff (Sample B plan) stretches arrivals into a ~15-20 us effective
slope plus a wrapped thermal bump -> flatter pedestal.  Uses the fixed-physics
_mc_state.pkl (RES A = no steel, RES B = 0.5 cm steel).
"""
import os, pickle, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib_neutrons import get_endf_cross_sections

st = pickle.load(open('_mc_state.pkl', 'rb'))
RES = st['RES']
n_H_scint, n_C_scint = st['n_H_scint'], st['n_C_scint']
SCINT_L = st['SCINT_THICKNESS_CM']
E_TH = st['E_THERMAL_EV']
n_B10_scint = 1.05 * 0.02 * 0.199 / 10.0 * 6.022e23

PERIOD_US, GATE_US, LAM_TILE_US = 50.0, 1.0, 2.8
MEAS_LAMBDA_US = 4.06
rng = np.random.default_rng(20260824)
R = 40

def p_capture_first(E):
    sH, sC = get_endf_cross_sections(np.maximum(E, 0.025))
    Sc = n_B10_scint * 3840.0 * np.sqrt(E_TH / np.maximum(E, E_TH)) * 1e-24
    Ss = (n_H_scint * sH + n_C_scint * sC) * 1e-24
    St = Sc + Ss
    return Sc / St * (1.0 - np.exp(-St * SCINT_L))

def folded(res, gap_cm):
    E, t_sh = res['E_surv'], res['t_shield'] * 1e6          # us
    tof = t_sh + gap_cm / (1.3831e6 * np.sqrt(np.maximum(E, 1e-6))) * 1e6
    w = p_capture_first(E)
    t_emit = rng.uniform(0.0, GATE_US, size=(R, E.size))
    t_dw = rng.exponential(LAM_TILE_US, size=(R, E.size))
    tau = (t_emit + tof[None, :] + t_dw) % PERIOD_US
    bins = np.arange(0, PERIOD_US + 0.25, 0.5)
    h = np.histogram(tau.ravel(), bins=bins,
                     weights=np.broadcast_to(w, (R, E.size)).ravel() / R)[0]
    return 0.5 * (bins[:-1] + bins[1:]), h

def model(t, A, lam, C):
    return A * np.exp(-t / lam) + C

CASES = [
    ('tile 2 cm from shield (bench-like)', RES['A'], 2.0, 'crimson'),
    ('Sample A: 5 cm air',                 RES['A'], 5.0, 'darkorange'),
    ('Sample B: 15 cm, 0.5 cm steel',      RES['B'], 14.5, 'steelblue'),
]

fig, ax = plt.subplots(figsize=(10.5, 6))
for label, res, gap, col in CASES:
    ctr, h = folded(res, gap)
    m = (ctr > 2.0) & (ctr < 20)
    popt, pcov = curve_fit(model, ctr[m], h[m],
                           p0=(h.max(), 4.0, max(h.min(), 1e-4)),
                           bounds=([0, 0.5, 0], [np.inf, 100, np.inf]), maxfev=40000)
    lam, lam_e = popt[1], np.sqrt(pcov[1, 1])
    ax.step(ctr, h / h.max(), where='mid', color=col, lw=1.7,
            label=f'{label}:  λ_eff = {lam:.1f} ± {lam_e:.1f} µs')
    print(f"{label:38s}: lambda_eff = {lam:5.2f} +- {lam_e:4.2f} us   "
          f"flat/peak = {popt[2]/h.max()*100:4.1f}%")

tt = np.linspace(1.0, 25, 200)
ax.plot(tt, np.exp(-(tt - 1.0) / MEAS_LAMBDA_US), 'k--', lw=1.6,
        label=f'measured die-away  λ = {MEAS_LAMBDA_US} µs')
ax.set_yscale('log'); ax.set_ylim(3e-3, 1.3); ax.set_xlim(0, 50)
ax.set_xlabel('time after gate rise (mod 50 µs)  [µs]', fontsize=11)
ax.set_ylabel('predicted B-10 captures  (peak-normalised)', fontsize=11)
ax.set_title('Folded MC capture-time prediction vs measured die-away — the shield→tile distance is the knob\n'
             f'emission (1 µs gate) ⊕ shield transit ⊕ air drift ⊕ tile dwell (λ={LAM_TILE_US} µs), folded at 50 µs',
             fontsize=10.5)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('sampleB_fold_dieaway.png', dpi=120)
print("Saved -> sampleB_fold_dieaway.png")
