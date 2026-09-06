"""Composite capture-timing model vs the 20260731 period-500us data.

Model of the post-gate rate in the tile:
  R(t) = A_g * g_shield(t)          shield-capture GAMMAS: 478 keV emitted at the
                                    (MC-timestamped) capture time inside the
                                    B-HDPE, attenuation-weighted, arriving at
                                    light speed -> Compton events in the tile
       + A_t * tile_dwell(t)        tile captures fed promptly by the fast flux
                                    moderating locally: exp dwell, lambda free
       + C                          ambient / room-return flat
both signal terms convolved with the 0.5 us gate emission window.

Fits the 31 July tau histogram (0-45 us) and reports component fractions.
"""
import os, pickle, sys, time as time_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib_neutrons import get_endf_cross_sections

# ── shield MC (identical physics to sampleB_fixed_mc), recording capture t,z ──
E0_eV = 2.45e6
N_A = 6.022e23
PE_CM, WOOD_CM, STEEL_CM = 14.0, 1.0, 0.5
n_B10 = 1.0 * 0.05 * 0.199 / 10.0 * N_A
n_H_bh, n_C_bh = 1.0 * 0.143 * N_A, 1.0 * 0.807 / 12.0 * N_A
n_H_wd = 0.6 * 0.06 * N_A
n_Ceq_wd = (0.6 * 0.50 / 12.0 + 0.6 * 0.44 / 16.0) * N_A
n_Fe = 7.85 / 55.85 * N_A
S_B10_TH, S_HG_TH, S_FE_TH, E_TH = 3840.0, 0.332, 2.56, 0.0253

_FE_E = np.array([1e-2, 1e0, 1e2, 1e3, 1e4, 3e4, 5e4, 1e5, 3e5, 5e5, 1e6, 2.45e6, 5e6])
_FE_S = np.array([11.4, 11.3, 11.2, 11.1, 11.0, 10.0, 8.0, 6.5, 4.2, 3.8, 3.2, 2.9, 2.5])
_E = np.logspace(np.log10(0.025), np.log10(E0_eV * 1.5), 2000)
_sH, _sC = get_endf_cross_sections(_E)
_sFe = 10 ** np.interp(np.log10(_E), np.log10(_FE_E), np.log10(_FE_S))
_lE = np.log(_E)

def mk(name, th, z0, nH, nhv, shv, A, absorbers):
    Ss = (nH * _sH + nhv * shv) * 1e-24
    fH = np.divide(nH * _sH * 1e-24, Ss, out=np.zeros_like(Ss), where=Ss > 0)
    return dict(name=name, z0=z0, z1=z0 + th, A=A, absorbers=absorbers, Ss=Ss, fH=fH)

LAY = [mk('bhdpe', PE_CM, 0.0, n_H_bh, n_C_bh, _sC, 12.0,
          [('B10', n_B10, S_B10_TH), ('H', n_H_bh, S_HG_TH)])]
LAY.append(mk('wood', WOOD_CM, LAY[-1]['z1'], n_H_wd, n_Ceq_wd, _sC, 12.0,
              [('H', n_H_wd, S_HG_TH)]))
LAY.append(mk('steel', STEEL_CM, LAY[-1]['z1'], 0.0, n_Fe, _sFe, 55.85,
              [('Fe', n_Fe, S_FE_TH)]))
ZX = LAY[-1]['z1']

def vel(E): return 1.3831e6 * np.sqrt(max(E, 1e-6))

def upmu(mu, ct):
    st = np.sqrt(max(1 - ct * ct, 0.0)); sm = np.sqrt(max(1 - mu * mu, 0.0))
    if sm < 1e-10: return ct if mu > 0 else -ct
    return mu * ct + sm * st * (2 * np.random.random() - 1)

CACHE = '_shield_capture_times.npz'
if os.path.exists(CACHE):
    z = np.load(CACHE)
    t_cap, z_cap = z['t_cap'], z['z_cap']
    print(f"loaded cache: {t_cap.size} shield B-10 captures")
else:
    np.random.seed(20260824)
    N = 100_000
    MU0 = np.cos(np.radians(20.0))
    t_cap, z_cap = [], []
    t0 = time_module.time()
    for i in range(N):
        E, pos, t = E0_eV, 0.0, 0.0
        mu = MU0 + (1 - MU0) * np.random.random()
        while True:
            lay = next(l for l in LAY if pos < l['z1'] - 1e-12) if pos < ZX else LAY[-1]
            lgE = np.log(max(E, 0.025))
            Ss = float(np.interp(lgE, _lE, lay['Ss']))
            iv = np.sqrt(E_TH / max(E, E_TH))
            Sa_l = [(tag, n * s * iv * 1e-24) for tag, n, s in lay['absorbers']]
            Sa = sum(s for _, s in Sa_l)
            St = Ss + Sa
            step = -np.log(np.random.random()) / St
            df = (lay['z1'] - pos) / mu if mu > 0 else (pos - lay['z0']) / (-mu) if mu < 0 else 1e30
            if step >= df:
                t += df / vel(E); pos += df * mu + 1e-9 * np.sign(mu)
                if pos >= ZX or pos <= 0: break
                continue
            t += step / vel(E); pos += step * mu
            if np.random.random() * St < Sa:
                r = np.random.random() * Sa
                for tag, s in Sa_l:
                    r -= s
                    if r <= 0:
                        if tag == 'B10': t_cap.append(t); z_cap.append(pos)
                        break
                break
            if np.random.random() < float(np.interp(lgE, _lE, lay['fH'])):
                c = np.sqrt(np.random.random()); E = max(E * c * c, 0.025); mu = upmu(mu, c)
            else:
                A = lay['A']; cc = 2 * np.random.random() - 1
                dn = 1 + 2 * A * cc + A * A
                E = max(E * dn / (A + 1) ** 2, 0.025); mu = upmu(mu, (1 + A * cc) / np.sqrt(dn))
    t_cap, z_cap = np.array(t_cap), np.array(z_cap)
    np.savez(CACHE, t_cap=t_cap, z_cap=z_cap)
    print(f"shield MC: {t_cap.size} B-10 captures in {time_module.time()-t0:.0f}s")

# gamma weight: 478 keV attenuation through remaining B-HDPE (KN mu = 0.098/cm)
w_g = np.exp(-0.098 * np.maximum(PE_CM - z_cap, 0.0))
tc_us = t_cap * 1e6

# ── data: tau histogram of the 20260731 period-500us run ─────────────────────
DCACHE = '_tau_20260731.npz'
if os.path.exists(DCACHE):
    taus = np.load(DCACHE)['taus']
    print(f"loaded data cache: {taus.size} taus")
else:
    sys.path.insert(0, '/Users/virgolaema/Software/3det/pns-waveform-ana')
    from pnsana import io as pio, features as pft
    from pnsana.config import Config as PCfg, CH_TRIGGER as PC1
    run = '/Users/virgolaema/Software/3det/data/20260731_sample02_gate0.5us_period500us_triggerBoron_thr36mV'
    cfg = PCfg(data_dir=run)
    taus = []
    for ev in pio.list_events(cfg, channels=(PC1,)):
        try:
            tr = pio.load_event(int(ev), cfg, channels=(PC1,))['trigger']
        except Exception:
            continue
        g = pft.find_gate(tr.t_ns, tr.v, cfg)
        if g.present and np.isfinite(g.rise_ns) and not g.clipped_start:
            tau = -g.rise_ns / 1000.0
            if 0 <= tau <= 180:
                taus.append(tau)
    taus = np.array(taus)
    np.savez(DCACHE, taus=taus)
    print(f"data: {taus.size} taus extracted")

BW = 1.0
bins = np.arange(0, 45 + BW, BW)
ctr = 0.5 * (bins[:-1] + bins[1:])
h_data = np.histogram(taus, bins=bins)[0].astype(float)
err = np.sqrt(np.maximum(h_data, 1.0))

# ── model components on the same binning (gate emission uniform in 0.5 us) ───
rng = np.random.default_rng(7)
R = 30
emit = rng.uniform(0, 0.5, size=(R, tc_us.size))
g_smp = (tc_us[None, :] + emit).ravel()
g_w = np.broadcast_to(w_g, (R, tc_us.size)).ravel()
g_hist = np.histogram(g_smp, bins=bins, weights=g_w)[0]
g_shape = g_hist / g_hist.sum()

def tile_shape(lam):
    tt = np.linspace(0, 0.5, 6)
    s = np.zeros_like(ctr)
    for t0v in tt:
        s += np.exp(-np.maximum(ctr - t0v, 0) / lam) * (ctr >= t0v)
    return s / s.sum()

def model(x, A_g, A_t, lam_t, C):
    return A_g * g_shape + A_t * tile_shape(lam_t) + C

# fit only tau > 1.5 us (prompt in-gate scatters contaminate below ~1 us,
# same reason report.py fits the die-away from 1 us)
fm = ctr > 1.5
x, y, e = ctr[fm], h_data[fm], err[fm]

def chi2_of(f, p):
    return float(np.sum(((y - f(x, *p)) / e) ** 2)), y.size - len(p)

# (a) shield gammas only
fa = lambda t, A, C: A * np.interp(t, ctr, g_shape) + C
pa, _ = curve_fit(fa, x, y, p0=(y.sum(), np.median(y[x > 35])), sigma=e, maxfev=20000)
c2a, nda = chi2_of(fa, pa)
# (b) tile dwell only
fb = lambda t, A, lam, C: A * np.interp(t, ctr, tile_shape(lam)) + C
pb, cb = curve_fit(fb, x, y, p0=(y.sum(), 4.0, np.median(y[x > 35])), sigma=e,
                   bounds=([0, 1, 0], [np.inf, 10, np.inf]), maxfev=20000)
c2b, ndb = chi2_of(fb, pb)
# (c) both
p0 = (y.sum() * 0.3, y.sum() * 0.3, 4.0, np.median(y[x > 35]))
popt, pcov = curve_fit(lambda t, Ag, At, lam, C: Ag * np.interp(t, ctr, g_shape)
                       + At * np.interp(t, ctr, tile_shape(lam)) + C,
                       x, y, p0=p0, sigma=e,
                       bounds=([0, 0, 1.0, 0], [np.inf, np.inf, 10.0, np.inf]), maxfev=40000)
A_g, A_t, lam_t, C = popt
perr = np.sqrt(np.diag(pcov))
c2c = float(np.sum(((y - (A_g * np.interp(x, ctr, g_shape)
                          + A_t * np.interp(x, ctr, tile_shape(lam_t)) + C)) / e) ** 2))

print(f"\nfits on tau > 1.5 us:")
print(f"(a) shield gammas only : chi2/ndf = {c2a:.1f}/{nda}")
print(f"(b) tile dwell only    : chi2/ndf = {c2b:.1f}/{ndb}   lambda = {pb[1]:.2f} +- {np.sqrt(cb[1,1]):.2f} us")
print(f"(c) both               : chi2/ndf = {c2c:.1f}/{y.size-4}   "
      f"Ag {A_g:.0f}+-{perr[0]:.0f}  At {A_t:.0f}+-{perr[1]:.0f}  lam {lam_t:.2f}+-{perr[2]:.2f}")
print(f"shield capture times: median {np.median(tc_us):.2f} us, mean {np.mean(tc_us):.2f} us")

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.errorbar(ctr, h_data, yerr=err, fmt='o', ms=3.5, color='k', lw=0.8, label='data: 20260731, period 500 µs')
tot = A_g * g_shape + A_t * tile_shape(lam_t) + C
ax.plot(ctr, tot, color='steelblue', lw=2, label=f'composite fit  (χ²/ndf = {c2c:.0f}/{y.size-4}, τ>1.5 µs)')
ax.plot(ctr, A_g * g_shape + C, color='darkorange', lw=1.5, ls='--',
        label=f'shield-capture gammas (MC timing)  [{A_g/(A_g+A_t)*100:.0f}% of excess]')
ax.plot(ctr, A_t * tile_shape(lam_t) + C, color='seagreen', lw=1.5, ls='--',
        label=f'tile captures, dwell λ = {lam_t:.2f} µs  [{A_t/(A_g+A_t)*100:.0f}%]')
ax.axhline(C, color='0.6', lw=1, ls=':', label='flat (ambient)')
ax.set_yscale('log'); ax.set_ylim(max(C * 0.3, 1), None)
ax.set_xlabel('boron trigger time since gate rise  [µs]', fontsize=11)
ax.set_ylabel(f'triggers / {BW:.0f} µs', fontsize=11)
ax.set_title('Composite MC timing model vs data — shield-capture gammas + tile dwell + flat', fontsize=11)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('composite_timing_fit.png', dpi=120)
print("Saved -> composite_timing_fit.png")
