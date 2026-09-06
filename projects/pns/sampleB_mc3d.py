"""3-D shield MC for Sample B: full position+direction tracking, ray-traced to
the real 3x3 cm tile at 30 cm.  Fixes the 1-D MC's implicit assumption that
every transmitted neutron reaches the tile: forward fast punch-through mostly
does, diffuse (moderated) exits mostly miss.

Produces sampleB_tof_interactions_3d.png: TOF flux AT THE TILE overlaid with
flux x p(visible scatter, deposit>50 keV) and flux x p(B-10 capture first).
"""
import math, os, sys, time as time_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib_neutrons import get_endf_cross_sections

# ── geometry / materials (identical to sampleB_fixed_mc) ─────────────────────
E0_eV, N_A = 2.45e6, 6.022e23
# REAL bench geometry (user, 2026-08-24): 14 cm B-HDPE + 1 cm wood + 5 cm air
# + 1 cm steel + 2 cm air -> tile at 23 cm.  Air layers are tracked (zero
# cross-section) so the slow-drift TIME is in the clock automatically.
PE_CM, WOOD_CM, AIR1_CM, STEEL_CM, AIR2_CM = 14.0, 1.0, 5.0, 1.0, 2.0
D_TILE, HALF_TILE = 23.0, 1.5           # tile plane z=23 cm, 3x3 cm
n_B10 = 1.0 * 0.05 * 0.199 / 10.0 * N_A
n_H_bh, n_C_bh = 1.0 * 0.143 * N_A, 1.0 * 0.807 / 12.0 * N_A
n_H_wd = 0.6 * 0.06 * N_A
n_Ceq_wd = (0.6 * 0.50 / 12.0 + 0.6 * 0.44 / 16.0) * N_A
n_Fe = 7.85 / 55.85 * N_A
S_B10_TH, S_HG_TH, S_FE_TH, E_TH = 3840.0, 0.332, 2.56, 0.0253
n_B10_scint = 1.05 * 0.02 * 0.199 / 10.0 * N_A
n_H_sc, n_C_sc = 5.2514e22, 4.7262e22

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

def mk_air(th, z0):
    return dict(name='air', z0=z0, z1=z0 + th, A=1.0, absorbers=[],
                Ss=np.zeros_like(_E), fH=np.zeros_like(_E))

LAY = [mk('bhdpe', PE_CM, 0.0, n_H_bh, n_C_bh, _sC, 12.0,
          [('B10', n_B10, S_B10_TH), ('H', n_H_bh, S_HG_TH)])]
LAY.append(mk('wood', WOOD_CM, LAY[-1]['z1'], n_H_wd, n_Ceq_wd, _sC, 12.0,
              [('H', n_H_wd, S_HG_TH)]))
LAY.append(mk_air(AIR1_CM, LAY[-1]['z1']))
LAY.append(mk('steel', STEEL_CM, LAY[-1]['z1'], 0.0, n_Fe, _sFe, 55.85,
              [('Fe', n_Fe, S_FE_TH)]))
LAY.append(mk_air(AIR2_CM, LAY[-1]['z1']))
ZX = LAY[-1]['z1']                      # = 30 cm = the tile plane itself

rand = np.random.random

def vel(E): return 1.3831e6 * math.sqrt(max(E, 1e-6))

def rotate(ux, uy, uz, c):
    """Rotate unit vector u by lab angle acos(c), random azimuth."""
    s = math.sqrt(max(1.0 - c * c, 0.0))
    phi = 2.0 * math.pi * rand()
    cp, sp = math.cos(phi), math.sin(phi)
    if abs(uz) < 0.999999:
        den = math.sqrt(1.0 - uz * uz)
        ax, ay, az = uy / den, -ux / den, 0.0
        bx, by, bz = uz * ux / den, uz * uy / den, -den
    else:
        ax, ay, az = 1.0, 0.0, 0.0
        bx, by, bz = 0.0, uz, 0.0
    return (c * ux + s * (cp * ax + sp * bx),
            c * uy + s * (cp * ay + sp * by),
            c * uz + s * (cp * az + sp * bz))

def run(N):
    MU0 = math.cos(math.radians(20.0))
    out = []                                 # per transmitted: E, t_stack, x,y,z_exit, u
    fates = dict(tran=0, back=0, B10=0, H=0, Fe=0)
    t0 = time_module.time()
    for i in range(N):
        E, t = E0_eV, 0.0
        x = y = z = 0.0
        mu = MU0 + (1 - MU0) * rand()
        phi = 2.0 * math.pi * rand()
        sn = math.sqrt(1 - mu * mu)
        ux, uy, uz = sn * math.cos(phi), sn * math.sin(phi), mu
        while True:
            lay = None
            for l in LAY:
                if z < l['z1'] - 1e-12:
                    lay = l; break
            if lay is None:
                lay = LAY[-1]
            lgE = math.log(max(E, 0.025))
            Ss = float(np.interp(lgE, _lE, lay['Ss']))
            iv = math.sqrt(E_TH / max(E, E_TH))
            Sa_l = [(tag, n * s * iv * 1e-24) for tag, n, s in lay['absorbers']]
            Sa = sum(sv for _, sv in Sa_l)
            St = Ss + Sa
            step = -math.log(rand()) / St if St > 1e-12 else 1e30
            df = (lay['z1'] - z) / uz if uz > 0 else (z - lay['z0']) / (-uz) if uz < 0 else 1e30
            if step >= df:
                t += df / vel(E)
                x += df * ux; y += df * uy; z += df * uz + 1e-9 * math.copysign(1, uz)
                if z >= ZX:
                    fates['tran'] += 1
                    out.append((E, t, x, y, ux, uy, uz))
                    break
                if z <= 0.0:
                    fates['back'] += 1; break
                continue
            t += step / vel(E)
            x += step * ux; y += step * uy; z += step * uz
            if rand() * St < Sa:
                r = rand() * Sa
                tag = Sa_l[-1][0]
                for tg, sv in Sa_l:
                    r -= sv
                    if r <= 0: tag = tg; break
                fates[tag] += 1; break
            if rand() < float(np.interp(lgE, _lE, lay['fH'])):
                c = math.sqrt(rand())
                E = max(E * c * c, 0.025)
                ux, uy, uz = rotate(ux, uy, uz, c)
            else:
                A = lay['A']
                cc = 2 * rand() - 1
                dn = 1 + 2 * A * cc + A * A
                E = max(E * dn / (A + 1) ** 2, 0.025)
                ux, uy, uz = rotate(ux, uy, uz, (1 + A * cc) / math.sqrt(dn))
    print(f"3-D MC: {N:,} n in {time_module.time()-t0:.0f}s   fates%: "
          + "  ".join(f"{k} {100*v/N:.2f}" for k, v in fates.items()))
    return out, fates

np.random.seed(20260824)
N = 10_000_000
CACHE = '_mc3d_cache.npz'
if os.path.exists(CACHE):
    _c = np.load(CACHE, allow_pickle=True)
    arr = _c['arr']; fates = _c['fates'].item(); N = int(_c['N'])
    print(f"loaded cache: {arr.shape[0]:,} transmitted of {N:,}")
else:
    res, fates = run(N)
    arr = np.array(res)
    np.savez(CACHE, arr=arr, fates=fates, N=N)

E_t, t_st = arr[:, 0], arr[:, 1]
X, Y = arr[:, 2], arr[:, 3]
UX, UY, UZ = arr[:, 4], arr[:, 5], arr[:, 6]

# ray-trace to the tile plane
L = (D_TILE - ZX) / UZ
xt, yt = X + UX * L, Y + UY * L
hit = (np.abs(xt) <= HALF_TILE) & (np.abs(yt) <= HALF_TILE) & (UZ > 0)
t_air = L / (1.3831e6 * np.sqrt(np.maximum(E_t, 1e-6)))
tof_us = (t_st + t_air) * 1e6

# acceptance by energy group
print(f"transmitted {arr.shape[0]:,}   hit tile {hit.sum():,}  ({100*hit.mean():.2f}%)")
for lo, hi, lbl in [(1e5, 4e6, 'fast   (>100 keV)'), (1.0, 1e5, 'epithermal'), (0, 1.0, 'thermal (<1 eV)')]:
    m = (E_t >= lo) & (E_t < hi)
    if m.sum():
        print(f"  {lbl:20s}: {100*hit[m].mean():6.2f}% acceptance   "
              f"({m.sum():6d} transmitted -> {hit[m].sum():4d} at tile)")

# tile first-interaction probabilities for the HITS
Eh, th = E_t[hit], tof_us[hit]
sH_s, sC_s = get_endf_cross_sections(np.maximum(Eh, 0.025))
Sig_c = n_B10_scint * 3840.0 * np.sqrt(E_TH / np.maximum(Eh, E_TH)) * 1e-24
Sig_sH = n_H_sc * sH_s * 1e-24
Sig_sC = n_C_sc * sC_s * 1e-24
Sig_t = Sig_c + Sig_sH + Sig_sC
p_int = 1.0 - np.exp(-Sig_t * 1.0)
p_cap = Sig_c / Sig_t * p_int
E_THR = 5e4
f_H = np.clip(1.0 - E_THR / np.maximum(Eh, 1e-3), 0, 1)
f_C = np.clip(1.0 - E_THR / (0.284 * np.maximum(Eh, 1e-3)), 0, 1)
p_vis = (Sig_sH * f_H + Sig_sC * f_C) / Sig_t * p_int

GREEN_C, RED_S, BLUE = 'seagreen', 'crimson', 'steelblue'
tb = np.logspace(np.log10(max(th.min() * 0.8, 5e-3)), np.log10(max(th.max() * 1.3, 200)), 50)
ctr = np.sqrt(tb[:-1] * tb[1:])
h_all = np.histogram(th, bins=tb)[0]
h_cap = np.histogram(th, bins=tb, weights=p_cap)[0]
h_vis = np.histogram(th, bins=tb, weights=p_vis)[0]

fig, ax = plt.subplots(figsize=(10.5, 6))
ax.hist(th, bins=tb, color=BLUE, alpha=0.30, label=f'arrivals AT THE TILE ({hit.sum():,} of {N:,} emitted)')
ax.step(ctr, h_vis, where='mid', color=RED_S, lw=1.8,
        label='flux × p(scatter first) — recoil deposit >50 keV')
ax.step(ctr, h_cap, where='mid', color=GREEN_C, lw=1.8,
        label='flux × p(B-10 capture first)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.05, None)
ax.set_xlabel('Time of flight, generation → Sample B tile  [µs]', fontsize=11)
ax.set_ylabel('Neutrons / bin', fontsize=11)
ax.grid(True, which='both', alpha=0.25)
ax.axvline(0.5, color='mediumblue', lw=3)
ax.text(0.5, ax.get_ylim()[1] * 0.5, 'TOF cut\nat 500ns  ', ha='right',
        fontsize=12, color='mediumblue', fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_title('3-D MC, real bench geometry (14 B-HDPE / 1 wood / 5 air / 1 steel / 2 air, tile at 23 cm)\n'
             'curves overlaid, not stacked', fontsize=10.5)
plt.tight_layout()
plt.savefig('sampleB_tof_interactions_3d.png', dpi=120)
print(f"expected tile events per {N:,} emitted: capture {h_cap.sum():.1f} | visible scatter {h_vis.sum():.1f}")
print("Saved -> sampleB_tof_interactions_3d.png")

# ── same figure, linear vertical scale (two panels: full, and zoomed so the
#    interaction curves are visible against the arrival peak) ────────────────
figl, axl = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for ax_, ymax in ((axl[0], None), (axl[1], 250)):
    ax_.hist(th, bins=tb, color=BLUE, alpha=0.30,
             label=f'arrivals AT THE TILE ({hit.sum():,} of {N:,} emitted)')
    ax_.step(ctr, h_vis, where='mid', color=RED_S, lw=2.2,
             label='flux × p(scatter first) — recoil deposit >50 keV')
    ax_.step(ctr, h_cap, where='mid', color=GREEN_C, lw=2.2,
             label='flux × p(B-10 capture first)')
    ax_.axvline(0.5, color='mediumblue', lw=3)
    ax_.set_xscale('log')
    if ymax:
        ax_.set_ylim(0, ymax)
    ax_.set_ylabel('Neutrons / bin', fontsize=13)
    ax_.tick_params(labelsize=12)
    ax_.grid(True, which='both', alpha=0.25)
axl[0].legend(fontsize=12, loc='upper right')
axl[0].text(0.5, axl[0].get_ylim()[1] * 0.55, 'TOF cut\nat 500ns  ', ha='right',
            fontsize=13, color='mediumblue', fontweight='bold')
axl[0].set_title('linear scale — full range', fontsize=12)
axl[1].set_title('linear scale — zoomed to the interaction curves', fontsize=12)
axl[1].set_xlabel('Time of flight, generation → Sample B tile  [µs]', fontsize=13)
plt.tight_layout()
plt.savefig('sampleB_tof_interactions_3d_linear.png', dpi=120)
print("Saved -> sampleB_tof_interactions_3d_linear.png")

# ── scatter and capture each on its own linear panel, side by side ─────────
figs, axs_ = plt.subplots(1, 2, figsize=(15, 6))
for ax_, hh, col, ttl, lab in (
        (axs_[0], h_vis, RED_S, 'flux × p(scatter first) — recoil deposit >50 keV',
         f'{h_vis.sum():.0f} visible scatters per {N:,} emitted'),
        (axs_[1], h_cap, GREEN_C, 'flux × p(B-10 capture first)',
         f'{h_cap.sum():.1f} captures per {N:,} emitted')):
    ax_.fill_between(ctr, hh, step='mid', color=col, alpha=0.25)
    ax_.step(ctr, hh, where='mid', color=col, lw=2.4, label=lab)
    ax_.axvline(0.5, color='mediumblue', lw=3)
    ax_.set_xscale('log'); ax_.set_ylim(0, None)
    ax_.text(0.5, ax_.get_ylim()[1] * 0.70, 'TOF cut\nat 500ns  ', ha='right',
             fontsize=13, color='mediumblue', fontweight='bold')
    ax_.set_xlabel('Time of flight, generation → Sample B tile  [µs]', fontsize=13)
    ax_.set_ylabel('Expected tile events / bin  (linear)', fontsize=13)
    ax_.set_title(ttl, fontsize=13)
    ax_.tick_params(labelsize=12)
    ax_.grid(True, which='both', alpha=0.25); ax_.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('sampleB_tof_interactions_3d_split.png', dpi=120)
print("Saved -> sampleB_tof_interactions_3d_split.png")

# ── fates figure incl. tile acceptance ───────────────────────────────────────
n_hit = int(hit.sum())
cats = ['B-10 capture\n(shield)', 'Backscattered', 'Transmitted,\nmiss tile',
        'Reach tile', 'H capture\n(shield+wood)', 'Fe capture\n(steel)']
vals = [100 * fates['B10'] / N, 100 * fates['back'] / N,
        100 * (fates['tran'] - n_hit) / N, 100 * n_hit / N,
        100 * fates['H'] / N, 100 * fates['Fe'] / N]
fig2, axf = plt.subplots(figsize=(10, 5.5))
bars = axf.bar(np.arange(len(cats)), vals, 0.62, color='steelblue')
bars[3].set_color('seagreen')
for rect, v in zip(bars, vals):
    axf.text(rect.get_x() + rect.get_width() / 2, v * 1.18, f'{v:.3g}%',
             ha='center', fontsize=11)
axf.set_yscale('log'); axf.set_ylim(5e-3, 300)
axf.set_xticks(np.arange(len(cats))); axf.set_xticklabels(cats, fontsize=10)
axf.set_ylabel('Fraction of emitted neutrons  [%]  (log)', fontsize=11)
axf.set_title(f'Neutron fates, real bench geometry — {N:,} neutrons at 2.45 MeV into the 20° cone\n'
              f'tile acceptance: {100 * n_hit / max(fates["tran"], 1):.1f}% of transmitted reach the 3×3 cm tile',
              fontsize=11)
axf.grid(True, axis='y', which='both', alpha=0.25)
plt.tight_layout()
plt.savefig('sampleB_fates_3d.png', dpi=120)
print("Saved -> sampleB_fates_3d.png")
