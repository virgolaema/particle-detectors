"""Energy spectrum of the neutrons that actually get through the shield.

Left  : spectrum per unit lethargy, dN/dln(E), normalised to unit area — the
        correct way to read a spectrum on a log-E axis (equal areas = equal
        numbers of neutrons).
Right : cumulative fraction, so medians and the fraction above/below any energy
        can be read directly.

Both panels use only the MC-transmitted population (the survivors), for
Sample A (20 cm, air) and Sample B (30 cm, behind 0.5 cm steel).
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

nb = json.load(open('pns.ipynb'))
g = {'__name__': '__main__'}
for i in range(0, 5):                      # cells 0-4: MC + fluxes
    src = ''.join(nb['cells'][i]['source'])
    if src.strip():
        exec(compile(src, f'<cell {i}>', 'exec'), g)

np = g['np']
RES, SAMPLES = g['RES'], g['SAMPLES']
E_TH = g['E_THERMAL_EV']

bins = np.logspace(np.log10(0.02), np.log10(4e6), 90)
bc = np.sqrt(bins[:-1] * bins[1:])
dlnE = np.diff(np.log(bins))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

info = {}
for key, s in SAMPLES.items():
    r = RES[key]
    E = r['E_surv']
    n = len(E)
    counts, _ = np.histogram(E, bins=bins)
    dens = counts / n / dlnE                      # dN/dlnE, unit area

    ax1.step(bc, dens, where='mid', color=s['color'], lw=2.2,
             label=f"{key} ({r['distance_cm']:.0f} cm): {n:,} survivors, "
                   f"transm. {100*r['transmission']:.2f}%")

    order = np.sort(E)
    cum = np.arange(1, n + 1) / n
    ax2.semilogx(order, 100 * cum, color=s['color'], lw=2.2, label=f'{key}')

    info[key] = dict(
        thermal=100 * np.mean(E < 1.0),
        epi=100 * np.mean((E >= 1.0) & (E < 1e5)),
        fast=100 * np.mean(E >= 1e5),
        floor=100 * np.mean(E <= E_TH * 1.001),
        median=np.median(E), mean=np.mean(E),
        ncoll=np.mean(r['n_coll'][r['transmitted']]))

for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.set_xlim(0.02, 4e6)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel('Neutron energy at the shield exit  [eV]', fontsize=12)
    for E_b, lab in [(1.0, '1 eV'), (1e5, '100 keV')]:
        ax.axvline(E_b, color='gray', ls=':', lw=1.0)
    ax.axvline(E_TH, color='seagreen', ls=':', lw=1.2)
    ax.axvline(g['E0_MEV'] * 1e6, color='crimson', ls='--', lw=1.2)

ax1.set_yscale('log')
ax1.set_ylabel('dN / dln E   (unit area)', fontsize=12)
ax1.set_title('Spectrum of the surviving neutrons (per unit lethargy)', fontsize=12)
ax1.set_ylim(1e-4, 1)
ax1.text(g['E0_MEV'] * 1e6 * 0.8, 3e-4, 'source 2.45 MeV', fontsize=8, color='crimson',
         rotation=90, ha='right', va='bottom')
ax1.text(E_TH * 1.15, 3e-4, 'thermal floor (MC cut-off)', fontsize=8, color='seagreen',
         rotation=90, ha='left', va='bottom')
ax1.legend(fontsize=9, loc='upper left')

ax2.set_ylabel('Cumulative fraction of survivors  [%]', fontsize=12)
ax2.set_title('Cumulative distribution', fontsize=12)
ax2.set_ylim(0, 100)
txt = '\n'.join(
    f"{k}:  thermal <1 eV {info[k]['thermal']:.0f}%   "
    f"1 eV–100 keV {info[k]['epi']:.0f}%   >100 keV {info[k]['fast']:.0f}%\n"
    f"      median {info[k]['median']:.2g} eV,  mean {info[k]['ncoll']:.1f} collisions"
    for k in SAMPLES)
ax2.text(0.03, 0.97, txt, transform=ax2.transAxes, fontsize=8.5, va='top',
         bbox=dict(fc='white', ec='0.7', alpha=0.9))
ax2.legend(fontsize=9, loc='lower right')

fig.suptitle('Energy spectrum of the neutrons that survive the shield '
             '(14 cm B-HDPE + 1 cm wood; B additionally behind 0.5 cm steel)\n'
             'the spike at the thermal floor is an artefact of the 25.3 meV '
             'cut-off in the MC — no S(α,β) treatment',
             fontsize=11, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('pns_transmitted_spectrum.png', dpi=130)
print('Saved → pns_transmitted_spectrum.png\n')

for k, v in info.items():
    print(f"Sample {k}: thermal {v['thermal']:.1f}%  epi {v['epi']:.1f}%  "
          f"fast {v['fast']:.1f}%  | at the 25.3 meV floor {v['floor']:.1f}%  "
          f"| median {v['median']:.3g} eV  mean {v['mean']:.3g} eV  "
          f"| {v['ncoll']:.1f} collisions")
