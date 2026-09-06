# PNS boron-capture MC

Neutron and gamma Monte Carlo for the pulsed-neutron-source test of a
boron-loaded PVT tile behind a 14 cm borated-HDPE shield. Companion data
analysis: https://github.com/virgolaema/3det-pns-neutron-ana

Run everything from this directory with the repo's `energy-deposits-env`.

| script | what it does |
|---|---|
| `sampleB_mc3d.py` | 3-D neutron transport, real bench stack, ray-trace to the 3x3 cm tile; TOF/fates/spectrum figures. Loads `_mc3d_cache.npz` (10M neutrons) if present, else regenerates it (hours). |
| `sampleB_gamma.py` | capture vertices -> photon transport -> in-tile response -> predicted delayed charge spectrum and time profile, compared with the data (needs the analysis repo's feature caches). ~20 s. |
| `h_hydrogen.py` | the hydrogen accounting behind `HYDROGEN.md` |
| `composite_timing.py`, `fold_dieaway_check.py` | shield-capture clock folded with the gate vs the measured die-away |
| `sampleB_fixed_mc.py` | earlier 1-D stack MC (kept for the H-capture fix history) |
| `cube_coincidence.py`, `scint_hcapture.py`, `tof_*.py`, `transmitted_spectrum.py` | side studies |
| `checks/` | one-off cross-check scripts referenced from `HYDROGEN.md` |

Documents: `NOTES.md` (dated working notes), `HYDROGEN.md` (hydrogen review),
`GAMMA_MC.md` (gamma-level MC and its verdicts). Caches `_*.npz` / `_*.pkl`
are not tracked.
