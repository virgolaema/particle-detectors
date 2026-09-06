# PNS MC — working notes

Personal/working notes for the PNS neutron MC. Not part of the LaTeX note in
`../../../pns-3det-notes` — that document is kept in sync separately, and as of
2026-08-06 it is current with the notebook.

---

## 2026-08-18 — cross-cube coincidence + hydrogen capture channel

Three additions, each written as a standalone script (same replay-the-notebook
convention as `tof_delay_scan.py`) plus report figures and text.

**1. `cube_coincidence.py` — segmented 3x3x1 tile.** Photon MC of the 478 keV
capture gamma in a nine-cube (1 cm^3 each) tile; the alpha is point-like and
stays in its cube. Klein-Nishina Compton tracking. Key result: only **3.7 %** of
captures give a gamma deposit in a cube *other* than the alpha cube (3.2 % above
50 keV); **92 %** of gammas escape the tile entirely, because mfp = 10.2 cm >>
1 cm. Robust to capture profile (3.5 % front-biased). Figures
`cube_coincidence_outcomes.png`, `cube_coincidence_map.png`. New report section
`06_segmentation.tex`.
  - NB: the 478 keV attenuation is taken from Klein-Nishina directly
    (mu = 0.098/cm), NOT from `lib_photon`'s NIST polyethylene table, which is
    ~4x too low at 0.5 MeV (gives mu/rho 0.023 where KN and the water value put
    it at ~0.096 cm^2/g). Worth fixing in lib_photon eventually.

**2 + 3. `scint_hcapture.py` — H(n,gamma) in C8H8 and the interaction budget.**
Treats the plastic as pure C8H8 (n_H = n_C = 4.86e22 /cm^3). Figures:
  - `pns_hcapture_xsec.png`: sigma(E) for H(n,gamma) (1/v, 332 mb th), H and C
    elastic. H capture is ~250x below H elastic even at thermal.
  - `pns_tof_xsec_overlay.png`: arrival ToF spectrum with H-capture sigma
    overlaid — arrivals are fast (short ToF, low sigma); sigma only large in the
    slow tail.
  - `pns_interactions_by_energy.png`: arriving spectrum x single-pass p(E) per
    process, log-y. Elastic (fast-peaked) vs H(n,gamma) (thermal-peaked), ~10^3
    apart.
  - Numbers: single-pass H-capture chance per arriving neutron = **0.021 %**
    (both samples); elastic = 36 %. In the borated tile H(n,gamma) is ~1.6 % of
    captures (Sigma_a ratio B:H ~ 60:1) — a minor 2223 keV contamination.
  Added as subsection `\label{sec:hcapture}` in `04_scintillator.tex`.

Regenerate: run both scripts in `projects/pns/`, then
`cp pns_*.png cube_*.png ../../../pns-3det-notes/figures/` and rebuild the note
(`bash compile.sh`). The scripts read `_mc_state.pkl` (arriving spectrum, dumped
by `_dump_state.py`, replay to cell `1bc91a56`) or replay the notebook live.

---

## 2026-08-06 — the MC was answering the wrong question

Triggered by review feedback on why the MC predicted a delayed thermal bump
that the data does not show, and why the measured signal is an exponential
die-away instead.

### What was wrong

**1. The tile had no clock (the important one).**

The tile was modelled as a single-pass attenuator: for each arriving neutron,
`p(E) = 1 − exp(−Σ_B(E)·t)` evaluated at the *entrance* energy, with zero
elapsed time. So a neutron either captured instantly on arrival or never.

But the tile is borated PVT — a hydrogenous moderator with the absorber
dissolved in it. A neutron arriving epithermal does not capture on the way
through; it scatters, thermalizes locally, random-walks, and captures on the
tile's own timescale. The old model contained no mechanism that could produce
a delayed signal at all, so the ~3.5 µs die-away we actually measure was
unreachable *in principle*. This is the root cause: the model and the detector
were answering different questions.

**2. ToF was free-flight from the shield exit.**

`t = r / v(E_exit)` discards all the time spent slowing down inside the
shield. Now that time is accumulated (`Σ s/v` per step), it turns out the
median is unaffected (13 ns — the fast flux dominates) but the *mean* is
0.45 µs (A) / 1.14 µs (B) and the tail reaches 28 µs (A) / 73 µs (B). The
tail is exactly the slow population that could produce delayed signal, so the
old approximation was wrong precisely where it mattered.

**3. The H elastic cross section was badly wrong.** (found while fixing 1–2)

`lib_neutrons.get_endf_cross_sections` interpolated a sparse knot set with a
*cubic spline* in log-log space, which rings. It returned **5.19 b at 1 eV**
where the true H elastic cross section is ~20.5 b — a factor of 4 low, in the
epithermal region that drives moderation. Worse, the tabulated knots below
1 eV followed a 1/√E shape (that is an *absorber* curve, not elastic
scattering) reaching 164 b at 1 meV.

Fixed by: correct ENDF/B-VIII.0 knots, PCHIP (shape-preserving, no ringing)
instead of cubic spline, and an explicit bound-atom enhancement for H bound in
a molecular lattice (σ → 4× free-atom as E → 0, i.e. 82 b at thermal, the
standard value for H in polyethylene). New kwarg `bound_H=True` by default —
correct for every material in this package (HDPE, PVT, wood); pass `False`
for free hydrogen.

⚠️ **Blast radius:** this changes every neutron number in the package, not
just PNS. Other notebooks that call `get_endf_cross_sections` should be
re-run and their conclusions re-checked.

**4. §3 and §4 of the LaTeX note disagreed on the thermal fraction** (7.6% vs
1.3%). Not two merged MC runs, as suspected — a binning mismatch. The library's
internal `calculate_energy_fractions` uses bins
cold `<0.1 eV` / thermal `0.1–100 eV` / epi `100 eV–500 keV` / fast `>500 keV`,
while §4 and the tile cell use thermal `<1 eV` / epi `1 eV–100 keV` /
fast `>100 keV`. §3 was quoting the library's `thermal` bin and labelling it
"E < 1 eV". Both numbers were right for their own convention; the prose
attributed one to the other.

**5. Self-inflicted, from the previous session:** the scatter-vs-capture cell
normalised its two histograms to hard-coded fractions of the capture rate
(0.5 and 0.05), so it reported a scatter/capture ratio of **0.55× by
construction** — identical for both samples, and contradicting the ~38× from
the rate cell two cells above. That bogus number reached the LaTeX note.
Removed; the histograms now carry MC rates end to end.

### What changed

| | old | new |
|---|---|---|
| Shield transit time | not tracked | accumulated `Σ s/v` per step |
| Tile model | single-pass `p(E)`, instantaneous | 3-D random walk in the 3×3×1 box with a clock |
| Predicted observable | ToF bump at 20–140 µs | **die-away**, λ ≈ 2.8 µs |
| Capture efficiency | ⟨p⟩ = 0.99 % | ε = 1.96 % (A) / 1.93 % (B) |
| Rate @100 % duty | 12.7 / 5.2 Hz | 25.2 / 10.1 Hz |
| σ_H at 1 eV | 5.19 b (spline ringing) | 20.5 b |
| σ_H at thermal | 34 b | 82 b (bound) / 20.5 b (free) |
| scatter/capture | 0.55× (fabricated) | 14× (both samples) |

New notebook cell `tile_dieaway_001` → `pns_tile_dieaway.png`.

### Does it agree with the data now?

Partly, and the residual is informative.

- **Functional form: yes.** The prediction is now an exponential die-away, not
  a ToF distribution. This is the qualitative point the review made — a
  power-law flux maps to a power law in time, never to an exponential, so the
  old model could not have fit the data at any parameter value.
- **λ: 20 % low.** MC gives **2.835 ± 0.063 µs** (A) and **2.801 ± 0.061 µs**
  (B) against measured **3.44 ± 0.13 µs**. That the two samples agree with
  each other is a good internal check: λ is a property of the tile, not of the
  source distance, and the MC reproduces that independence.
- **The residual points at the boron loading.** λ is a steep function of it:

  | B [% wt] | λ [µs] | ε (thermal) |
  |---|---|---|
  | 1.2 | 3.69 | 46.0 % |
  | 1.4 | 3.45 | 49.9 % |
  | 1.6 | 3.17 | 53.1 % |
  | 1.8 | 3.00 | 56.1 % |
  | 2.0 | 2.84 | 58.4 % |

  Measured λ = 3.44 µs ⇒ **B ≈ 1.4 % wt**, against the 2 % nominal we assumed
  and never verified. Other candidates for the residual: tile thicker than
  1 cm (λ = 3.17 µs at 1.5 cm), or neutron albedo off the light guide / wrapping
  returning leaked neutrons.

**Worth doing:** the die-away is a boron-loading meter. λ is set by
1/(Σ_a v) against leakage and is insensitive to the thermal spectrum details
— including the 25.3 meV energy-floor artifact — because Σ_a·v is constant for
a 1/v absorber. So it measures the boron content with essentially no model
dependence. Getting the actual loading and tile thickness from the
manufacturer would close this out.

### Consequences for the analysis

- The absent 20–140 µs bump is **expected**, not a problem. The MC's own
  capture-weighted peak at 91 µs (A) / 136 µs (B) was an artifact of the
  free-flight ToF model; there was never a reason to search there.
- Energy alone cannot separate scatters from captures — the recoil continuum
  covers the quenched ~90 keVee capture line and outnumbers it ~14x. The
  discriminant is the delayed gate, not pulse height.
- Rates roughly doubled (the time-resolved tile captures ~2× more than the
  single-pass model, since neutrons get multiple chances).

### 2026-08-06 (later) — full regeneration of the LaTeX note

Everything above is now propagated into `pns-3det-notes`. Done in that pass:

- One energy-bin convention everywhere: thermal `<1 eV`, epithermal
  `1 eV–100 keV`, fast `>100 keV`. Defined once in §3 and used in every table.
- All numbers regenerated: transmission 15.2/13.8% → **7.2/6.6%**, flux
  302/122 → **143/58 n/cm²/s**, R_surf 2716/1097 → **1285/525 n/s**, fate table,
  spectrum fractions, mean collisions, naive-exponential estimate
  (2.5% → 1.3%), B-HDPE MFP 3.83 → 3.29 cm, wood MFP 10.5 → 9.3 cm.
- Table 4 rates now use the time-resolved efficiency: **25.2 / 10.1 Hz** at
  100% duty (was 21.9 / 8.38).
- New §4.4 "Time-resolved capture in the tile: the die-away" — the physics, the
  4.70 µs pure-capture lifetime, leakage, the boron-loading scan, and the
  comparison with the measurement.
- §5 "ToF separation" replaced. It used to predict the 20–140 µs arrival peak;
  it now states plainly that the prediction was an artifact and gives the
  practical consequence (~10 µs gate, not ~100 µs).
- New §5.2 on the photon background from shield captures — it was computed in
  the notebook but the note still carried a caveat saying it was not tracked.
  At Sample A the gammas (43.8 Hz) exceed the neutron signal (25.2 Hz).
- Abstract and intro rewritten to match.

PDF: `3det_pns_MCstudy_v3.pdf` (14 pages).

Notebook change in the same pass: the tile cell moved before the gamma cells,
so the gamma/signal ratio uses the time-resolved rate (A: 3.5× → **1.7×**,
B: 1.7× → **0.87×**). Cell 6 no longer quotes a scatter/capture ratio, because
its capture normalisation is the single-pass lower bound and it contradicted
the 14× from the time-resolved cell.

### Still to do

- [x] LaTeX note numbers pass — done, see above.
- [x] Die-away section in the note — now §4.4.
- [ ] **Measure the boron loading and tile thickness.** The die-away says
      ~1.4 % wt against a 2 % nominal. This is the single biggest open item:
      it shifts the predicted rate and is cheap to settle.
- [ ] Room return is still not modelled. λ cannot distinguish direct from
      wall-scattered neutrons — it is a tile property — so only the
      *normalisation* can tell us how much room return there is. That makes
      the absolute rate the interesting comparison, and it is the number the
      1-D model is least able to defend.
- [ ] Re-run the other notebooks in this package against the corrected
      cross sections (`notebooks/neutrons.ipynb` is the one that uses them).
- [ ] Neutron albedo off the light guide / wrapping is not modelled; it would
      lengthen λ and raise the efficiency, and is the other candidate for the
      20 % residual.

---

## 2026-08-24 — H-capture bug fix + Sample B report plots

**Bug found in the shield MC (pns.ipynb cell 2):** H radiative capture was a
flat `ABSORPTION_PROB = 2%` per H-scatter at *every* energy. H(n,gamma) is 1/v
(0.332 b thermal), so that probability is right only AT thermal and ~1e-4
during slowdown; with ~18 slowdown collisions the flat hack killed ~23% of all
neutrons mid-moderation. Also Fe(n,gamma) captures in Sample B's steel were
lumped into "H capture".

**Fix (`sampleB_fixed_mc.py`, standalone):** H is a proper 1/v absorber in
every H-bearing layer (competing in Sigma_tot with B-10 / Fe); Fe captures are
their own fate. Validation: analytic thermal branching B10:(B10+H) in B-HDPE =
98.8%, MC gives 98.8%.

**Sample B fates, old -> fixed (100k, seed 20260724):**
transmitted 6.6 -> 6.9% | backscattered 16.0 -> 17.6% |
B-10 capture 53.9 -> 74.6% | H capture 23.5 -> 0.9% | Fe 0.04%.
Transmitted spectrum barely changes (soft tail slightly up): the wrongly-killed
neutrons were destined for B-10 capture, not transmission.
Fix PORTED into the notebook same day (cells 0, 2 and the gamma-depth MC in
cell 12); _mc_state.pkl regenerated with the fixed physics (backup of the
pre-fix notebook: pns.ipynb.bak_hfix). Fixed fates: A 7.75% transm / 73.8%
B-10; B 6.88% / 74.5%. TOF figure reworked to FIRST-interaction branching
(Sigma_i/Sigma_tot x (1-exp(-Sigma_tot L)) — independent 1-exp(-Sigma_i L)
curves double-count) plus a visible-scatter curve (E_n > 100 keV): late
thermal arrivals scatter a lot but deposit nothing visible — TOF>10 us
expects ~21 captures vs ~0 visible scatters per 100k emitted.

**New figures:** `sampleB_spectrum.png` (2.1% th / 20.7% epi / 77.2% fast at
the tile), `sampleB_fates.png`, `sampleB_tof_interactions.png` (TOF 14 ns
fast peak -> ~90 us thermal bump incl. 14.5 cm air; flux x p_capture peaks at
the thermal bump, p_cap(1 cm tile) up to 62% thermal, p_scat ~70-99%).
Fixed-MC state cached in `_mc_state_fixedB.pkl`.

---

## 2026-08-24 (later) — making the MC reproduce the measured timing

**No-bump check (data):** 20260731 period-500us run, tau to 175 us: rate in the
MC ballistic-drift window (55-115 us) = 7.65/us vs flat 7.42/us -> +0.4 sigma.
The single-pass drift channel is NOT in the data.

**Diagnosis:** (1) OVERESTIMATED - the 1-D MC counts every transmitted neutron
as "arriving at the tile"; diffuse (thermalised) exits have ~1% chance of
hitting a 3x3 cm tile at 15 cm vs ~10-50x more for forward fast punch-through
-> slow arrivals suppressed 10-100x -> no drift bump, consistent with data.
(2) MISSING - shield-capture GAMMAS: 74% of cone neutrons capture in the
shield; the 478 keV gammas reach the tile at light speed carrying the shield's
capture-time profile (median 1.3 us, mean 2.0 us: thermalisation + 2 us dwell
at 5% B). (3) MISSING - local moderation feeding tile captures on the tile
dwell clock (2.8-4.7 us).

**Composite fit (`composite_timing.py`, fit tau>1.5us on the 20260731 run):**
shield-gamma shape straight from the MC clock (NO free shape parameter) +
tile-dwell exponential + flat:  chi2/ndf = 43/39, split 83% shield-gamma /
17% tile (lambda_tile = 6.8+-3.3 us). Model comparison: shield-only 61/41,
tile-only 46/40 (lambda 2.50+-0.16). Degenerate-ish but composite preferred;
MATCHES the data-side finding that the delayed excess is mostly capture
gammas with ~half of the alpha-slice PSD-tagged.

**Implication:** the measured die-away rate is NOT the tile capture rate — it
is dominated by shield-capture gammas Compton-scattering in the tile; the
alpha-tagged count is the clean tile-capture number. Boron-loading inference
from lambda must use the composite, not a single exponential.

**Next for full reproduction:** upgrade shield MC to 3-D (track position +
direction, ray-trace to the tile) for honest arrival weights and absolute
rates; add a minimal local-moderation stage (tile + shield-face cavity).

**2026-08-24 (later still) — real bench geometry + 3-D MC (`sampleB_mc3d.py`):**
true stack (user): 14 B-HDPE / 1 wood / 5 air / 1 steel / 2 air, tile at 23 cm.
3-D tracking + ray-trace to the 3x3 cm tile. Acceptance nearly energy-uniform
(fast 2.9%, epi 1.8%, thermal 1.8%): ALL exits are diffuse, so acceptance is a
~x40 absolute scale factor, not a shape cut (earlier "slow exits miss more"
intuition was wrong). With only 7 cm air the thermal drift compresses: the
isolated 66-110 us bump of the planned 15 cm standoff disappears; the direct-
arrival capture channel becomes a LOW BROAD 1-70 us component — matching the
small ~3 sigma excess the data shows at 15-35 us (9.8 vs 7.4 flat /us) and the
55-115 us null. Visible scatters all <70 ns. Fates: 7.2 tran / 20.0 back /
71.8 B10 / 0.9 H / 0.1 Fe. Per 1M cone-emitted: 6.4 direct captures, 360
visible scatters. Full picture: die-away 0-10 us = shield gammas + locally
moderated captures; 15-35 us tail = direct slow arrivals; flat = ambient.

## 2026-08-26 — hydrogen review and the gamma-level MC

Two written reviews now sit next to the code, both driven by a factor-10 puzzle
(the MC predicted 10-50 fast-neutron recoil triggers per tile capture, the data
show ~2.6) and by the new measured tile energy scale (478 keV Compton edge fitted
at 2.75 V ns -> 113 keVee/V ns, alpha+7Li ~71 keVee; the adopted 85 keVee is retired).

**`HYDROGEN.md`** (`h_hydrogen.py`, cross-checks in `checks/`): H in the shield
takes 1/80.5 of the captures (0.89 %, the MC's 0.90 % is exact arithmetic); the
thermal diffusion length in the borated block is 0.24 cm, so the shield emits
gammas, not thermal neutrons. In the tile H takes 1.8 % of thermal captures.
With Birks (kB = 0.0125) the 47 keVee trigger is a 404 keV proton, so the MC's
50 keV recoil cut was 8x too low; the recoil:capture ratio goes 54 -> 21 (light
threshold) -> 3.8 (tile treated as a moderator, not an attenuator) against 2.6
measured once the in-gate window is counted correctly (tau_rise-based). Room
return is not needed.

**`GAMMA_MC.md`** (`sampleB_gamma.py`, ~20 s): capture vertices re-recorded
(4M n), photon transport with buildup, full in-tile cascade with electron escape,
time-resolved neutron walk in the tile, and the SAME delayed-minus-late
subtraction as the data. Verdicts: (a) H-capture gammas are 1.07 % of tile
gamma interactions -- negligible, and 78x too few to be the hard tail; (b) the
MC reproduces the delayed spectrum below the 478 keV edge to x1.3-1.7 (a 20 deg
-> ~28 deg illuminated cone closes the normalisation) but is x12 low above the
edge and x82 low above the 4.4 V ns clip, where the data put 21 % of the
excess; the timing is reproduced by neither MC component (shield-gamma clock
1.8 us vs 4.1 measured; tile capture 11.8 vs 3.3); (c) the ranked explanation
is (n,gamma) on structure ADJACENT to the tile (Fe/Al holder, bracket, bench)
fed by the missed-tile flux, which is 37x the flux reaching the tile: 3.2 % of
it capturing next to the tile supplies the hard excess with the right few-us
local clock. Excluded with numbers: pile-up, the modelled steel plate, room
return, albedo, H. The MC's world must not end at the tile plane.

The July-31 run (500 us period) IS reproduced by the shield-capture clock
(72 % share); every August run rejects it and decays with 3.5-3.8 us on both
tiles, independent of period (same-day 500/50 us test on Aug 18). Whatever
holds the tile is the leading suspect for both the hard tail and the slow
clock; it changed between the two campaigns.

Caches (`_*.npz`, `_*.pkl`) are git-ignored; `sampleB_mc3d.py` regenerates
`_mc3d_cache.npz` (10M neutrons, ~hours) and `sampleB_gamma.py` its own
`_gam_*.npz` in seconds.
