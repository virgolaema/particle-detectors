# What the tile records after the gate — a gamma-level MC of the PNS boron experiment

**Everything below is produced by `sampleB_gamma.py` in this directory.**
`sampleB_mc3d.py` and `_mc3d_cache.npz` are **not modified**; the cache is read
only, for the tile-arrival list. New caches written: `_gam_vertices.npz`
(capture vertices from a fresh 4 M-neutron run), `_gam_tilewalk.npz` (in-tile
neutron random walk), `_gam_data.npz` (data spectra pulled from
`pns-waveform-ana`). Figures: `gamma_mc_spectrum.png`, `gamma_mc_sources.png`,
`gamma_mc_timing.png`.

Run with
`../../energy-deposits-env/bin/python sampleB_gamma.py` from this directory.
Date: 2026-09-06.

---

## 0. Executive summary

1. **The MC reproduces the delayed spectrum below the 478 keV Compton edge and
   nothing above it.** Normalised to the measured tile-capture rate, it gives
   15,752 delayed events above threshold against 24,507 measured (data/MC 1.56),
   but only 687 above 2.75 V.ns against 8,598 (×12.5 short) and 64 above the
   4.4 V.ns ceiling against 5,256 (**×82 short**).
2. **Hydrogen-capture gammas are negligible** — 0.9 % of the predicted delayed
   counts, 1.07 % of the gamma interactions in the tile. They are nevertheless
   the *largest modelled* contributor to the hard tail (78 % of the predicted
   >4.4 V.ns events), and they are still 78× too few to explain it. Since the
   H:B-10 branch is fixed arithmetic (1 : 80.5 in the shield), that factor
   cannot be bought. **H is excluded as the origin of the hard excess.**
3. **The die-away is wrong in both directions.** The shield-gamma clock is
   λ = 1.8 µs (measured 478-band λ = 4.06 ± 0.14); the tile-capture clock is
   λ = 11.8 µs (measured alpha ROI 3.33 ± 0.11). The data sits *between* the
   MC's two clocks and closer to neither. The composite "83 % shield gamma +
   17 % tile" picture does not survive: 83 % of a 1.8 µs component cannot give
   a 4 µs die-away.
4. **The single largest physics correction found here overturns HYDROGEN.md
   §5.2b.** A 1 cm PVT tile is *not* a moderator for fast neutrons: the
   time-resolved random walk gives ε_capture = **0.022 %** for arrivals above
   100 keV, against 4.9 % (epithermal) and 48.6 % (thermal). The ×4.6 boost
   over the single-pass model is real, but it comes entirely from multi-pass
   capture of neutrons that were *already slow on arrival*, not from moderating
   the 81 % fast flux. The quoted "×5.6, fast neutrons thermalise in situ" is
   the right number for the wrong reason.
5. **The missing hard component is (n,γ) on high-Q material immediately around
   the tile** — its mount, frame, can, or the bench — fed by the neutron flux
   that passes the shield and misses the tile (6.9 % of cone-emitted neutrons,
   37× the tile-hit flux). It needs only **3.2 %** of that flux to capture on
   such material. Every other candidate is excluded by a factor 10–5000 or by
   its time constant.

---

## 1. What was built

### 1.1 Capture vertices (new transport run)

`transport()` is a *vectorised* rewrite of `sampleB_mc3d.run()` — identical
cross sections, identical geometry, identical scattering kernels — that stores
`(x, y, z, t)` for every capture and tags it B-10 / H / Fe. Validation against
the untouched 10 M cache (4 M neutrons, 20° cone):

| fate | new run | 10 M cache |
|---|---|---|
| transmitted | 7.106 % | 7.127 % |
| backscattered | 20.025 % | 20.037 % |
| B-10 capture | 71.874 % | 71.837 % |
| H capture | 0.893 % | 0.897 % |
| Fe capture | 0.102 % | 0.102 % |

Capture-time / depth (the quantities the gamma stage needs):

| species | ⟨z⟩ | median t | mean t | fraction in 0.5–15 µs |
|---|---|---|---|---|
| B-10, shield | 5.91 cm | **1.33 µs** | 2.04 µs | 76.4 % |
| H, shield+wood | 5.99 cm | 1.37 µs | 2.41 µs | 76.3 % |
| Fe, steel | 20.41 cm | **37.96 µs** | 54.37 µs | **28.0 %** |

The B-10 numbers reproduce HYDROGEN.md §1.3 (median 1.34 µs, ⟨z⟩ 5.74 cm) to
better than 3 %. The **Fe result is new and important**: thermal neutrons rattle
inside the 1 cm steel plate for tens of microseconds (Σ_a,Fe = 0.217 cm⁻¹ →
1/(Σ_a v) = 21 µs), so Fe capture gammas arrive on a ~40 µs clock, i.e. *outside*
the delayed window and inside the flat pedestal that the late-window subtraction
removes.

### 1.2 Gamma transport, source → tile

Not a solid-angle formula: a photon Monte Carlo with a **next-event estimator**.
For each capture vertex a photon is emitted isotropically and tracked through
the z-slab stack; at birth and after each Compton scatter (up to 5 generations,
implicit absorption on the weight), the expected number of *first interactions
in the tile* is scored by sampling `NE_SAMP = 3` points uniformly in the 9 cm³
tile volume:

```
w_j = w · (V/M) · µ_tile(E) / (4π r_j²) · exp(−Σ_i µ_i L_i)
```

with the path decomposed exactly through the slabs and through the tile itself
(3-D ray–box entry, so gammas entering the side faces are handled). This
integrates the geometry *and* the attenuation *and* **buildup** — the scattered
photons that a `exp(−µd) × Ω/4π` estimate throws away:

| line | uncollided fraction | buildup factor |
|---|---|---|
| 478 keV | 36.6 % | **2.73** |
| 2223 keV | 42.2 % | 2.37 |
| 7.6 MeV | 60.0 % | 1.67 |

### 1.3 In-tile response

A full photon cascade inside the tile: Compton → Compton → … until escape or
absorption, plus pair production above 1.022 MeV with both annihilation quanta
tracked. This subsumes the requested "second Compton": the 478 keV component
shows a multiple-scatter tail running from the Compton edge (2.75 V.ns) up to
the full-energy point (4.23 V.ns), visible in the right panel of
`gamma_mc_sources.png`.

**Compton-electron escape is included** and it matters: the tile is 1 cm thick
and the Katz–Penfold practical range in PVT is 0.39 cm at 1 MeV, 0.90 cm at
2 MeV and 3.4 cm at 7 MeV. Deposits are `T · min(1, s/R(T))` with `s` the
distance to the boundary along the electron's own direction (Compton-electron
angle from the exact kinematics). Without this, the 2.2 MeV and 7.6 MeV lines
would be far too hard.

### 1.4 The tile as a neutron target

A time-resolved 3-D random walk in the 3×3×1 cm tile, fed by the arrival list
from the 10 M cache (18,840 hits, each replayed 3×), with ENDF H and C elastic
scattering, B-10 and H 1/v capture, Birks-quenched proton light (the L(E_p)
table of HYDROGEN.md §3.1, kB = 0.0125), carbon recoils at ~1 % light, and a
150 ns pulse-integration window for summing recoils.

| arrival group | arrivals | ε_capture | captures | median t_arrival |
|---|---|---|---|---|
| fast > 100 keV | 45,768 | **0.022 %** | 10 | 0.01 µs |
| epithermal | 9,924 | 4.86 % | 482 | 0.29 µs |
| thermal < 1 eV | 828 | 48.6 % | 402 | 20.87 µs |
| **all** | 56,520 | **1.61 %** | 894 | — |

This is the result that corrects HYDROGEN.md §5.2b (see §5.1 below).

---

## 2. Assumptions — all of them, explicitly

**Measured, taken as given (not re-derived):** 113 keVee/V.ns; alpha line
0.63 ± 0.10 V.ns; threshold 0.35 V.ns; ceilings 4.4 (run 20260814) and
9.5 V.ns (20260820); gate 0.958 µs; delayed 0.5–15 µs, late 15–18.5 µs scaled
by 14.5/3.5; 60,356 gate-tagged records; 3,300 tile captures; 8,534 in-gate
recoils.

**Resolution.** σ/E = 10 % at 71.2 keVee scaling as 1/√E (photostatistics, as
instructed). For the alpha line only, an extra constant term is added in
quadrature so the line comes out at the measured σ = 0.10 V.ns. *This is an
assumption*: part of the measured alpha width is the 6 % ⁷Li ground-state branch
(alpha 1.78 MeV instead of 1.47), not resolution.

**Gamma yields.** B-10: 0.94 × 478 keV. H: 1 × 2223 keV. Fe: 50 % a single
7.638 MeV photon, 50 % a three-photon cascade sharing 7.646 MeV with a
Dirichlet(1,1,1) split ⇒ mean multiplicity **2.0** (the real Fe-56 cascade has
≈ 2.1 photons and ≈ 51 % intensity in the 7631/7646 doublet). The Fe term is
0.2 % of the prediction, so this approximation is irrelevant to every conclusion.

**Attenuation.** Plastics use the XCOM CH₂ table in `xcom_polyethylene.txt`
scaled by electron density (n_e = 3.421e23 cm⁻³ B-HDPE, 1.913e23 wood,
3.419e23 PVT, 3.6e20 air). Incoherent scattering is the exact Klein–Nishina
× n_e; everything left over (µ_tot − µ_incoh) is treated as pair production
above 1.022 MeV and as absorption below it. At 478 keV that residual is 0.16 %
of µ_tot, at 100 keV 1.6 %, at 50 keV 12 % — so coherent scattering is
mis-booked as absorption only where it cannot matter. Iron uses a hard-coded
NIST µ/ρ table: **0.0840 cm²/g at 500 keV** (→ µ = 0.659 cm⁻¹, exp(−µ·1 cm) =
0.517), **0.0411 at 2.223 MeV**, **0.0299 at 7.6 MeV**. (The brief suggested
0.045 at 2.2 MeV; XCOM says 0.041. Over 0.5 cm of steel this is a 1 % effect.)

**Known approximations, stated because they bias the answer:**
- The slab stack is **laterally infinite**, both for neutrons and for scattered
  photons. For the near-forward rays that reach the tile this is exact; for the
  buildup component it is generous (a real 20 × 20 cm block lets wide-angle
  scattered photons escape), so the ×2.7 buildup factor is an **upper bound**.
- Air layers have **zero neutron cross section** (as in `sampleB_mc3d.py`), so a
  slow neutron crosses the 7 cm of air ballistically. Combined with infinite
  slabs, this creates a late (20–40 µs) thermal-arrival population at the tile
  that a real open bench would lose sideways.
- Electron escape uses a straight-line practical range; it ignores the fact
  that dE/dx is not uniform along the track.
- Nothing exists behind, beside, or around the tile: no mount, no can, no
  bench, no room. That is precisely the gap identified in §5.
- Gamma flight time (0.8 ns over 23 cm) is neglected.

---

## 3. Results

### 3.1 Normalisation and the implied source strength

Per cone-emitted neutron the MC gives

| quantity | value |
|---|---|
| tile B-10 captures, all times | 2.980e-05 |
| tile B-10 captures, inside the net delayed window | 1.286e-05 |
| in-gate recoil triggers (light > 0.35 V.ns) | 1.621e-04 |
| **delayed** recoil triggers | **0** |

so the two normalisations are

| anchored on | cone-emitted neutrons per gate-tagged record | isotropic-equivalent per pulse |
|---|---|---|
| tile captures, 3,300 / 60,356 | **4,251** | 1.41e5 |
| in-gate recoils, 8,534 / 60,356 | **872** | 2.89e4 |

(the isotropic-equivalent column divides by the 20° cone's 3.016 % of 4π; it is
*not* a source-strength measurement, because "per gate-tagged record" is per
*sampled* pulse, not per physical pulse.)

**They disagree by ×4.9.** Equivalently, in-gate recoils : tile captures is
**5.4 : 1** in the MC (all captures) or 12.6 : 1 (captures inside the net
window), against **2.59 : 1** measured. HYDROGEN.md's ledger predicted 3.8 : 1
using an assumed ×5.6 moderator boost; the explicit random walk gives a smaller
boost placed on a different population, and the residual disagreement is a
factor 2–5, not 1.5.

The tile-capture normalisation is used everywhere below (it is the one anchored
on the same delayed window as the spectrum being predicted). Using the recoil
normalisation instead would divide every MC number by 4.9 and make every
shortfall five times worse.

**Delayed neutron recoils are exactly zero**, and not by assumption: of 18,840
tile arrivals, **none** has E > 404 keV later than 1.5 µs. A neutron cannot be
both slow enough to arrive late and fast enough to make 47 keVee of light.

### 3.2 The predicted charge spectrum

Counts per 60,356 gate-tagged records, delayed window minus the scaled late
window — the *same* subtraction the data receives:

| component | > 0.35 V.ns | > 2.75 | > 4.4 |
|---|---|---|---|
| shield B-10 478 keV | 12,269 | 438 | 2 |
| shield/wood H 2.22 MeV | 148 | 70 | 50 |
| steel Fe 7.6 MeV cascade | 35 | 16 | 11 |
| tile capture: alpha (+ its own 478 keV) | 3,300 | 162 | 0 |
| tile H capture 2.22 MeV | 0 | 0 | 0 |
| delayed neutron recoil | 0 | 0 | 0 |
| **MC total** | **15,752** | **687** | **64** |
| **DATA 20260814 net excess** | **24,507** | **8,598** | **5,256** |
| **data / MC** | **1.56** | **12.5** | **82.2** |
| *fraction of the excess above the cut — MC* | — | **4.4 %** | **0.41 %** |
| *fraction of the excess above the cut — data* | — | **35.1 %** | **21.4 %** |

Bin by bin (`MC uncoll` = the same prediction with buildup switched off):

| band [V.ns] | data | MC | d/MC | MC uncoll | d/unc | MC composition |
|---|---|---|---|---|---|---|
| 0.35–0.70 | 2,948 | 4,217 | 0.70 | 1,616 | 1.82 | shield 73 %, tile 27 % |
| 0.70–1.00 | 3,494 | 4,323 | 0.81 | 2,346 | 1.49 | shield 58 %, tile 42 % |
| 1.00–1.50 | 3,584 | 2,541 | 1.41 | 881 | 4.07 | shield 96 % |
| 1.50–2.00 | 2,417 | 1,739 | 1.39 | 810 | 2.98 | shield 95 % |
| 2.00–2.50 | 2,390 | 1,418 | 1.69 | 902 | 2.65 | shield 96 % |
| 2.50–2.75 | 1,077 | 827 | 1.30 | 700 | 1.54 | shield 96 % |
| 2.75–3.50 | 1,731 | 562 | 3.08 | 514 | 3.37 | shield 75 %, tile 23 % |
| 3.50–4.40 | 1,493 | 61 | **24.5** | 52 | 28.7 | tile 56 %, shield 23 %, H 17 % |
| > 4.4 | 5,256 | 64 | **82.2** | — | — | H 78 %, Fe 17 % |

Read the columns, not the bottom line. Below the edge the MC is right to within
a factor 1.3–1.7 in shape *and* would be right in normalisation with a ×1.6
increase in the shield gamma flux — except that its slope is wrong in a
specific way: it is **too soft**, over-predicting 0.35–1.0 V.ns by 25 % while
under-predicting 1.0–2.5 V.ns by 40 %. Switching buildup off inverts the sign of
that error (the pure uncollided 478 keV continuum rises towards its edge), so
the truth is somewhere between the infinite-slab buildup used here and no
buildup at all — exactly what a finite shield block would give. Part of the
0.35–0.7 V.ns excess in the MC is also the trigger turn-on, which the MC applies
as a hard step at 0.35 V.ns and the electronics does not.

Above the edge nothing works. **The 478 keV line cannot deposit more than
311 keVee = 2.75 V.ns in a single Compton**, and multiple scattering can only
reach the 478 keV full-energy point at 4.23 V.ns with a probability of order
10⁻³. Run 20260820 (ceiling 9.5 V.ns) shows the excess continuing *smoothly* to
9 V.ns (= 1.07 MeVee) with a further 684 net events beyond, i.e. deposits above
1 MeVee. Nothing in the modelled inventory can put 21 % of the delayed excess
above 500 keVee.

### 3.3 Interaction ratios in the tile

Per cone-emitted neutron, gamma interactions in the tile (any deposited energy):

| | rate | ratio to B-10 |
|---|---|---|
| shield B-10 478 keV | 1.204e-04 | 1 |
| shield/wood H 2.223 MeV | 1.291e-06 | **1.07 %** |
| steel Fe 7.6 MeV cascade | 1.013e-06 | **0.84 %** |

The H:B interaction ratio of 1.07 % sits inside HYDROGEN.md's hand-calculated
1.0–1.4 % band — the two independent calculations agree. The Fe:B ratio of
0.84 % is ~4× larger than HYDROGEN.md §1.5's 3.7 %-of-478-rate estimate once the
proper solid angle of a 3×3 cm tile 2.5 cm from a 1 cm plate is integrated
rather than approximated, but the *recorded* Fe contribution is much smaller
than that (0.2 %) because 72 % of Fe captures happen later than 15 µs.

### 3.4 Die-away

Exp + flat fits over 0.5–18.5 µs, MC and data treated identically:

| band | MC λ [µs] | data λ [µs] |
|---|---|---|
| alpha ROI 0.45–0.85 | 4.41 | 3.33 ± 0.11 |
| 478 Compton 1.0–2.5 | 1.94 | 4.06 ± 0.14 |
| above edge 3.0–4.4 | 2.74 | 3.84 ± 0.19 |
| clipped > 4.4 | 2.30 | 3.22 ± 0.11 |
| *pure shield gamma* | **1.84** | — |
| *pure tile capture* | **11.80** | — |

Three separate failures:

- **The shield-gamma clock is too fast.** 1.84 µs is the B-HDPE thermal dwell
  (1/(Σ_a v) = 1.95 µs) shortened by the 64 % of captures that happen during
  slowing-down. The measured 478-region die-away is 4.06 µs — more than twice
  as long. A component that is 78 % of the MC's delayed spectrum and decays with
  1.8 µs cannot produce a 4.1 µs measured decay. **The "83 % shield gamma"
  composite result is not reproduced here.**
- **The tile-capture clock is too slow** (11.8 µs vs 3.33 µs measured), because
  in the MC 45 % of tile captures come from thermal neutrons that arrive at a
  median of 20.9 µs. Those late arrivals are an artefact of the infinite slabs
  plus the zero-cross-section air (§2). Take them out and the MC tile clock
  becomes the epithermal-arrival clock, median t_capture 4.15 µs — which does
  match the data. This is a concrete, testable statement: **the measured
  3.33 µs alpha die-away says the tile is fed by epithermal, not thermal,
  arrivals.**
- **The data's four bands all decay with 3.2–4.1 µs**, i.e. within ~25 % of each
  other, while the MC's bands span 1.8–11.8 µs. Whatever makes the hard tail
  shares a clock with the alpha line to within 4 %. That is a strong
  constraint: it points at a *neutron population local to the tile*, not at the
  shield.

### 3.5 A data cross-check on the nature of the hard events

PSD (tail/total), ambient-subtracted the same way:

| selection | net | ⟨PSD⟩ | frac PSD > 0.24 |
|---|---|---|---|
| alpha ROI 0.45–0.85 | 5,183 | 0.2263 | 0.386 |
| 478 Compton 1.0–2.5 | 7,914 | 0.2198 | 0.276 |
| above edge 3.0–4.4 | 2,487 | **0.2120** | 0.144 |
| clipped > 4.4 | 5,296 | 0.2686 | 0.563 |

The **unclipped** hard band (3.0–4.4 V.ns) is *more* electron-like than the
478 keV region — the 2.75–4.4 V.ns excess is genuinely Compton electrons, i.e.
genuinely gammas. The clipped row must **not** be read as "heavy": clipping
truncates the prompt peak, which mechanically inflates tail/total. (In-gate
clipped events show the same 0.28.)

---

## 4. Answers to the three questions

**(a) Are H-capture gammas negligible? Yes — and the question is now closed.**
They are 1.07 % of the tile's gamma interactions, 0.9 % of the predicted
delayed counts, and 0.5 % of the counts below the edge. The one place they are
not negligible is *within the MC's own hard tail*, where they supply 78 % of
the (tiny) >4.4 V.ns prediction. To supply the measured 5,256 hard events they
would have to be **78× more numerous**, i.e. 70 % of all neutrons would have to
capture on hydrogen. Σ_H/Σ_B10 = 1/80.5 in the shield is fixed 1/v arithmetic,
so this is excluded outright. HYDROGEN.md's §1.5 conclusion ("the H gammas are
buried, not excluded") stands, with the numbers now coming from a transport
calculation rather than a solid-angle estimate.

**(b) Does the MC reproduce the post-gate die-away spectrum? Below the
Compton edge, roughly. Above it, not at all.** Shape below 2.75 V.ns is right
to a factor 1.3–1.7 with a systematic softness; normalisation is 1.56× low,
which a ×1.7 increase in illuminated shield closes. Above 2.75 V.ns the MC is
12× low and above 4.4 V.ns it is **82× low**. The timing is reproduced by
neither component: the modelled dominant component decays 2.2× too fast, the
tile component 3.5× too slow, and the data's bands are all within 25 % of each
other where the MC's span a factor 6.

**(c) Where and why does it fail?** Three distinct failures:

1. *A whole class of gamma sources is absent.* The MC's world ends at the tile
   plane. There is no mount, no housing, no bench, no room — and no (n,γ) in
   any of them. The only high-Q captor in the model (1 cm of steel, 2 cm away)
   is both under-illuminated and on a 40 µs clock.
2. *The illuminated shield volume is set by an arbitrary 20° cone.* An
   isotropic D-D source lights up whatever the block subtends; the cone is
   3.0 % of 4π.
3. *The tile's neutron physics is right but its feed is wrong.* The random walk
   is sound; the arrival spectrum handed to it contains a spurious late-thermal
   population created by infinite slabs and collisionless air.

---

## 5. Candidate ranking for the missing signal

Everything is normalised to the tile-capture anchor (4,251 cone-emitted
neutrons per record). Missing rates, per cone-emitted neutron:

| region | missing rate |
|---|---|
| total > 0.35 V.ns | 3.41e-05 |
| above the edge (> 2.75) | 1.04e-05 |
| hard (> 4.4) | **2.02e-05** |

Available neutron populations, per cone-emitted neutron:

| population | rate |
|---|---|
| reach the tile | 1.88e-03 |
| leak back **out** of the tile (98.4 % of arrivals) | 1.85e-03 |
| pass the shield and **miss** the tile | **6.94e-02** (37× the tile-hit flux) |
| backscattered out of the shield front face | 2.00e-01 |

A probe calculation — captures placed 1 mm in front of the tile face, spread
over ±3 cm, with the full photon MC — gives the detection probability per such
capture:

| gamma | p(> 0.35 V.ns) | p(> 4.4 V.ns) |
|---|---|---|
| B-10 478 keV | 0.0180 | 0.0000 |
| H 2223 keV | 0.0105 | 0.0073 |
| Fe 7.6 MeV cascade (×2 photons) | 0.0112 | **0.0091** |
| Al-27 7.72 MeV (×2) | 0.0108 | 0.0088 |

So the hard excess requires **2.2e-3 adjacent (n,γ) events per cone-emitted
neutron** on a high-Q captor.

### Ranking

| # | candidate | can it supply the 5,256 hard events? | verdict |
|---|---|---|---|
| **1** | **(n,γ) on structure around the tile, fed by the missed-tile flux** | needs **3.2 %** of the 6.94e-02 missed-tile flux to capture on Fe/Al adjacent to the tile | **the explanation**; timing (local moderation, few µs) matches the measured 3.2 µs |
| 2 | shield illuminated beyond the 20° cone | ×1.7 on all shield gammas closes the **total** (⇒ ~27–30° illuminated half-angle, entirely reasonable). Supplies **2 %** of the hard excess | real, but only fixes the soft part |
| 3 | same structural captures fed by the tile's own leakage | would need 120 % of the leak-out flux — there is not enough of it | insufficient on its own |
| 4 | the modelled 1 cm steel plate with more thermal flux | needs ×465 more Fe captures = 48 % of all neutrons capturing on Fe; and its gammas peak at 38 µs | excluded |
| 5 | Fe cascade modelling (multiplicity, hardness) | bounded by ~×2 on a term worth 0.2 % of the prediction | negligible |
| 6 | pile-up in the 150 ns integration window | doubles of 478 keV Comptons cap at 5.5 V.ns and would decay with λ/2 = 2.0 µs; measured 3.22 ± 0.11 µs; run 20260820 shows a smooth continuum past 9.5 V.ns | disfavoured; cannot reach > 5.5 V.ns |
| 7 | neutron albedo from behind the tile | raises the alpha line and lengthens the tile clock; cannot deposit above ~0.9 V.ns | 0 % of the hard excess |
| 8 | room / floor return (concrete H) | 0.3–2 ms lifetime ⇒ flat inside an 18.5 µs record ⇒ removed by the late-window subtraction; the hard band decays with 3.22 µs | excluded as the *shape* |
| 9 | H-capture gammas from the shield | ×78 short (see §4a) | excluded |

The winner is not a fudge: a 3×3×1 cm tile has to be *held* by something, and
whatever holds it — a steel or aluminium frame, a can, a bracket, the bench top
— sits inside a neutron field 37× more intense than the one reaching the tile
itself. Fe(n,γ) releases 7.646 MeV in ~2 photons and Al-27(n,γ) 7.724 MeV; both
put ~1 % of their captures above 500 keVee in the tile from 1 mm away, and both
run on the local thermalisation clock of a few µs, which is what the data
measures for the hard band (3.22 ± 0.11 µs, within 4 % of the alpha line's
3.33 ± 0.11 µs).

---

## 6. Corrections to HYDROGEN.md

1. **§5.2b, "the tile is a moderator, ×5.6" — right factor, wrong mechanism.**
   The explicit time-resolved walk gives ε = 1.61 % per arriving neutron
   (×4.6 over the single-pass 0.348 %), but fast arrivals contribute
   **0.022 %**: 81 % of the flux still supplies 1 % of the captures. A 1 cm PVT
   slab is ~0.3 mean free paths at 1 MeV, and a neutron needs ~18 hydrogen
   collisions to thermalise — it escapes first. The boost is multi-pass capture
   of the epithermal and thermal arrivals, whose ε rises from 0.9 %/12 %
   (single pass) to 4.9 %/48.6 %.
2. **§3.1 footnote, "clipping cannot be removing proton recoils" — wrong under
   the new energy scale.** With the *measured* 113 keVee/V.ns (rather than the
   assumed 134.9), the 4.4 V.ns ceiling is 497 keVee = a **2.13 MeV** proton, well
   below the 2.45 MeV kinematic maximum (610 keVee = 5.4 V.ns). In-gate proton
   recoils **do** clip. This explains the 1,338 in-gate clipped events without
   invoking anything exotic. It does *not* apply to the delayed window, where
   there are no fast neutrons at all (§3.1).
3. **§4, "83 % of the delayed excess is shield-capture gammas".** The MC
   shield-gamma component decays with λ = 1.84 µs; the measured 478-band decay
   is 4.06 ± 0.14 µs. The composite fit that produced the 83 % should be
   re-examined — most likely the flat term is absorbing the mismatch.
4. **§1.5, Fe:B.** The correct solid-angle integration gives the steel plate
   0.84 % of the B-10 interaction rate in the tile, not 3.7 % of the recorded
   478 keV rate; but its *recorded* share is smaller still (0.2 %) because 72 %
   of Fe captures occur after 15 µs.

---

## 7. What to do next

1. **Measure the thing the MC cannot invent.** Photograph / list what is within
   5 cm of the tile and identify every gram of steel, aluminium and copper. A
   single steel bracket reproduces the entire hard excess with a 3 %
   capture efficiency of the missed-tile flux.
2. **Test it without touching the hardware.** Wrap the tile in ~1 mm of
   cadmium or in borated rubber: that removes the thermal flux feeding the
   adjacent captures and should kill the > 4.4 V.ns excess while leaving the
   shield-gamma continuum below the edge almost untouched. Prediction: the
   hard band drops by > 5×, the 1–2.5 V.ns band by < 20 %.
3. **Second test, purely in analysis:** the hard excess should scale with the
   *square* of nothing and the *first power* of the beam current, unlike
   pile-up which scales quadratically. A current scan separates them in one
   afternoon.
4. **Fix the two MC artefacts** before re-running: give the air a real (tiny)
   cross section and terminate slow neutrons that wander more than ~10 cm off
   axis, and either bound the shield laterally or state the buildup as an upper
   limit. Then repeat the illumination scan against the actual block dimensions
   instead of a cone angle.
5. **Re-derive the composite die-away fit** now that the shield-gamma shape is
   known to be a 1.8 µs component, not a 3.6 µs one.
