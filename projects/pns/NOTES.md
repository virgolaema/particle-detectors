# PNS MC — working notes

Personal/working notes for the PNS neutron MC. Not part of the LaTeX note in
`../../../pns-3det-notes` — that document is kept in sync separately, and as of
2026-08-06 it is current with the notebook.

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
