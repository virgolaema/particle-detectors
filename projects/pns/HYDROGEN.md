# Hydrogen in the PNS boron-capture experiment

**What hydrogen does, quantitatively, and what it does to the measured signals.**

Date: 2026-09-06. All numbers derived in `h_hydrogen.py` (physics) and
`checks/h_data_check*.py` (cache cross-checks) in this directory. Neither script
regenerates the MC; both only read `_mc3d_cache.npz` and
`_shield_capture_times.npz`. Cross sections come from
`lib_neutrons.get_endf_cross_sections` (ENDF/B-VIII.0, PCHIP in log-log, bound-H
enhancement on) plus the fixed thermal values quoted below.

Cross sections used throughout: H elastic 20.5 b at 1 eV, **82 b at thermal**
(bound-atom, 4x free), 2.56 b at 2.45 MeV; H radiative capture **0.332 b**
thermal, 1/v; C elastic 4.75 b thermal, 2.07 b at 2.45 MeV; B-10(n,alpha)
**3840 b** thermal, 1/v; Fe(n,gamma) 2.56 b thermal.

---

## 0. Executive summary

1. Hydrogen **moderates**: 18 of the 20 collisions that take a 2.45 MeV neutron
   to thermal in the shield are on H. Without it there is no thermal neutron and
   no boron capture anywhere in this experiment.
2. Hydrogen **competes for captures** but barely: Sigma_H/Sigma_B10 = 1/80.5 in
   the shield, 1/55 in the tile. The MC's 0.90 % H-capture fate is exactly what
   that ratio predicts (0.892 % expected) — it is arithmetic, not a free
   parameter, and it is right.
3. Hydrogen **is the recoil signal**: essentially every triggerable neutron pulse
   in the tile is a proton recoil (C recoils quench to 2-8 keVee, 6-20x below the
   47 keVee threshold, and can never trigger).
4. Hydrogen **inside the tile is why captures happen at all** for the epithermal
   and fast arrivals: it thermalises them in situ on a ~1 us clock. Treating the
   tile as a single-pass attenuator (which is what the 3-D MC does) undercounts
   captures by a factor ~5.6.
5. **The puzzle (item 5) resolves.** MC 54:1 -> data 1.3:1 is closed by three
   corrections, none exotic: the trigger threshold is on *quenched light*, not
   proton energy (54 -> 21.5); the tile is a moderator, not an attenuator
   (21.5 -> 3.8); and **half of the in-gate window is silently thrown away by the
   analysis** (data 1.3 -> 2.6). 3.8 vs 2.6 is agreement at the level this
   comparison can support. Room return is *not* required.
6. **One premise in the brief does not survive checking**: the "delayed excess
   above the 478 keV Compton edge = 4 +- 2 events vs 13,987 below" does not
   reproduce. With the repo's own ambient subtraction the beam-correlated delayed
   excess above 2.31 V.ns is **8,169 +- 95** counts, 56 % of the below-edge
   excess, with no break at the edge. See §1.5. The data therefore does **not**
   bound the H-capture gamma at the 5e-4 level, and the MC's 0.90 % is not in
   tension with anything.

---

## 1. Hydrogen in the shield and the wood

### 1.1 Number densities

The MC (`sampleB_mc3d.py`, lines 28-35) defines, with N_A = 6.022e23:

| material | composition in the code | n [cm^-3] |
|---|---|---|
| B-HDPE, rho = 1.00 | H 14.3 wt% | n_H = **8.612e22** |
| | C 80.7 wt% | n_C = 4.050e22 |
| | B 5.0 wt% nat, 19.9 % B-10 | n_B10 = **5.992e20** (n_B11 = 2.193e21) |
| wood, rho = 0.60 | H 6 wt% | n_H = **2.168e22** |
| | C 50 % + O 44 % as "C-equivalent" | n_Ceq = 2.499e22 |
| steel, rho = 7.85 | Fe | n_Fe = 8.464e22 |
| tile, rho = 1.05 | PVT | n_H = **5.2514e22**, n_C = 4.7262e22 |
| | B 2 wt% nat | n_B10 = **2.517e20** |

So the **shield boron loading is 5 wt% natural boron** and the **tile is 2 wt%**
— both as nominal. Two small bookkeeping notes:

* The B-HDPE H:C atom ratio is 2.126, not the 2.000 of CH2: 0.143/(0.143+0.807)
  = 0.1505 H mass fraction of the hydrocarbon against 2/14 = 0.1429. The MC
  therefore carries ~6 % more hydrogen than pure borated polyethylene. Harmless
  at this level, but it is an inconsistency.
* The MC uses rho = 1.05 for the tile while the datasheet number quoted in the
  brief is 1.03. That is a 2 % shift in n_B10 (2.517e20 -> 2.469e20) and moves
  the pure-capture die-away from 4.62 to 4.71 us. Irrelevant next to the
  1.4-vs-2.0 wt% question NOTES.md already flags.

### 1.2 Mean free paths

Sigma_i = n_i sigma_i; lambda_i = 1/Sigma_i. In the B-HDPE:

| E | sigma_H | sigma_C | sigma_B10(n,a) | lambda_H | lambda_C | lambda_B10 | lambda_tot |
|---|---|---|---|---|---|---|---|
| 2.45 MeV | 2.56 b | 2.07 b | 0.390 b | 4.54 cm | 11.9 cm | 4277 cm | **3.28 cm** |
| 100 keV | 12.90 b | 4.55 b | 1.93 b | 0.90 cm | 5.43 cm | 864 cm | 0.77 cm |
| 1 keV | 20.10 b | 4.75 b | 19.3 b | 0.58 cm | 5.20 cm | 86.4 cm | 0.52 cm |
| 1 eV | 29.68 b | 4.75 b | 611 b | 0.39 cm | 5.20 cm | 2.73 cm | 0.32 cm |
| thermal | 82.0 b | 4.75 b | 3840 b | 0.14 cm | 5.20 cm | 0.435 cm | **0.105 cm** |

At 2.45 MeV **72.4 %** of collisions in the shield are on hydrogen; boron is
irrelevant as a scatterer/absorber there (lambda_B10 = 43 m). Fourteen cm of
B-HDPE is 4.3 fast mean free paths. The 1 cm of wood is only 0.107 mfp at
2.45 MeV (lambda_tot = 9.3 cm) — the wood is a spectator for fast neutrons, but
see §4.

### 1.3 Moderation: collisions and time

Average lethargy gain per collision xi = 1 + a ln a/(1-a), a = ((A-1)/(A+1))^2:
**xi_H = 1.000**, **xi_C = 0.1578**. Weighting by Sigma_s(E) and integrating over
lethargy u = ln(E0/E) from 0 to ln(2.45e6/0.0253) = 18.4:

* mean collisions 2.45 MeV -> thermal in B-HDPE = **20.3**
  (pure H would be 18.4; the C collisions are nearly wasted);
* lethargy-averaged share of collisions on H = **0.893**, i.e. **18.1 on H, 2.2 on C**.

Slowing-down time, t = Int dE / (E xibar Sigma_s(E) v(E)):

    t_sd(2.45 MeV -> thermal) = 2.3 us

(the classic asymptotic formula 2/(xibar Sigma_s v_th) gives ~4 us; the
difference is the bound-H enhancement raising Sigma_s in the last decade, which
in a slowing-down integral is an over-correction — read this as **2-4 us**).
The MC's own clock gives capture-time **median 1.34 us, mean 1.96 us** — shorter
than the full thermalisation time, and §1.6 explains why.

Thermal dwell: Sigma_a(thermal, shield) = n_B10 x 3840 b + n_H x 0.332 b =
2.301 + 0.0286 = **2.329 cm^-1**, so 1/(Sigma_a v_th) = **1.95 us**. That is the
B-HDPE die-away quoted in HANDOFF (1.98 us).

**Thermal neutrons cannot leak out of the shield.** Sigma_s(thermal) = 7.25/cm,
mubar = 0.650, Sigma_tr = 2.54/cm, D = 0.132 cm, so the diffusion length is

    L = sqrt(D/Sigma_a) = 0.238 cm

The MC's captures occur at a median depth of **5.29 cm** (mean 5.74 cm), leaving
8.7 cm to the exit face: exp(-8.7/L) = 1e-16. This is the single most important
structural fact about the experiment — **the shield emits gammas, not thermal
neutrons.** It also kills candidate (c) in §5: halving the shield boron loading
only doubles L to 0.33 cm, which changes nothing.

### 1.4 Capture competition H vs B-10 — energy independent

Both are 1/v absorbers, so Sigma_H/Sigma_B10 is a pure number:

| material | Sigma_H(th) | Sigma_B10(th) | B:H | H branch |
|---|---|---|---|---|
| B-HDPE shield | 0.02859 /cm | 2.3009 /cm | **80.5 : 1** | **1.23 %** |
| tile (rho 1.05) | 0.01743 /cm | 0.9664 /cm | 55.4 : 1 | 1.77 % |
| tile (rho 1.03) | 0.01743 /cm | 0.9480 /cm | 54.4 : 1 | 1.81 % |

**Consistency check on the MC.** The MC reports B-10 capture in the shield =
71.8 % of emitted neutrons. Every one of those neutrons passed the same 1/v
competition, so the expected H-capture fate is

    71.8 % x (1/80.5) = 0.892 %

against the MC's quoted **0.90 %** for "H capture (shield + wood)". The agreement
is exact, and it also tells us the wood contributes essentially nothing (thermal
neutrons never get there — §1.3). The 2026-08-24 bug fix in NOTES.md is
confirmed: the old flat 2 %-per-H-scatter absorption was killing ~23 % of
neutrons; the 1/v treatment gives 0.9 %, which is what the cross sections demand.

### 1.5 Gamma yields and propagation to the tile

Per cone-emitted neutron:

* 478 keV (7Li*): 0.718 x 0.94 = **0.6749**
* 2223 keV (H(n,gamma)D): **0.0090**
* ratio at birth: **1.33 %**

Both gammas are born from the *same* thermal/epithermal capture distribution in
the same material (both absorbers are 1/v), so the source depth profile and the
solid angle to the tile **cancel exactly** in the ratio. What does not cancel is
attenuation and detection.

Klein-Nishina, electron densities n_e(B-HDPE) = 3.430e23, n_e(PVT) = 3.364e23
(Compton only; photoelectric and pair production are <1 % in C/H here):

| Eg | sigma_KN | mu(B-HDPE) | mu(PVT) | mu(Fe) [XCOM] | Compton edge | p(interact, 1 cm PVT) |
|---|---|---|---|---|---|---|
| 478 keV | 0.2946 b/e | 0.1011 /cm | 0.0991 /cm | 0.659 /cm | 311.5 keV | 9.44 % |
| 2223 keV | 0.1377 b/e | 0.0472 /cm | 0.0463 /cm | 0.324 /cm | 1993.8 keV | 4.53 % |

Uncollided escape, averaging exp(-mu x (14 cm - z_cap)) over the MC's own capture
depths, then 1 cm of steel:

    478 keV : <exp(-mu dz)> = 0.461  x 0.518 (Fe) = 0.239
    2223 keV: <exp(-mu dz)> = 0.686  x 0.723 (Fe) = 0.496

Detection: the charge cut 2.31 V.ns = 312 keVee (at 135 keVee/V.ns) is the 478 keV
Compton edge. The fraction of 2223 keV Compton deposits above it is 0.885; the
fraction of 478 keV deposits above the 47 keVee trigger is 0.831. Hence

    N(>edge)/N(<edge) = [0.0090 x 0.496 x 0.0453 x 0.885]
                      / [0.6749 x 0.239 x 0.0944 x 0.831]  =  1.42 %

Buildup (scattered photons) favours the 478 keV line — B ~ 1+mu*d gives 1.9 vs
1.4 — so the honest prediction is **~1.0-1.4 %**.

**The claimed data bound does not reproduce.** Running the repo's own ambient
subtraction (`alphareport.ambient_scale` against the dedicated background run
`20260817_sample02_background`, scale 0.334) on the merged run
(`checks/h_data_check4.py`, `checks/h_data_check5.py`):

| delayed 0.5-15 us, unclipped | beam | ambient expected | **net** |
|---|---|---|---|
| below 2.306 V.ns | 27,623 | 13,030 | **+14,593 +- 179** |
| above 2.306 V.ns | 8,868 | 699 | **+8,169 +- 95** |

The below-edge net (14,593) is close to the brief's "13,987", so the quoted
number is a net; but the above-edge net is **8,169 +- 95**, not 4 +- 2, and the
0.1 V.ns scan across the edge (2.0 -> 3.5 V.ns) is a smooth, structureless
decline with **no break at 2.31 V.ns**. `fig_integral` in `alphareport.py` plots
only 0-3.0 V.ns; a count taken inside that frame sees 4,440 of the 8,169. This is
exactly the "a spectrum can be truncated at the top as well as the bottom" trap
that HANDOFF.md §5 already warns about, and I suspect the 4 +- 2 came from a
narrow slice rather than from the region above the edge.

Consequences:

* The MC's 0.90 % H capture is **consistent** with the data. It predicts ~1 % of
  the 478 keV signal above the edge; the data has 56 % there, so the H gammas are
  buried, not excluded.
* The 56 % is *not* explained by H(n,gamma) either — nor by Fe(n,gamma). The 7.6 MeV
  cascade from the 1 cm steel is closer to the tile (2.5 vs 9.5 cm, a x14 solid
  angle gain) and escapes its own 1 cm far better (0.94), but the yield is only
  0.10 % of emitted (x2 gammas) and 7.6 MeV photons interact in 1 cm of PVT with
  probability 2.1 %: the net is **3.7 % of the 478 keV rate**
  (`h_hydrogen.py` §11) — an order of magnitude short of 56 %. The
  simplest explanation is that **2.31 V.ns is not the 478 keV Compton edge**: the
  tile energy scale rests entirely on the assumed 85 keVee alpha light
  (`ALPHA_KEVEE = 85.0`, `TILE_KEVEE_PER_VNS = 134.9`), which HANDOFF §7b states
  is not measured. If the true scale were ~3x smaller, 2.31 V.ns would be ~100
  keVee and the smooth continuum is unremarkable.
* **Recommendation:** do not quote a hydrogen-capture bound from this comparison
  until the tile scale is established.

### 1.6 Where the shield captures actually happen

Capture-during-slowing-down integral I = Int Sigma_a/(xibar Sigma_t) du over the
full lethargy range gives I = 1.010, so

    P(captured before reaching thermal) = 1 - exp(-I) = 0.64

**64 % of shield captures are epithermal**, and 92 % of that comes from the last
three decades (25 eV -> thermal). This is why the MC's capture-time median
(1.34 us) is shorter than thermalisation (2.3 us) plus the thermal dwell
(1.95 us): most neutrons never complete the trip. It also means the shield-gamma
arrival profile is set by the *slowing-down* clock, i.e. by hydrogen, more than
by the boron dwell time.

---

## 2. Hydrogen in the tile

n_H = 5.2514e22, n_C = 4.7262e22, n_B10 = 2.517e20 cm^-3.

At thermal:

    Sigma_sH  = 5.2514e22 x 82 b     = 4.306  /cm
    Sigma_sC  = 4.7262e22 x 4.75 b   = 0.2245 /cm
    Sigma_B10 = 2.517e20  x 3840 b   = 0.9664 /cm
    Sigma_H(n,g) = 5.2514e22 x 0.332 b = 0.01743 /cm
    Sigma_tot = 5.514 /cm

* **H share of tile captures = 0.01743/(0.01743+0.9664) = 1.77 %.**
  Those give a 2223 keV gamma and **no alpha**, so they are invisible to the PSD
  alpha tag and contribute only a Compton continuum. At 1.77 % of ~3,300 captures
  that is ~58 events in the whole run — unobservable.
* Pure-capture tile lifetime 1/(Sigma_a v_th) = **4.62 us** (4.71 us at rho 1.03);
  measured die-away 3.44-3.6 us after leakage. Consistent.
* **Thermal capture probability in 1 cm** (single pass, with scattering
  competition): p_int = 1 - exp(-5.514) = 0.996 and
  p_cap = (Sigma_B10/Sigma_tot) p_int = **17.5 %**.
  Note that a naive 1 - exp(-Sigma_B10 x 1 cm) = 62 % is 3.5x too large — the
  neutron is far more likely to scatter first. (The old 1-D MC used the naive
  form; that is the origin of its 0.99 % single-pass efficiency.)
* **Epithermal**, single pass, 1 cm:

  | E | Sigma_cap | Sigma_scat | p_cap | p_scat |
  |---|---|---|---|---|
  | 1 eV | 0.1537 | 1.783 | 6.8 % | 78.8 % |
  | 100 eV | 0.0154 | 1.293 | 0.86 % | 72.1 % |
  | 10 keV | 0.0015 | 1.238 | 0.09 % | 71.0 % |
  | 100 keV | 0.0005 | 0.892 | 0.03 % | 59.0 % |

  An epithermal neutron is **80-100x more likely to scatter than to capture on
  its first pass** — and every one of those scatters is on hydrogen (H carries
  89 % of Sigma_s at 1 eV). That is the mechanism that turns a single-pass
  attenuator into a capture detector, and it is the missing physics in §5.

---

## 3. Hydrogen scattering in the tile, and what the trigger actually sees

### 3.1 Birks light output for protons

L(E) = Int_0^E dE' / (1 + kB (dE/dx)(E')), with **kB = 0.0125 g cm^-2 MeV^-1**
and dE/dx a PSTAR-like proton mass stopping power table for polyvinyltoluene
(peak 805 MeV cm^2/g at ~60 keV; 792 at 100 keV, 642 at 200 keV, 420 at 500 keV,
256 at 1 MeV, 157 at 2 MeV). Quoting stopping power in mass units and kB in
g cm^-2 MeV^-1 makes the result density-independent.

| E_p [keV] | dE/dx [MeV cm^2/g] | **L [keVee]** | L/E |
|---|---|---|---|
| 50 | 794 | 6.1 | 0.122 |
| **100** | 792 | **10.6** | 0.106 |
| **200** | 642 | **20.7** | 0.103 |
| 300 | 544 | 32.7 | 0.109 |
| **500** | 420 | **61.7** | 0.123 |
| 750 | 328 | 106.0 | 0.141 |
| **1000** | 256 | **160.4** | 0.160 |
| 1500 | 194 | 293.6 | 0.196 |
| 2000 | 157 | 451.3 | 0.226 |
| 2450 | 139 | 609.8 | 0.249 |

Sanity check on kB with the same machinery (`h_hydrogen.py` §11): with an
ASTAR-like alpha stopping table for PVT (1790 MeV cm^2/g at 1.47 MeV) and 7Li
treated as a Z_eff = 2.6 ion at the same velocity as a proton of E/7, the boron
products give **57.7 + 14.4 = 72 keVee**, against the adopted 85 keVee — i.e.
kB = 0.0125 reproduces the capture line to ~15 % with no tuning. (Read the
residual as the uncertainty on the heavy-ion stopping powers, not on kB.)

**The threshold in proton energy.** Trigger 25 mV = 0.35 V.ns = 47 keVee:

    L(E_p) = 47 keVee   =>   E_p = 404 keV

The observed turn-on in the data is a little lower (min charge 0.226 V.ns =
31 keVee => E_p = 283 keV). Either way the MC's `E_THR = 5e4` (50 keV proton) is
**8x too low**: it corresponds to 6 keVee, roughly a quarter of a photoelectron.

The C2 ceiling (382 mV, clip flag) sits at ~5 V.ns = 675 keVee, i.e. E_p =
2.6 MeV — above the 2.45 MeV kinematic maximum (L(2.45 MeV) = 610 keVee = 4.5
V.ns). **Clipping does not remove proton recoils**; it removes something else
(§5, footnote).

### 3.2 Carbon recoils

Maximum recoil fraction 4A/(A+1)^2 = **0.284**, so E_C^max = 696 keV from a
2.45 MeV neutron. A carbon ion at that energy has Z_eff^2 x (velocity-scaled)
dE/dx of order 10^4 MeV cm^2/g, i.e. kB dE/dx ~ 100:

| E_C | L | L/E |
|---|---|---|
| 100 keV | ~2.1 keVee | 0.021 |
| 300 keV | ~4.2 keVee | 0.014 |
| 696 keV (max) | ~7.5 keVee | 0.011 |

**Carbon recoils are 6-20x below the trigger and can never fire it.** Any
"visible scatter" term must be hydrogen only. The MC's `f_C` branch (which lets
C recoils above 50 keV count as visible) is wrong and should be deleted.

### 3.3 The MC visible-scatter count, recomputed

Using the MC's own arrival sample (18,840 tile hits per 1e7 cone-emitted),
first-interaction branching Sigma_i/Sigma_tot x (1-exp(-Sigma_tot x 1 cm)), and a
flat proton recoil spectrum f_H = 1 - E_thr/E_n:

| threshold | visible scatters / 1e7 | ratio to 65.5 captures |
|---|---|---|
| 50 keV proton (MC as written, incl. C) | **3527.2** | **53.8 : 1** |
| 404 keV proton (= 47 keVee trigger), H only | **1406.1** | **21.5 : 1** |
| 283 keV proton (observed turn-on), H only | 1609.0 | 24.6 : 1 |

**Correcting the threshold alone buys a factor 2.5.**

---

## 4. What hydrogen does to the timing

**Shield-capture gammas (1-2 us).** The 478 keV gammas travel at c, so their
arrival profile at the tile *is* the shield's capture-time profile, which is
hydrogen's slowing-down clock plus the boron dwell. From the MC:
median 1.34 us, mean 1.96 us, with 64 % of captures happening epithermal
(§1.6). Without hydrogen there is no such profile: a pure-carbon shield would
need 117 collisions (18.4/0.1578) and hundreds of microseconds. The composite fit in
`composite_timing.py` already uses this shape with no free shape parameter and
gets chi2/ndf = 43/39, splitting the delayed excess 83 % shield-gamma / 17 %
tile — i.e. **most of the measured 3.6 us die-away is hydrogen-set gamma timing,
not tile capture.**

**Tile dwell (3-5 us).** Hydrogen in the tile thermalises whatever arrives and
holds it there; 1/(Sigma_a v) = 4.62 us shortened by leakage from a 1 cm slab to
~3-3.5 us. This is a property of the tile only and is independent of the source
distance — the MC reproduces that independence, which is a good internal check.

**Room return (hundreds of us to ms).** Hydrogen in concrete floor/walls
(~1 wt% H, plus Ca, Si, O) moderates the ~27 % of neutrons that leave the shield
region (20 % backscattered + 7 % transmitted-but-miss-tile), and the ~97 % of an
isotropic source that never enters the 20 deg cone at all. A concrete room's
thermal die-away is **0.3-2 ms**, i.e. 100-500x the beam period of 50 us, so
room-return neutrons **pile up into a flat, beam-uncorrelated floor**, constant
in tau across the whole period.

**How to tell them apart in the data.** Three signatures:

1. *Time shape.* Local moderation is exponential with lambda = 3-5 us; room
   return is flat within the 50 us period. Any component that does not decay
   inside the record is room return (or ambient).
2. *Period scan.* Lengthen the period to 500 us or 1 ms. The 3.6 us component is
   untouched; the flat floor rises or falls with the duty cycle in a way that
   measures the room lifetime directly. The 20260731 500 us run and the
   20260818 sample-11 500 us run already exist for this.
3. *Beam-off subtraction.* The 20260817 background run gives the truly ambient
   part; what is left flat after subtracting it is room return.

The measured post-gate lambda = 3.6 us is squarely in the local-moderation
regime. **There is no ms-scale room-return excess visible in the current data**,
which is the argument that closes §5 below.

---

## 5. The puzzle: 54:1 predicted, 1.3:1 measured

### 5.1 The ledger

| step | ratio recoil-triggers : tile captures |
|---|---|
| MC as written (50 keV proton, C recoils included, single-pass tile) | **53.8 : 1** |
| (a) real trigger threshold: 47 keVee => 404 keV proton; H only | **21.5 : 1** |
| (b) tile treated as a moderator, not an attenuator (x5.6 on captures) | **3.8 : 1** |
| data as quoted in the brief | 1.3 : 1 |
| (d) data corrected for the lost half of the in-gate window | **2.6 : 1** |

The remaining 3.8 vs 2.6 is a factor 1.5, comfortably inside the joint
uncertainty of the tile boron loading (NOTES.md: lambda says 1.4 wt%, not 2.0 —
that alone moves captures by ~30 %), the assumed 40 % PSD tag efficiency, and the
arrival spectrum.

### 5.2 Candidate by candidate

**(a) Trigger threshold on quenched proton light — real, factor 2.5.**
Already in §3. The MC cut at 50 keV *proton energy* corresponds to 6 keVee; the
real cut is 47 keVee = 404 keV proton. Dropping the carbon term (§3.2) is part of
the same correction. 3527 -> 1406 visible scatters. This is the single easiest
fix and it should go into `sampleB_mc3d.py`.

**(b) The tile is a moderator — real, factor ~5.6, and it is a HYDROGEN effect.**
This is the big one and it is not "missing room return", it is missing physics
*inside the tile*. The 3-D MC scores only the first interaction: a neutron whose
first interaction is an elastic scatter is booked as "scatter" and discarded. But
after that scatter it is still in the tile, now at half the energy, with
Sigma_s ~ 1.3-5.5 /cm — it random-walks, thermalises in a few H collisions, and
captures on the 4.6 us clock. The capture budget by arrival group makes this
obvious:

| group | arrivals /1e7 | share | single-pass captures | <p_cap> |
|---|---|---|---|---|
| fast > 100 keV | 15,256 | 81.0 % | 1.76 | 0.00012 |
| epithermal | 3,308 | 17.6 % | 30.12 | 0.0091 |
| thermal < 1 eV | 276 | 1.5 % | 33.65 | 0.122 |

81 % of the arriving flux contributes 2.7 % of the captures *because the model
does not let it slow down*. NOTES.md already measured what happens when you do:
the time-resolved 3x3x1 tile MC gives **eps = 1.93-1.96 % captures per arriving
neutron**, against 65.5/18,840 = **0.348 %** single-pass here — a factor **5.6**.
(The 1-D note quotes "0.99 % -> 1.96 %, roughly doubled", but its 0.99 % used the
naive 1 - exp(-Sigma_B10 t) with no scattering competition, which is 3x too
generous; corrected, the 1-D single pass is ~0.33 %, matching the 3-D 0.348 %,
and the true boost is 5.6, not 2.)

Applied: 18,840 x 0.0195 = **367 captures** per 1e7, so 1406/367 = **3.8 : 1**.

There is also a physical subtlety that makes the "ratio" less than a ratio: a
fast neutron that produces a visible proton recoil in the gate and *then*
thermalises and captures 3 us later contributes to **both** columns. The same
neutron is a prompt trigger and a delayed capture. At 81 % fast arrivals this is
the normal case, and it means the true ratio is bounded below by the survival
probability, not by independent populations.

**(c) Source spectrum / cone / shield boron loading — small, and testable.**
The MC's 20 deg cone means "per emitted" = per cone-emitted; the ratio is
insensitive to the cone opening because captures and recoils both come from the
transmitted beam. Shield boron loading is *not* a lever on the thermal flux at
the tile: at 5 wt% the thermal diffusion length is 0.238 cm, and halving the
loading only takes it to 0.33 cm — thermal neutrons still cannot escape 8.7 cm of
shield (§1.3). Loading changes the *gamma* yield and the shield die-away (1.95 us
at 5 %, 3.9 us at 2.5 %), which is testable against the composite fit, but not
the recoil:capture ratio. **Rank: low.**

**(d) The in-gate window — real, factor 2, and it is an analysis bug.**
Found in `checks/h_data_check6.py` / `checks/h_data_check8.py`. In the merged run the
"in-gate" definition is `tau_rise > 0 & tau_end < 0`. The `tau_rise`
distribution of the events that pass has a **hard edge at 0.392 us** — nothing
below it, ever, in a 0.958 us gate. The cause is not the 450 ns TTL correction
(that is applied correctly) and not clipping: it is that `tau_end` is only filled
when the gate's falling edge is cleanly measured (`extract.py`, lines 185-188),
and for pulses in the first 0.4 us of the gate it is not, so `tau_end` is NaN and
the event is silently dropped.

| selection | all | unclipped | tau_rise range |
|---|---|---|---|
| kept (`tau_rise>0 & tau_end<0`) | 5,730 | **4,369** | 0.392 - 0.964 us |
| dropped (`tau_rise` in gate, `tau_end` NaN) | 5,603 | **4,166** | 0.000 - 0.415 us |
| **true in-gate** | **11,333** | **8,535** | |

So **the quoted 4,369 is 51 % of the real in-gate population**. Corrected:
0.141 in-gate triggers per beam pulse instead of 0.072, and

    recoil : capture  =  8535 / 3300  =  2.6 : 1   (not 1.3 : 1)

The fix is to define in-gate as `0 < tau_rise < gate_w` and not require a finite
`tau_end`; or to reconstruct `gate_w` from the run rather than per-record.
(Clipping is a separate 24 % loss but it is flat in tau_rise and, per §3.1,
cannot be removing proton recoils: the clip ceiling at 675 keVee is above the
610 keVee that a full-energy 2.45 MeV proton produces. The clipped population is
something harder — Fe(n,gamma) cascade gammas from the steel and cosmics are the
obvious candidates.)

**(e) Other things found in the code.**
* `f_C` in `sampleB_mc3d.py` lets carbon recoils above 50 keV count as visible.
  They are quenched to 2-8 keVee and never trigger. Remove the term (it is ~10 %
  of the 3527).
* `E_THR = 5e4` is documented as "recoil deposit > 50 keV (proton energy, NOT
  electron-equivalent)" — correct as documented, but it is compared with a
  measurement whose threshold is in keVee. Replace with a light-output threshold.
* The MC scores only the *first* interaction; sub-threshold first scatters that
  are followed by a visible second scatter are lost. This pushes the visible
  count slightly up, i.e. against the resolution, but it is a ~10 % effect.
* The tile density inconsistency (1.05 in the MC vs 1.03 quoted) and the B-HDPE
  H:C = 2.126 vs 2.000 are both ~2-6 % bookkeeping.

**(b') Room return — not needed, and disfavoured.**
For completeness, what it *would* take. With the corrected threshold, reaching a
given ratio purely by adding thermal flux (<p_cap> = 0.122 per thermal arrival)
requires:

| target ratio | captures needed | extra thermal arrivals | x present thermal | thermal share of arrivals |
|---|---|---|---|---|
| 2.3 : 1 | 611 | +4,476 | 16x | 20 % |
| 1.3 : 1 | 1,082 | +8,333 | 30x | 32 % |

A 16-30x boost of the thermal flux with the tile's own moderation still switched
off. But §4 says room return arrives with a 0.3-2 ms lifetime, i.e. as a flat
floor, and the delayed signal is a clean 3.6 us exponential. **Room return cannot
supply captures inside the die-away window.** Local moderation in the tile
(hypothesis b) supplies exactly the same captures on exactly the right clock.
Rank room return last as an explanation of the ratio; it remains the right
explanation of any flat floor, and is worth measuring separately (§4).

### 5.3 Ranking

1. **(b) tile moderation** — factor 5.6, correct physics, correct timing. Dominant.
2. **(d) the lost in-gate window** — factor 2, a straightforward analysis bug.
3. **(a) the light threshold** — factor 2.5 on the MC side.
4. (e) carbon recoils, first-interaction-only scoring — ~10-20 % each.
5. (c) source cone / shield loading — not a lever on this ratio.
6. (b') room return — wrong timescale; not the explanation.

### 5.4 The one change that settles it

**Put the time-resolved tile back into the 3-D MC.** Replace the single-pass
first-interaction scoring of `sampleB_mc3d.py` with the 3-D random walk that
already exists (the `tile_dieaway_001` machinery in `pns.ipynb`), fed by the 3-D
arrival list, and score with a **light-output threshold** L(E_p) > 47 keVee
instead of `E_THR = 5e4`, H only. That single change replaces the two largest
factors (5.6 and 2.5) with a first-principles calculation and predicts the ratio,
the die-away, and the absolute rate at once.

**The one measurement:** re-run the in-gate count with the corrected definition
(`0 < tau_rise < gate_w`, no `tau_end` requirement) — five lines, no new data —
and take a longer-period run (500 us or 1 ms) to separate the flat room-return
floor from the 3.6 us die-away. If after both the ratio is still below ~2 : 1,
the residual is the tile boron loading, and the die-away already says it is
1.4 wt%, not 2.0.

---

## 6. Files

| file | what it does |
|---|---|
| `h_hydrogen.py` | all of §1-§3 and §5.1-5.2; reads the MC cache read-only |
| `checks/h_data_check.py` | clip level, in-gate/delayed counts, charge spectra |
| `checks/h_data_check2.py` | in-gate vs ambient floor, tau_end profile |
| `checks/h_data_check3.py` | above/below-edge split vs tau_end |
| `checks/h_data_check4.py` | ambient-subtracted above/below-edge net (§1.5) |
| `checks/h_data_check5.py` | edge scan, alpha-tagged split (§1.5) |
| `checks/h_data_check6.py` | the tau_rise hard edge at 0.392 us |
| `checks/h_data_check7.py` | the same check across every cached run |
| `checks/h_data_check8.py` | the lost in-gate population (§5.2d) |

`h_hydrogen.py` runs with `energy-deposits-env/bin/python` from this directory;
the `checks/h_data_check*.py` scripts run with `/Users/virgolaema/Software/neutron-env/bin/python`
and `PYTHONPATH=/Users/virgolaema/Software/3det/pns-waveform-ana`.
