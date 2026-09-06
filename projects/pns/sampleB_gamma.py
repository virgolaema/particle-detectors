"""sampleB_gamma.py — what the boron-loaded tile actually RECORDS after the gate.

Extends the neutron-only MC (`sampleB_mc3d.py`, untouched, cache untouched) with

  1. a vectorised re-run of the same 3-D transport that RECORDS CAPTURE VERTICES
     (x, y, z, t, isotope) for B-10 and H in the shield/wood and Fe in the steel;
  2. a photon transport (source vertex -> tile) with a next-event estimator and
     up to 5 Compton scatters in the shield (i.e. buildup is included, not just
     the uncollided beam);
  3. a full in-tile photon cascade (Compton -> Compton -> ..., pair production,
     annihilation quanta) with Compton-ELECTRON ESCAPE from the 1 cm slab;
  4. a time-resolved neutron RANDOM WALK inside the tile (the tile as a
     MODERATOR, not an attenuator — HYDROGEN.md §5.2b) giving tile captures,
     their times, and the in-gate proton-recoil triggers with a LIGHT threshold;
  5. gate folding, charge conversion (113 keVee / V.ns, measured), resolution,
     threshold (0.35 V.ns) and clipping (4.4 V.ns), and a comparison with the
     measured beam-correlated delayed excess of runs 20260814 and 20260820.

Run:  cd .../projects/pns && ../../energy-deposits-env/bin/python sampleB_gamma.py
Caches it writes (all new, nothing existing is touched):
    _gam_vertices.npz   capture vertices + fates from the new transport run
    _gam_tilewalk.npz   tile random-walk results
    _gam_data.npz       data spectra/lifetimes pulled from pns-waveform-ana
"""
import math, os, subprocess, sys, time as time_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

# ═══════════════════════════════════════════════════════════════════════════
#  §0  CONFIGURATION — every number that is an assumption lives here
# ═══════════════════════════════════════════════════════════════════════════
N_NEUTRON   = int(os.environ.get('GAM_N', 4_000_000))   # new transport run (the 10 M cache is NOT touched)
SEED        = 20260906

# --- measured tile calibration (from the data; NOT derived here) ------------
KEVEE_PER_VNS = 113.0        # 478 keV Compton edge 311.5 keVee at 2.75 V.ns
Q_ALPHA       = 0.63         # V.ns, alpha+7Li capture line  (= 71.2 keVee)
S_ALPHA       = 0.10         # V.ns, measured width of that line
Q_THR         = 0.35         # V.ns trigger threshold        (= 39.6 keVee)
Q_CLIP_14     = 4.4          # V.ns ceiling, run 20260814 (cast tile, 382 mV)
Q_CLIP_20     = 9.5          # V.ns ceiling, run 20260820 (3-D print, 751 mV)
RES_E0, RES_S0 = 71.2, 0.10  # sigma/E = 0.10 at 71.2 keVee, scaling 1/sqrt(E)
GATE_US       = 0.958        # measured gate width
TAU_LO, TAU_HI = 0.5, 15.0   # delayed window, us after gate end
TAU_L0, TAU_L1 = 15.0, 18.5  # late window used for the flat subtraction

# --- measured anchors (data, given; not re-derived) -------------------------
N_REC          = 60356       # gate_ok records in run 20260814 ("beam pulses")
DATA_ALPHA     = 3300.0      # tile captures, tag-efficiency corrected
DATA_INGATE    = 8534.0      # in-gate recoil triggers (corrected window)
DATA_LAMBDA = {              # exp+flat die-away fits, us
    'alpha ROI'        : (3.33, 0.11),
    '478 Compton 1-2.5': (4.06, 0.14),
    'above edge 3-4.4' : (3.84, 0.19),
    'clipped >4.4'     : (3.22, 0.11)}

# --- gamma yields -----------------------------------------------------------
Y_B478  = 0.94              # 478 keV photons per B-10(n,alpha)
E_B478  = 0.478             # MeV
Y_H2223 = 1.00
E_H2223 = 2.2233
E_FE_TOT = 7.646            # MeV, Fe-56(n,g) Q value
FE_SINGLE_FRAC = 0.50       # 50 % of captures: one 7.638 MeV photon
                            # 50 %: a 3-photon cascade sharing 7.646 MeV
                            # -> mean multiplicity 2.0 (real ~2.1). ASSUMPTION.

# --- photon-transport controls ---------------------------------------------
N_SRC_B10 = 120_000         # B-10 source photons actually tracked (weighted)
N_REP_H   = 6               # repetitions per H vertex
N_REP_FE  = 40              # repetitions per Fe capture
NE_SAMP   = 3               # next-event sample points in the tile per collision
MAX_SCAT_SHIELD = 5         # Compton generations tracked in the shield
E_PHOT_CUT = 0.010          # MeV, photon tracking cut-off

# ═══════════════════════════════════════════════════════════════════════════
#  §1  GEOMETRY AND NEUTRON MATERIALS  (copied verbatim from sampleB_mc3d.py)
# ═══════════════════════════════════════════════════════════════════════════
from lib_neutrons import get_endf_cross_sections

E0_eV, N_A = 2.45e6, 6.022e23
PE_CM, WOOD_CM, AIR1_CM, STEEL_CM, AIR2_CM = 14.0, 1.0, 5.0, 1.0, 2.0
D_TILE, HALF_TILE = 23.0, 1.5
TILE_T = 1.0                                   # tile thickness, cm
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
_lE0, _dlE = _lE[0], _lE[1] - _lE[0]
NE_GRID = _E.size


def _lookup(tab2d, lay, lgE):
    """tab2d[nlay, NE] linear-in-log-E lookup, vectorised over particles."""
    u = np.clip((lgE - _lE0) / _dlE, 0.0, NE_GRID - 1.0000001)
    i0 = u.astype(np.int64)
    fr = u - i0
    return tab2d[lay, i0] * (1.0 - fr) + tab2d[lay, i0 + 1] * fr


# layer tables: 0 bhdpe, 1 wood, 2 air, 3 steel, 4 air
_LZ0 = np.array([0.0, 14.0, 15.0, 20.0, 21.0])
_LZ1 = np.array([14.0, 15.0, 20.0, 21.0, 23.0])
_LA  = np.array([12.0, 12.0, 1.0, 55.85, 1.0])
NLAY = 5
_SS = np.zeros((NLAY, NE_GRID))
_FH = np.zeros((NLAY, NE_GRID))
_SS[0] = (n_H_bh * _sH + n_C_bh * _sC) * 1e-24
_FH[0] = n_H_bh * _sH * 1e-24 / _SS[0]
_SS[1] = (n_H_wd * _sH + n_Ceq_wd * _sC) * 1e-24
_FH[1] = n_H_wd * _sH * 1e-24 / _SS[1]
_SS[3] = (n_Fe * _sFe) * 1e-24
# thermal absorption coefficient per layer  Sa(E) = CA * sqrt(E_TH/E)
_CA = np.array([(n_B10 * S_B10_TH + n_H_bh * S_HG_TH) * 1e-24,
                (n_H_wd * S_HG_TH) * 1e-24, 0.0,
                (n_Fe * S_FE_TH) * 1e-24, 0.0])
_FB10 = np.array([n_B10 * S_B10_TH / (n_B10 * S_B10_TH + n_H_bh * S_HG_TH),
                  0.0, 0.0, 0.0, 0.0])       # rest of the absorption is on H
_ISFE = np.array([0, 0, 0, 1, 0], dtype=bool)


# ═══════════════════════════════════════════════════════════════════════════
#  §2  VECTORISED NEUTRON TRANSPORT THAT RECORDS CAPTURE VERTICES
#      Physics is identical to sampleB_mc3d.run(); only the loop is rewritten
#      (per-particle Python -> numpy over the whole population) and captures
#      are stored instead of being counted.
# ═══════════════════════════════════════════════════════════════════════════
def _rotate(ux, uy, uz, c, rng):
    s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    phi = 2.0 * np.pi * rng.random(ux.size)
    cp, sp = np.cos(phi), np.sin(phi)
    pol = np.abs(uz) >= 0.999999
    den = np.sqrt(np.maximum(1.0 - uz * uz, 1e-30))
    ax, ay, az = uy / den, -ux / den, np.zeros_like(ux)
    bx, by, bz = uz * ux / den, uz * uy / den, -den
    ax = np.where(pol, 1.0, ax); ay = np.where(pol, 0.0, ay); az = np.where(pol, 0.0, az)
    bx = np.where(pol, 0.0, bx); by = np.where(pol, uz, by); bz = np.where(pol, 0.0, bz)
    return (c * ux + s * (cp * ax + sp * bx),
            c * uy + s * (cp * ay + sp * by),
            c * uz + s * (cp * az + sp * bz))


def transport(N, seed, chunk=200_000, cone_deg=20.0):
    rng = np.random.default_rng(seed)
    MU0 = math.cos(math.radians(cone_deg))
    fates = dict(tran=0, back=0, B10=0, H=0, Fe=0)
    vB, vH, vF = [], [], []
    t0 = time_module.time()
    done = 0
    while done < N:
        n = min(chunk, N - done); done += n
        mu = MU0 + (1 - MU0) * rng.random(n)
        phi = 2 * np.pi * rng.random(n)
        sn = np.sqrt(1 - mu * mu)
        ux, uy, uz = sn * np.cos(phi), sn * np.sin(phi), mu
        E = np.full(n, E0_eV); t = np.zeros(n)
        x = np.zeros(n); y = np.zeros(n); z = np.zeros(n)
        it = 0
        while E.size and it < 20000:
            it += 1
            lay = np.clip(np.searchsorted(_LZ1, z + 1e-12, side='right'), 0, NLAY - 1)
            lgE = np.log(np.maximum(E, 0.025))
            Ss = _lookup(_SS, lay, lgE)
            Sa = _CA[lay] * np.sqrt(E_TH / np.maximum(E, E_TH))
            St = Ss + Sa
            step = np.where(St > 1e-12, -np.log(rng.random(E.size)) / np.maximum(St, 1e-300), 1e30)
            df = np.where(uz > 0, (_LZ1[lay] - z) / np.where(uz > 0, uz, 1),
                          np.where(uz < 0, (z - _LZ0[lay]) / np.where(uz < 0, -uz, 1), 1e30))
            v = 1.3831e6 * np.sqrt(np.maximum(E, 1e-6))
            cross = step >= df
            # --- boundary crossings ------------------------------------------
            d = np.where(cross, df, step)
            t = t + d / v
            x = x + d * ux; y = y + d * uy; z = z + d * uz
            z = np.where(cross, z + 1e-9 * np.sign(uz), z)
            gone_f = cross & (z >= D_TILE)
            gone_b = cross & (z <= 0.0)
            fates['tran'] += int(gone_f.sum()); fates['back'] += int(gone_b.sum())
            # --- collisions ---------------------------------------------------
            coll = ~cross
            absb = coll & (rng.random(E.size) * np.maximum(St, 1e-300) < Sa)
            if absb.any():
                isb10 = rng.random(E.size) < _FB10[lay]
                bmask = absb & isb10 & ~_ISFE[lay]
                fmask = absb & _ISFE[lay]
                hmask = absb & ~bmask & ~fmask
                if bmask.any():
                    vB.append(np.column_stack([x[bmask], y[bmask], z[bmask], t[bmask]]).astype(np.float32))
                if hmask.any():
                    vH.append(np.column_stack([x[hmask], y[hmask], z[hmask], t[hmask]]).astype(np.float32))
                if fmask.any():
                    vF.append(np.column_stack([x[fmask], y[fmask], z[fmask], t[fmask]]).astype(np.float32))
                fates['B10'] += int(bmask.sum()); fates['H'] += int(hmask.sum())
                fates['Fe'] += int(fmask.sum())
            scat = coll & ~absb
            if scat.any():
                onH = rng.random(E.size) < _lookup(_FH, lay, lgE)
                mH = scat & onH
                mA = scat & ~onH
                if mH.any():
                    c = np.sqrt(rng.random(int(mH.sum())))
                    E[mH] = np.maximum(E[mH] * c * c, 0.025)
                    ux[mH], uy[mH], uz[mH] = _rotate(ux[mH], uy[mH], uz[mH], c, rng)
                if mA.any():
                    A = _LA[lay][mA]
                    cc = 2 * rng.random(int(mA.sum())) - 1
                    dn = 1 + 2 * A * cc + A * A
                    E[mA] = np.maximum(E[mA] * dn / (A + 1) ** 2, 0.025)
                    ux[mA], uy[mA], uz[mA] = _rotate(ux[mA], uy[mA], uz[mA],
                                                     (1 + A * cc) / np.sqrt(dn), rng)
            alive = ~(gone_f | gone_b | absb)
            if not alive.all():
                E = E[alive]; t = t[alive]; x = x[alive]; y = y[alive]; z = z[alive]
                ux = ux[alive]; uy = uy[alive]; uz = uz[alive]
    print(f"  transport: {N:,} n in {time_module.time()-t0:.0f}s  fates%: "
          + "  ".join(f"{k} {100*v/N:.3f}" for k, v in fates.items()))
    cat = lambda L: (np.concatenate(L) if L else np.zeros((0, 4), np.float32))
    return cat(vB), cat(vH), cat(vF), fates


VCACHE = os.environ.get('GAM_VCACHE', '_gam_vertices.npz')
if os.path.exists(VCACHE):
    _v = np.load(VCACHE, allow_pickle=True)
    vB, vH, vF, fates, N_NEUTRON = _v['vB'], _v['vH'], _v['vF'], _v['fates'].item(), int(_v['N'])
    print(f"loaded {VCACHE}: {vB.shape[0]:,} B-10 / {vH.shape[0]:,} H / {vF.shape[0]:,} Fe "
          f"capture vertices from {N_NEUTRON:,} neutrons")
else:
    print(f"running new 3-D transport, {N_NEUTRON:,} neutrons (recording capture vertices)")
    vB, vH, vF, fates = transport(N_NEUTRON, SEED)
    if vB.shape[0] > 600_000:            # keep the cache small; weights come
        _k = np.random.default_rng(7).choice(vB.shape[0], 600_000, replace=False)
        vB = vB[_k]                      # from `fates`, so subsampling is safe
    np.savez_compressed(VCACHE, vB=vB, vH=vH, vF=vF, fates=fates, N=N_NEUTRON)

for k, v in fates.items():
    print(f"    fate {k:5s}: {100*v/N_NEUTRON:7.3f} %")
for _nm, _v in (('B-10 (shield)', vB), ('H (shield+wood)', vH), ('Fe (steel)', vF)):
    if _v.shape[0]:
        _t = _v[:, 3] * 1e6
        print(f"    capture time, {_nm:16s}: median {np.median(_t):7.2f} us  "
              f"mean {_t.mean():7.2f} us  frac in 0.5-15 us "
              f"{100*((_t > 0.5) & (_t < 15)).mean():5.1f} %  "
              f"depth <z> {_v[:, 2].mean():5.2f} cm")

# tile arrivals: read (read-only!) the existing 10 M cache — 5x the statistics
_c = np.load('_mc3d_cache.npz', allow_pickle=True)
_arr, N_ARR, FAT_C = _c['arr'], int(_c['N']), _c['fates'].item()
E_t, t_st = _arr[:, 0], _arr[:, 1]
_hit = (np.abs(_arr[:, 2]) <= HALF_TILE) & (np.abs(_arr[:, 3]) <= HALF_TILE) & (_arr[:, 6] > 0)
ARR = dict(E=E_t[_hit], t=t_st[_hit] * 1e6, x=_arr[_hit, 2], y=_arr[_hit, 3],
           ux=_arr[_hit, 4], uy=_arr[_hit, 5], uz=_arr[_hit, 6])
N_HIT = ARR['E'].size
print(f"    tile arrivals (from the untouched 10 M cache): {N_HIT:,} per {N_ARR:,} emitted")


# ═══════════════════════════════════════════════════════════════════════════
#  §3  PHOTON PHYSICS
# ═══════════════════════════════════════════════════════════════════════════
MEC2 = 0.510999
R_E = 2.8179403e-13          # cm


def kn_sigma(E):
    """Klein-Nishina total cross section per electron, cm^2.  E in MeV."""
    a = np.asarray(E, float) / MEC2
    a = np.maximum(a, 1e-6)
    l = np.log1p(2 * a)
    return (2 * np.pi * R_E ** 2) * ((1 + a) / a ** 2 * (2 * (1 + a) / (1 + 2 * a) - l / a)
                                     + l / (2 * a) - (1 + 3 * a) / (1 + 2 * a) ** 2)


# electron densities [cm^-3]
NE_BHDPE = 1.00 * N_A * (0.143 * 0.99212 + 0.807 * 0.49955 + 0.05 * 0.46250)
NE_WOOD  = 0.60 * N_A * (0.06 * 0.99212 + 0.50 * 0.49955 + 0.44 * 0.50000)
NE_AIR   = 1.205e-3 * N_A * 0.49919
NE_FE    = 7.85 * N_A * (26.0 / 55.845)
NE_PVT   = 1.0 * n_H_sc + 6.0 * n_C_sc + 1.05 * 0.02 * 0.4625 * N_A

# XCOM CH2 mass attenuation (file in this directory: E[MeV], mu/rho, mu_en/rho)
_x = np.loadtxt('xcom_polyethylene.txt')
_XE, _XMU = _x[:, 0], _x[:, 1]
ZA_CH2, ZA_PVT, ZA_WOOD = 0.57034, 0.54141, 0.52930


def mu_plastic(E, ne):
    """Total linear attenuation [1/cm] for a CH-like plastic of electron
    density ne, taken from the XCOM CH2 table scaled by electron density."""
    mrho = np.exp(np.interp(np.log(np.maximum(E, 1e-3)), np.log(_XE), np.log(_XMU)))
    return mrho * (ne / (N_A * ZA_CH2))       # mu/rho * rho_equivalent


# iron, NIST/XCOM mu/rho [cm^2/g] (with coherent)
_FEE = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20,
                 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0,
                 6.0, 8.0, 10.0, 15.0])
_FEM = np.array([170.6, 57.08, 25.68, 8.176, 3.629, 1.958, 1.205, 0.5952, 0.3717,
                 0.1964, 0.1460, 0.1099, 0.0940, 0.0840, 0.0767, 0.0664, 0.0599,
                 0.0546, 0.0500, 0.0425, 0.0361, 0.0331, 0.0314, 0.0305, 0.0298,
                 0.0300, 0.0320])


def mu_iron(E):
    return 7.85 * np.exp(np.interp(np.log(np.maximum(E, 1e-3)), np.log(_FEE), np.log(_FEM)))


def mu_tot_layer(ilay, E):
    if ilay == 0: return mu_plastic(E, NE_BHDPE)
    if ilay == 1: return mu_plastic(E, NE_WOOD)
    if ilay == 3: return mu_iron(E)
    return mu_plastic(E, NE_AIR)


def mu_tile(E):
    return mu_plastic(E, NE_PVT)


def mu_incoh_tile(E):
    return NE_PVT * kn_sigma(E)


def e_range(T):
    """Katz-Penfold practical range of an electron of kinetic energy T [MeV]
    in PVT (rho = 1.05), cm.  Practical (not CSDA) range: detour included."""
    T = np.maximum(np.asarray(T, float), 1e-4)
    r = np.where(T <= 3.0, 0.412 * T ** (1.265 - 0.0954 * np.log(T)), 0.530 * T - 0.106)
    return r / 1.05


def sample_kn(E, rng):
    """Sample the Compton scattered-photon energy fraction eps = E'/E and the
    photon scattering cosine, from the exact Klein-Nishina kernel."""
    n = E.size
    a = E / MEC2
    e0 = 1.0 / (1.0 + 2.0 * a)
    A = np.log(1.0 / e0)
    B = 0.5 * (1.0 - e0 * e0)
    eps = np.empty(n); ok = np.zeros(n, bool)
    for _ in range(60):
        m = ~ok
        k = int(m.sum())
        if k == 0: break
        u1, u2, u3 = rng.random(k), rng.random(k), rng.random(k)
        br = u1 < (A[m] / (A[m] + B[m]))
        e = np.where(br, e0[m] * np.exp(u2 * A[m]),
                     np.sqrt(e0[m] ** 2 + u2 * (1.0 - e0[m] ** 2)))
        ct = 1.0 - (1.0 - e) / (a[m] * e)
        s2 = np.maximum(1.0 - ct * ct, 0.0)
        acc = u3 < (1.0 - e * s2 / (1.0 + e * e))
        idx = np.flatnonzero(m)
        eps[idx[acc]] = e[acc]
        ok[idx[acc]] = True
    eps[~ok] = 1.0
    ct = np.clip(1.0 - (1.0 - eps) / (np.maximum(a, 1e-9) * eps), -1.0, 1.0)
    return eps, ct


def iso_dir(n, rng):
    c = 2 * rng.random(n) - 1
    s = np.sqrt(np.maximum(1 - c * c, 0))
    p = 2 * np.pi * rng.random(n)
    return s * np.cos(p), s * np.sin(p), c


def rot_vec(ux, uy, uz, c, rng):
    return _rotate(ux, uy, uz, c, rng)


def box_exit(x, y, z, ux, uy, uz):
    """Distance from an interior point to the tile boundary along (ux,uy,uz)."""
    big = 1e30
    dx = np.where(ux > 0, (HALF_TILE - x) / np.where(ux > 0, ux, 1),
                  np.where(ux < 0, (-HALF_TILE - x) / np.where(ux < 0, ux, 1), big))
    dy = np.where(uy > 0, (HALF_TILE - y) / np.where(uy > 0, uy, 1),
                  np.where(uy < 0, (-HALF_TILE - y) / np.where(uy < 0, uy, 1), big))
    dz = np.where(uz > 0, (D_TILE + TILE_T - z) / np.where(uz > 0, uz, 1),
                  np.where(uz < 0, (D_TILE - z) / np.where(uz < 0, uz, 1), big))
    return np.maximum(np.minimum(np.minimum(dx, dy), dz), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  §4  IN-TILE PHOTON CASCADE  ->  deposited electron energy
# ═══════════════════════════════════════════════════════════════════════════
def tile_cascade(x, y, z, ux, uy, uz, E, rng, forced_first=True, ngen=8):
    """Track photons already inside the tile; return the deposited energy [MeV].
    The first interaction is forced at (x,y,z) when forced_first (the
    next-event estimator has already paid for the probability of getting
    there); afterwards free flight with mu_tot decides escape.
    Compton electrons (and pair electrons) escape the 1 cm slab according to
    the Katz-Penfold practical range along their emission direction."""
    dep = np.zeros(E.size)
    live = np.ones(E.size, bool)
    first = forced_first
    for g in range(ngen):
        idx = np.flatnonzero(live)
        if idx.size == 0: break
        Ei = E[idx]
        if not first:
            d = box_exit(x[idx], y[idx], z[idx], ux[idx], uy[idx], uz[idx])
            s = -np.log(rng.random(idx.size)) / np.maximum(mu_tile(Ei), 1e-12)
            esc = s >= d
            x[idx] += s * ux[idx]; y[idx] += s * uy[idx]; z[idx] += s * uz[idx]
            live[idx[esc]] = False
            idx = idx[~esc]
            if idx.size == 0: break
            Ei = E[idx]
        first = False
        # ---- branch ---------------------------------------------------------
        mt = mu_tile(Ei); mi = mu_incoh_tile(Ei)
        fc = np.clip(mi / np.maximum(mt, 1e-12), 0.0, 1.0)
        u = rng.random(idx.size)
        is_comp = (u < fc) & (Ei > E_PHOT_CUT)
        is_pair = (~is_comp) & (Ei > 1.022)
        is_abs = (~is_comp) & (~is_pair)
        # photo-absorption (and the low-energy remainder): full deposit
        if is_abs.any():
            j = idx[is_abs]
            dep[j] += E[j]; live[j] = False
        # pair production
        if is_pair.any():
            j = idx[is_pair]
            Tt = E[j] - 1.022
            f = rng.random(j.size)
            for Tk in (Tt * f, Tt * (1 - f)):
                ex, ey, ez = iso_dir(j.size, rng)
                s = box_exit(x[j], y[j], z[j], ex, ey, ez)
                dep[j] += Tk * np.minimum(1.0, s / np.maximum(e_range(Tk), 1e-6))
            # two 511 keV annihilation photons, single-interaction treatment
            for _k in range(2):
                ax_, ay_, az_ = iso_dir(j.size, rng)
                s = box_exit(x[j], y[j], z[j], ax_, ay_, az_)
                pint = 1.0 - np.exp(-mu_tile(np.full(j.size, MEC2)) * s)
                hit = rng.random(j.size) < pint
                if hit.any():
                    jj = j[hit]
                    e2, _ct = sample_kn(np.full(jj.size, MEC2), rng)
                    T2 = MEC2 * (1 - e2)
                    ex, ey, ez = iso_dir(jj.size, rng)
                    s2 = box_exit(x[jj], y[jj], z[jj], ex, ey, ez)
                    dep[jj] += T2 * np.minimum(1.0, s2 / np.maximum(e_range(T2), 1e-6))
            live[j] = False
        # Compton
        if is_comp.any():
            j = idx[is_comp]
            eps, ct = sample_kn(E[j], rng)
            T = E[j] * (1 - eps)
            a = E[j] / MEC2
            th = np.arccos(np.clip(ct, -1, 1))
            tane = 1.0 / np.maximum((1 + a) * np.tan(np.maximum(th, 1e-9) / 2), 1e-12)
            cte = np.cos(np.arctan(tane))
            ex, ey, ez = rot_vec(ux[j], uy[j], uz[j], cte, rng)
            s = box_exit(x[j], y[j], z[j], ex, ey, ez)
            dep[j] += T * np.minimum(1.0, s / np.maximum(e_range(T), 1e-6))
            E[j] = E[j] * eps
            ux[j], uy[j], uz[j] = rot_vec(ux[j], uy[j], uz[j], ct, rng)
            dead = E[j] <= E_PHOT_CUT
            if dead.any():
                dep[j[dead]] += E[j[dead]]; live[j[dead]] = False
    return dep


# ═══════════════════════════════════════════════════════════════════════════
#  §5  SHIELD -> TILE PHOTON TRANSPORT WITH A NEXT-EVENT ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════
V_TILE = (2 * HALF_TILE) ** 2 * TILE_T          # 9 cm^3


def slab_path(x0, y0, z0, x1, y1, z1):
    """Attenuation exp(-tau) along the segment through the z-slab stack and the
    tile.  The stack is treated as laterally infinite (rays to the tile are
    near-forward, so the lateral extent traversed is < 3 cm)."""
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    uzz = dz / np.maximum(r, 1e-12)
    return r, uzz


def next_event(px, py, pz, E, w, rng, nsamp):
    """For source points (px,py,pz) with photon energy E and weight w, score
    the expected number of FIRST INTERACTIONS in the tile, returning the
    sampled interaction points, incoming directions and weights."""
    n = px.size
    out = []
    for _ in range(nsamp):
        tx = (rng.random(n) - 0.5) * 2 * HALF_TILE
        ty = (rng.random(n) - 0.5) * 2 * HALF_TILE
        tz = D_TILE + rng.random(n) * TILE_T
        dx, dy, dz = tx - px, ty - py, tz - pz
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        ux, uy, uz = dx / r, dy / r, dz / r
        fwd = uz > 1e-6
        # path in each z-slab between the source and the tile entry
        # tile entry: ray-box in 3-D
        big = 1e30
        with np.errstate(divide='ignore', invalid='ignore'):
            t1x = np.where(np.abs(ux) > 1e-12, (-HALF_TILE - px) / ux, -big)
            t2x = np.where(np.abs(ux) > 1e-12, (HALF_TILE - px) / ux, big)
            t1y = np.where(np.abs(uy) > 1e-12, (-HALF_TILE - py) / uy, -big)
            t2y = np.where(np.abs(uy) > 1e-12, (HALF_TILE - py) / uy, big)
            t1z = (D_TILE - pz) / np.where(np.abs(uz) > 1e-12, uz, 1e-12)
            t2z = (D_TILE + TILE_T - pz) / np.where(np.abs(uz) > 1e-12, uz, 1e-12)
        tmin = np.maximum(np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y)),
                          np.minimum(t1z, t2z))
        s_ent = np.clip(tmin, 0.0, r)
        # attenuation in the stack from the source to the tile entry
        tau = np.zeros(n)
        z_a = pz
        z_b = pz + s_ent * uz
        for il in range(NLAY):
            lo = np.maximum(np.minimum(z_a, z_b), _LZ0[il])
            hi = np.minimum(np.maximum(z_a, z_b), _LZ1[il])
            seg = np.maximum(hi - lo, 0.0) / np.maximum(np.abs(uz), 1e-9)
            tau += mu_tot_layer(il, E) * seg
        # attenuation inside the tile up to the sampled point
        tau += mu_tile(E) * np.maximum(r - s_ent, 0.0)
        wgt = w * (V_TILE / nsamp) * mu_tile(E) / (4 * np.pi * np.maximum(r, 1e-6) ** 2) \
            * np.exp(-np.minimum(tau, 200.0))
        wgt = np.where(fwd, wgt, 0.0)
        out.append((tx, ty, tz, ux, uy, uz, E.copy(), wgt))
    return out


def shield_photons(vx, vy, vz, vt, Egam, wgt0, rng, tag):
    """Transport photons from capture vertices to the tile.  Returns
    (charge-less) deposit energies, weights, and capture times."""
    n = vx.size
    px, py, pz = vx.copy(), vy.copy(), vz.copy()
    E = np.asarray(Egam, float).copy()
    if E.size == 1: E = np.full(n, float(E))
    w = np.full(n, float(wgt0)) if np.isscalar(wgt0) else wgt0.copy()
    w_ref = np.maximum(np.abs(w), 1e-300)
    tt = vt.copy()
    ux, uy, uz = iso_dir(n, rng)
    dep_all, wt_all, t_all, gen_all = [], [], [], []
    live = np.ones(n, bool)
    for g in range(MAX_SCAT_SHIELD + 1):
        idx = np.flatnonzero(live)
        if idx.size == 0: break
        for (tx, ty, tz, dxu, dyu, dzu, Ee, ww) in next_event(
                px[idx], py[idx], pz[idx], E[idx], w[idx], rng, NE_SAMP):
            keep = ww > 0
            if not keep.any(): continue
            d = tile_cascade(tx[keep], ty[keep], tz[keep], dxu[keep], dyu[keep],
                             dzu[keep], Ee[keep].copy(), rng, forced_first=True)
            dep_all.append(d); wt_all.append(ww[keep]); t_all.append(tt[idx][keep])
            gen_all.append(np.full(d.size, g, np.int8))
        if g == MAX_SCAT_SHIELD: break
        # --- move the photon to its next collision in the stack --------------
        i = idx
        for _sub in range(40):
            if i.size == 0: break
            lay = np.clip(np.searchsorted(_LZ1, pz[i] + 1e-12, side='right'), 0, NLAY - 1)
            mt = np.zeros(i.size)
            for il in range(NLAY):
                m = lay == il
                if m.any(): mt[m] = mu_tot_layer(il, E[i][m])
            s = -np.log(rng.random(i.size)) / np.maximum(mt, 1e-12)
            df = np.where(uz[i] > 0, (_LZ1[lay] - pz[i]) / np.where(uz[i] > 0, uz[i], 1),
                          np.where(uz[i] < 0, (pz[i] - _LZ0[lay]) / np.where(uz[i] < 0, -uz[i], 1),
                                   1e30))
            cross = s >= df
            d = np.where(cross, df + 1e-9, s)
            px[i] += d * ux[i]; py[i] += d * uy[i]; pz[i] += d * uz[i]
            out = cross & ((pz[i] <= 0) | (pz[i] >= D_TILE))
            live[i[out]] = False
            i = i[cross & ~out]
        # --- Compton scatter, implicit absorption ----------------------------
        j = np.flatnonzero(live)
        if j.size == 0: break
        lay = np.clip(np.searchsorted(_LZ1, pz[j] + 1e-12, side='right'), 0, NLAY - 1)
        ne_lay = np.array([NE_BHDPE, NE_WOOD, NE_AIR, NE_FE, NE_AIR])[lay]
        mt = np.zeros(j.size)
        for il in range(NLAY):
            m = lay == il
            if m.any(): mt[m] = mu_tot_layer(il, E[j][m])
        w[j] *= np.clip(ne_lay * kn_sigma(E[j]) / np.maximum(mt, 1e-12), 0, 1)
        eps, ct = sample_kn(E[j], rng)
        E[j] *= eps
        ux[j], uy[j], uz[j] = rot_vec(ux[j], uy[j], uz[j], ct, rng)
        live[j] &= (E[j] > 0.030) & (w[j] > 1e-3 * w_ref[j])
    if not dep_all:
        z4 = np.zeros(0)
        return z4, z4, z4, np.zeros(0, np.int8)
    return (np.concatenate(dep_all), np.concatenate(wt_all),
            np.concatenate(t_all), np.concatenate(gen_all))


# ═══════════════════════════════════════════════════════════════════════════
#  §6  NEUTRON RANDOM WALK INSIDE THE TILE  (tile as MODERATOR)
# ═══════════════════════════════════════════════════════════════════════════
# Birks light output for protons in PVT, from HYDROGEN.md §3.1 (kB = 0.0125)
_LP_E = np.array([10., 25., 50., 100., 200., 300., 500., 750., 1000., 1500.,
                  2000., 2450., 4000.])
_LP_L = np.array([0.86, 2.5, 6.1, 10.6, 20.7, 32.7, 61.7, 106.0, 160.4, 293.6,
                  451.3, 609.8, 1150.])


def light_p(Ep_keV):
    return np.exp(np.interp(np.log(np.maximum(Ep_keV, 1.0)), np.log(_LP_E), np.log(_LP_L)))


def light_c(Ec_keV):        # carbon recoils: quenched to 1-8 keVee, never trigger
    return 0.011 * Ec_keV


COINC_NS = 150.0            # DAQ integration window (int_ns = -20..150 ns)
SIG_T_SC = n_H_sc * 1e-24
SIG_T_CC = n_C_sc * 1e-24
SIG_CAP_B = n_B10_scint * S_B10_TH * 1e-24
SIG_CAP_H = n_H_sc * S_HG_TH * 1e-24


def tile_walk(seed, nrep=3):
    """Random-walk every tile arrival inside the 3x3x1 cm PVT tile.
    Returns per-arrival: light of the prompt recoil burst [keVee], time of the
    first recoil [us], capture flag/time/position, and whether the capture was
    on B-10 (alpha) or H (2.22 MeV gamma)."""
    rng = np.random.default_rng(seed)
    n0 = N_HIT
    E = np.tile(ARR['E'], nrep).astype(float)
    t = np.tile(ARR['t'], nrep).astype(float)
    x = np.tile(ARR['x'], nrep).astype(float)
    y = np.tile(ARR['y'], nrep).astype(float)
    z = np.full(E.size, D_TILE + 1e-9)
    ux = np.tile(ARR['ux'], nrep).astype(float)
    uy = np.tile(ARR['uy'], nrep).astype(float)
    uz = np.tile(ARR['uz'], nrep).astype(float)
    n = E.size
    lig = np.zeros(n); t1 = np.full(n, np.nan)
    Eesc = np.full(n, np.nan); tesc = np.full(n, np.nan)
    capt = np.zeros(n, np.int8)          # 0 none, 1 B-10, 2 H
    tcap = np.full(n, np.nan)
    xc = np.zeros(n); yc = np.zeros(n); zc = np.zeros(n)
    idx = np.arange(n)
    for it in range(20000):
        if idx.size == 0: break
        lgE = np.log(np.maximum(E[idx], 0.025))
        sH = np.exp(np.interp(lgE, _lE, np.log(_sH)))
        sC = np.exp(np.interp(lgE, _lE, np.log(_sC)))
        Ss = n_H_sc * sH * 1e-24 + n_C_sc * sC * 1e-24
        iv = np.sqrt(E_TH / np.maximum(E[idx], E_TH))
        SaB = SIG_CAP_B * iv
        SaH = SIG_CAP_H * iv
        St = Ss + SaB + SaH
        s = -np.log(rng.random(idx.size)) / St
        d = box_exit(x[idx], y[idx], z[idx], ux[idx], uy[idx], uz[idx])
        v = 1.3831e6 * np.sqrt(np.maximum(E[idx], 1e-6))
        esc = s >= d
        step = np.where(esc, d, s)
        t[idx] += step / v * 1e6
        x[idx] += step * ux[idx]; y[idx] += step * uy[idx]; z[idx] += step * uz[idx]
        gone = idx[esc]
        Eesc[gone] = E[gone]; tesc[gone] = t[gone]
        act = idx[~esc]
        if act.size == 0:
            idx = np.zeros(0, np.int64); break
        r = rng.random(act.size) * St[~esc]
        cb = r < SaB[~esc]
        ch = (~cb) & (r < (SaB + SaH)[~esc])
        sc = ~(cb | ch)
        if cb.any():
            j = act[cb]; capt[j] = 1; tcap[j] = t[j]; xc[j] = x[j]; yc[j] = y[j]; zc[j] = z[j]
        if ch.any():
            j = act[ch]; capt[j] = 2; tcap[j] = t[j]; xc[j] = x[j]; yc[j] = y[j]; zc[j] = z[j]
        if sc.any():
            j = act[sc]
            fH = (n_H_sc * sH[~esc][sc] * 1e-24) / Ss[~esc][sc]
            onH = rng.random(j.size) < fH
            dE = np.zeros(j.size)
            # hydrogen: isotropic in CM, E' = E cos^2
            jc = np.sqrt(rng.random(j.size))
            dE_h = E[j] * (1 - jc * jc)
            # carbon
            cc = 2 * rng.random(j.size) - 1
            dn = 1 + 2 * 12.0 * cc + 144.0
            dE_c = E[j] * (1 - dn / 169.0)
            dE = np.where(onH, dE_h, dE_c)
            L = np.where(onH, light_p(dE / 1e3), light_c(dE / 1e3))
            newb = np.isnan(t1[j])
            t1[j] = np.where(newb, t[j], t1[j])
            inwin = (t[j] - np.where(newb, t[j], t1[j])) * 1e3 <= COINC_NS
            lig[j] += np.where(inwin, L, 0.0)
            Enew = np.where(onH, np.maximum(E[j] * jc * jc, 0.025),
                            np.maximum(E[j] * dn / 169.0, 0.025))
            E[j] = Enew
            cth = np.where(onH, jc, (1 + 12.0 * cc) / np.sqrt(dn))
            ux[j], uy[j], uz[j] = rot_vec(ux[j], uy[j], uz[j], cth, rng)
        idx = act[sc]
    return dict(light=lig, t1=t1, capt=capt, tcap=tcap, xc=xc, yc=yc, zc=zc,
                Eesc=Eesc, tesc=tesc,
                nrep=nrep, n0=n0)


WCACHE = '_gam_tilewalk.npz'
if os.path.exists(WCACHE):
    _w = np.load(WCACHE)
    WALK = {k: _w[k] for k in _w.files}
    WALK['nrep'] = int(WALK['nrep']); WALK['n0'] = int(WALK['n0'])
    print(f"loaded {WCACHE}")
else:
    print("running the in-tile neutron random walk (tile as moderator)...")
    _t0 = time_module.time()
    WALK = tile_walk(SEED + 1, nrep=3)
    np.savez_compressed(WCACHE, **WALK)
    print(f"  tile walk done in {time_module.time()-_t0:.0f}s")

NWALK = WALK['light'].size
NORM_WALK = WALK['nrep'] * N_ARR        # emitted neutrons represented
n_cap_B = int((WALK['capt'] == 1).sum())
n_cap_H = int((WALK['capt'] == 2).sum())
eps_cap = (n_cap_B + n_cap_H) / NWALK
print(f"  tile walk: {NWALK:,} arrivals tracked, captures {n_cap_B:,} on B-10 + "
      f"{n_cap_H:,} on H  ->  eps_cap = {100*eps_cap:.3f} % per arriving neutron")
print(f"    (single-pass first-interaction model gives 0.348 %; boost x{eps_cap/0.00348:.2f})")
_Ew = np.tile(ARR['E'], WALK['nrep']); _tw = np.tile(ARR['t'], WALK['nrep'])
_cb = WALK['capt'] == 1
print("    capture efficiency per ARRIVING neutron, by arrival energy group:")
for _lo, _hi, _lb in [(1e5, 4e6, 'fast    >100 keV'), (1.0, 1e5, 'epithermal'),
                      (0.0, 1.0, 'thermal  <1 eV')]:
    _m = (_Ew >= _lo) & (_Ew < _hi)
    print(f"      {_lb:18s} {int(_m.sum()):7d} arrivals  eps = {100*_cb[_m].mean():7.3f} %"
          f"  -> {int((_cb & _m).sum()):5d} captures,"
          f"  median t_arrival {np.median(_tw[_m]):7.2f} us")
_tc = WALK['tcap'][WALK['capt'] == 1]
print(f"    tile-capture clock: median {np.median(_tc):.2f} us, mean {_tc.mean():.2f} us")
# can a DELAYED neutron recoil ever trigger?  needs E_n > 404 keV (= 47 keVee)
_late = (ARR['t'] > 1.5) & (ARR['E'] > 4.04e5)
print(f"    arrivals with E>404 keV later than 1.5 us: {int(_late.sum())} of {N_HIT:,} "
      f"-> delayed recoil triggers are negligible by construction")


# ═══════════════════════════════════════════════════════════════════════════
#  §7  BUILD THE PREDICTED CHARGE SPECTRUM AND TIME PROFILE
# ═══════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED + 2)

# ---- (a) shield/steel capture gammas ---------------------------------------
def subsample(v, nmax, rng):
    if v.shape[0] <= nmax: return v, 1.0
    k = rng.choice(v.shape[0], nmax, replace=False)
    return v[k], v.shape[0] / nmax


comp = {}          # tag -> (dep MeV, weight per emitted neutron, t_us, gen)

# B-10, 478 keV
sB, fB = subsample(vB, N_SRC_B10, rng)
w0 = Y_B478 * (fates['B10'] / sB.shape[0]) / N_NEUTRON
d, w, tt, gg = shield_photons(sB[:, 0].astype(float), sB[:, 1].astype(float),
                              sB[:, 2].astype(float), sB[:, 3].astype(float) * 1e6,
                              E_B478, w0, rng, 'B10')
comp['B-10 478 keV'] = (d, w, tt, gg)

# H, 2.223 MeV
rep = N_REP_H
sH_v = np.repeat(vH, rep, axis=0)
w0 = Y_H2223 * (fates['H'] / sH_v.shape[0]) / N_NEUTRON
d, w, tt, gg = shield_photons(sH_v[:, 0].astype(float), sH_v[:, 1].astype(float),
                              sH_v[:, 2].astype(float), sH_v[:, 3].astype(float) * 1e6,
                              E_H2223, w0, rng, 'H')
comp['H 2.22 MeV'] = (d, w, tt, gg)

# Fe cascade
rep = N_REP_FE
sF = np.repeat(vF, rep, axis=0)
nF = sF.shape[0]
single = rng.random(nF) < FE_SINGLE_FRAC
Efe = np.where(single, 7.638, 0.0)
# 3-photon cascade for the rest: Dirichlet(1,1,1) split of 7.646 MeV
u1 = np.sort(rng.random((nF, 2)), axis=1)
frac = np.column_stack([u1[:, 0], u1[:, 1] - u1[:, 0], 1 - u1[:, 1]])
dF, wF, tF, gF = [], [], [], []
w0 = (fates['Fe'] / nF) / N_NEUTRON
m = single
if m.any():
    d, w, tt, gg = shield_photons(sF[m, 0].astype(float), sF[m, 1].astype(float),
                                  sF[m, 2].astype(float), sF[m, 3].astype(float) * 1e6,
                                  np.full(int(m.sum()), 7.638), w0, rng, 'Fe1')
    dF.append(d); wF.append(w); tF.append(tt); gF.append(gg)
m = ~single
if m.any():
    for k in range(3):
        Ek = np.maximum(frac[m, k] * E_FE_TOT, 0.05)
        d, w, tt, gg = shield_photons(sF[m, 0].astype(float), sF[m, 1].astype(float),
                                      sF[m, 2].astype(float), sF[m, 3].astype(float) * 1e6,
                                      Ek, w0, rng, f'Fe3_{k}')
        dF.append(d); wF.append(w); tF.append(tt); gF.append(gg)
comp['Fe 7.6 MeV cascade'] = (np.concatenate(dF), np.concatenate(wF),
                              np.concatenate(tF), np.concatenate(gF))

# ---- (b) tile B-10 captures: alpha line + its own 478 keV gamma -------------
mB = WALK['capt'] == 1
n_a = int(mB.sum())
xa, ya, za = WALK['xc'][mB], WALK['yc'][mB], WALK['zc'][mB]
ta = WALK['tcap'][mB]
# alpha light, measured line, smeared with the measured width later
alpha_keVee = np.full(n_a, Q_ALPHA * KEVEE_PER_VNS)
# 478 keV gamma emitted at the capture point, tracked in the tile
has_g = rng.random(n_a) < Y_B478
gx, gy, gz = iso_dir(n_a, rng)
Eg = np.full(n_a, E_B478)
xx, yy, zz = xa.copy(), ya.copy(), za.copy()
dgam = tile_cascade(xx, yy, zz, gx, gy, gz, Eg, rng, forced_first=False)
dgam = np.where(has_g, dgam, 0.0)
w_alpha = np.full(n_a, 1.0 / NORM_WALK)
comp['tile capture (alpha)'] = (alpha_keVee / 1e3 + dgam, w_alpha, ta,
                                np.zeros(n_a, np.int8))
frac_sum = float((dgam > 0.005).mean())
print(f"  tile captures whose own 478 keV gamma also deposits >5 keV: {100*frac_sum:.1f} %")

# ---- (c) tile H captures: 2.223 MeV gamma from inside the tile --------------
mH = WALK['capt'] == 2
n_h = int(mH.sum())
if n_h:
    gx, gy, gz = iso_dir(n_h, rng)
    xx, yy, zz = WALK['xc'][mH].copy(), WALK['yc'][mH].copy(), WALK['zc'][mH].copy()
    dh = tile_cascade(xx, yy, zz, gx, gy, gz, np.full(n_h, E_H2223), rng,
                      forced_first=False)
    comp['tile H capture'] = (dh, np.full(n_h, 1.0 / NORM_WALK),
                              WALK['tcap'][mH], np.zeros(n_h, np.int8))

# ---- (d) neutron recoils (mostly in-gate, kept for the ledger) --------------
lig = WALK['light']; t1 = WALK['t1']
mR = (lig > 0) & np.isfinite(t1)
comp['neutron recoil'] = (lig[mR] / 1e3, np.full(int(mR.sum()), 1.0 / NORM_WALK),
                          t1[mR], np.zeros(int(mR.sum()), np.int8))


# ---- (e) charge conversion, resolution, gate folding ------------------------
def to_charge(dep_MeV, is_alpha=False, rng=rng):
    keVee = dep_MeV * 1e3
    q = keVee / KEVEE_PER_VNS
    sig = np.where(q > 0, RES_S0 * np.sqrt(np.maximum(keVee, 1e-6) * RES_E0) / KEVEE_PER_VNS, 0.0)
    if is_alpha:
        sig = np.sqrt(sig ** 2 + max(S_ALPHA ** 2 - (RES_S0 * RES_E0 / KEVEE_PER_VNS) ** 2, 0.0))
    return np.maximum(q + sig * rng.standard_normal(q.size), 0.0)


EVT = {}
for tag, (d, w, tt, gg) in comp.items():
    if d.size == 0: continue
    q = to_charge(d, is_alpha=tag.startswith('tile capture'))
    te = tt + rng.random(d.size) * GATE_US - GATE_US
    EVT[tag] = dict(q=q, w=w, tau=te, gen=gg, dep=d)


SC_LATE = (TAU_HI - TAU_LO) / (TAU_L1 - TAU_L0)      # 14.5 / 3.5


def net_w(tau, w):
    """The MC seen through the SAME analysis as the data: delayed window minus
    the late window scaled by 14.5/3.5.  Anything flat inside the 18.5 us
    record (very late tile captures, room return) cancels, exactly as it does
    in the measurement."""
    return (w * ((tau >= TAU_LO) & (tau < TAU_HI))
            - SC_LATE * w * ((tau >= TAU_L0) & (tau < TAU_L1)))


def sel(tag, qlo=Q_THR, qhi=1e9):
    e = EVT[tag]
    m = (e['q'] >= qlo) & (e['q'] < qhi)
    return float(net_w(e['tau'], e['w'])[m].sum())


# ═══════════════════════════════════════════════════════════════════════════
#  §8  DATA
# ═══════════════════════════════════════════════════════════════════════════
DCACHE = '_gam_data.npz'
DUMPER = r'''
import numpy as np, sys
from pnsana.extract import Feat, cache_path
sc = 14.5/3.5
out = {}
for key, run, clipv in [('r14','20260814_sample02_gate1us_period50us_finalConfig_merged',4.4),
                        ('r20','20260820_sample11_gate1us_period50us_finalConfig',9.5)]:
    f = Feat(cache_path(run,'cache'))
    g=np.asarray(f.gate_ok); q=np.asarray(f.charge); te=np.asarray(f.tau_end_us); cl=np.asarray(f.clip)
    m=g&np.isfinite(te)
    dly=m&(te>=0.5)&(te<15); lat=m&(te>=15)&(te<18.5)
    b=np.arange(0.0, clipv+1e-9, 0.1)
    hd=np.histogram(q[dly&~cl],bins=b)[0]; hl=np.histogram(q[lat&~cl],bins=b)[0]
    out[key+'_bins']=b; out[key+'_net']=hd-sc*hl; out[key+'_err']=np.sqrt(hd+sc*sc*hl)
    out[key+'_nrec']=np.array([g.sum()])
    out[key+'_clipnet']=np.array([(dly&cl).sum()-sc*(lat&cl).sum()])
    tb=np.arange(0.5,18.51,0.5)
    bands=[('alpha',0.45,0.85),('c478',1.0,2.5),('above',3.0,min(4.4,clipv)),('lo',0.35,1.0)]
    for nm,lo,hi in bands:
        s=m&(q>=lo)&(q<hi)&~cl
        out[key+'_t_'+nm]=np.histogram(te[s],bins=tb)[0]
    s=m&cl
    out[key+'_t_clip']=np.histogram(te[s],bins=tb)[0]
    out[key+'_tbins']=tb
np.savez(sys.argv[1], **out)
print('data dump ok')
'''
if not os.path.exists(DCACHE):
    _tmp = os.path.join(HERE, '_gam_dump_tmp.py')
    open(_tmp, 'w').write(DUMPER)
    env = dict(os.environ, PYTHONPATH='/Users/virgolaema/Software/3det/pns-waveform-ana')
    try:
        subprocess.run(['/Users/virgolaema/Software/neutron-env/bin/python', _tmp,
                        os.path.join(HERE, DCACHE)],
                       cwd='/Users/virgolaema/Software/3det/pns-waveform-ana',
                       env=env, check=True)
    except Exception as ex:
        print('data dump failed:', ex)
    finally:
        if os.path.exists(_tmp): os.remove(_tmp)
DAT = np.load(DCACHE) if os.path.exists(DCACHE) else None


# ═══════════════════════════════════════════════════════════════════════════
#  §9  NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════
# MC yields PER CONE-EMITTED NEUTRON
y_alpha_dly = sel('tile capture (alpha)')                     # delayed, >thr
y_alpha_all = float(EVT['tile capture (alpha)']['w'].sum())
e = EVT['neutron recoil']
m_ing = (e['q'] >= Q_THR) & (e['tau'] < 0.0)
y_recoil_ingate = float(e['w'][m_ing].sum())
y_recoil_dly = sel('neutron recoil')

r_alpha = (DATA_ALPHA / N_REC) / max(y_alpha_dly, 1e-30)      # cone-neutrons/record
r_recoil = (DATA_INGATE / N_REC) / max(y_recoil_ingate, 1e-30)

print("\n" + "=" * 78)
print("NORMALISATION")
print("=" * 78)
print(f"  MC per cone-emitted neutron:")
print(f"    tile B-10 captures, all times          : {y_alpha_all:.3e}")
print(f"    tile B-10 captures, delayed & >thr     : {y_alpha_dly:.3e}")
print(f"    in-gate recoil triggers (>0.35 V.ns)   : {y_recoil_ingate:.3e}")
print(f"    delayed recoil triggers                : {y_recoil_dly:.3e}")
print(f"  implied cone-emitted neutrons per record:")
print(f"    from tile captures  (0.0547/rec)       : {r_alpha:8.0f}")
print(f"    from in-gate recoils(0.1414/rec)       : {r_recoil:8.0f}")
print(f"    ratio recoil-norm / capture-norm       : {r_recoil/r_alpha:8.2f}")
print(f"  in-gate recoils : tile captures")
print(f"    MC, ALL captures                  {y_recoil_ingate/max(y_alpha_all,1e-30):6.2f} : 1"
      f"   (HYDROGEN.md ledger predicted 3.8 : 1)")
print(f"    MC, captures inside the net window", end=" ")
print(f" {y_recoil_ingate/max(y_alpha_dly,1e-30):6.2f} : 1      "
      f"data {DATA_INGATE/DATA_ALPHA:6.2f} : 1      MC/data "
      f"{(y_recoil_ingate/max(y_alpha_dly,1e-30))/(DATA_INGATE/DATA_ALPHA):.2f}")
NORM = r_alpha          # default: normalise to the measured tile captures


# ═══════════════════════════════════════════════════════════════════════════
#  §10  REPORT NUMBERS
# ═══════════════════════════════════════════════════════════════════════════
QB = np.arange(0.0, 12.0001, 0.1)
QC = 0.5 * (QB[1:] + QB[:-1])
STACK = {}
ORDER = ['B-10 478 keV', 'H 2.22 MeV', 'Fe 7.6 MeV cascade',
         'tile capture (alpha)', 'tile H capture', 'neutron recoil']
LBL = {'B-10 478 keV': 'shield B-10 478 keV',
       'H 2.22 MeV': 'shield/wood H 2.22 MeV',
       'Fe 7.6 MeV cascade': 'steel Fe 7.6 MeV cascade',
       'tile capture (alpha)': 'tile capture: alpha (+ own 478 keV)',
       'tile H capture': 'tile H 2.22 MeV',
       'neutron recoil': 'neutron recoil (delayed part)'}
for tag in ORDER:
    if tag not in EVT: continue
    e = EVT[tag]
    m = e['q'] >= Q_THR
    STACK[tag] = np.histogram(e['q'][m], bins=QB,
                              weights=net_w(e['tau'], e['w'])[m])[0] * NORM * N_REC

STACK_UNC = {}
for tag in ORDER:
    if tag not in EVT: continue
    e = EVT[tag]
    m = (e['q'] >= Q_THR) & (e['gen'] == 0)
    STACK_UNC[tag] = np.histogram(e['q'][m], bins=QB,
                                  weights=net_w(e['tau'], e['w'])[m])[0] * NORM * N_REC
tot_unc = sum(STACK_UNC.values())
tot_pred = sum(STACK.values())
i275 = np.searchsorted(QB, 2.75) - 1
i440 = np.searchsorted(QB, Q_CLIP_14) - 1

print("\n" + "=" * 78)
print("PREDICTED DELAYED SPECTRUM  (normalised to the measured tile captures)")
print("=" * 78)
print(f"  {'component':38s} {'>0.35':>9s} {'>2.75':>9s} {'>4.4':>9s}")
for tag in ORDER:
    if tag not in STACK: continue
    h = STACK[tag]
    print(f"  {LBL[tag]:38s} {h.sum():9.0f} {h[i275:].sum():9.0f} {h[i440:].sum():9.0f}")
print(f"  {'TOTAL predicted':38s} {tot_pred.sum():9.0f} {tot_pred[i275:].sum():9.0f} "
      f"{tot_pred[i440:].sum():9.0f}")
if DAT is not None:
    dn = DAT['r14_net']; db = DAT['r14_bins']
    j275 = np.searchsorted(db, 2.75) - 1
    jthr = np.searchsorted(db, Q_THR) - 1
    dtot = dn[jthr:].sum() + float(DAT['r14_clipnet'][0])
    print(f"  {'DATA run 20260814 (net excess)':38s} {dtot:9.0f} "
          f"{dn[j275:].sum() + float(DAT['r14_clipnet'][0]):9.0f} "
          f"{float(DAT['r14_clipnet'][0]):9.0f}")

# interaction ratios in the tile
def ints(tag):
    return float(EVT[tag]['w'].sum()) if tag in EVT else 0.0


print("\n  gamma interactions in the tile per cone-emitted neutron (any energy):")
for tag in ['B-10 478 keV', 'H 2.22 MeV', 'Fe 7.6 MeV cascade']:
    print(f"    {LBL[tag]:38s} {ints(tag):.4e}")
print(f"    H : B-10  = {100*ints('H 2.22 MeV')/max(ints('B-10 478 keV'),1e-30):6.2f} %")
print(f"    Fe : B-10 = {100*ints('Fe 7.6 MeV cascade')/max(ints('B-10 478 keV'),1e-30):6.2f} %")
# above-threshold, delayed
print("  ... and of the DELAYED, above-threshold, recorded events:")
for tag in ['B-10 478 keV', 'H 2.22 MeV', 'Fe 7.6 MeV cascade']:
    print(f"    {LBL[tag]:38s} {STACK[tag].sum():9.0f}  "
          f"({100*STACK[tag].sum()/max(tot_pred.sum(),1e-9):5.1f} % of prediction)")

# uncollided vs buildup
print("\n  buildup (scattered photons in the shield) contribution:")
for tag in ['B-10 478 keV', 'H 2.22 MeV', 'Fe 7.6 MeV cascade']:
    e = EVT[tag]
    m = e['q'] >= Q_THR
    nw = net_w(e['tau'], e['w'])
    tot = nw[m].sum(); unc = nw[m & (e['gen'] == 0)].sum()
    print(f"    {LBL[tag]:38s} uncollided {100*unc/max(tot,1e-30):5.1f} %  "
          f"buildup factor {tot/max(unc,1e-30):4.2f}")




# ═══════════════════════════════════════════════════════════════════════════
#  §10b  HOW MUCH SHIELD DOES THE SOURCE ACTUALLY LIGHT UP?
#  The MC only follows a 20 deg cone.  The slabs are laterally infinite, so
#  re-running with a wider cone is exactly "a wider shield seen by the same
#  isotropic source".  Tile captures come from the near-axis flux and are
#  unchanged per steradian, so this is a clean multiplier on the gamma signal.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("SENSITIVITY: illuminated solid angle (shield lateral extent)")
print("=" * 78)
print(f"  {'cone half-angle':>16s} {'sr':>8s} {'478 keV tile hits':>19s} {'x(20 deg)':>10s}")
_cone_rows = []
_base = None
for _cd in (20.0, 30.0, 45.0, 60.0, 90.0):
    _Nv = 400_000
    _vb, _vh, _vf, _fa = transport(_Nv, SEED + 300 + int(_cd), cone_deg=_cd)
    _sr = 2 * np.pi * (1 - math.cos(math.radians(_cd)))
    _sb, _ = subsample(_vb, 40_000, rng)
    _w0 = Y_B478 * (_fa['B10'] / _sb.shape[0]) / _Nv
    _d, _w, _t, _g = shield_photons(_sb[:, 0].astype(float), _sb[:, 1].astype(float),
                                    _sb[:, 2].astype(float), _sb[:, 3].astype(float) * 1e6,
                                    E_B478, _w0, rng, 'cone')
    _q = to_charge(_d)
    _y = float(_w[_q >= Q_THR].sum())          # per neutron emitted into the cone
    _rate = _y * _sr                            # per unit source strength [/sr]
    if _base is None: _base = _rate
    _cone_rows.append((_cd, _sr, _y, _rate / _base))
    print(f"  {_cd:14.0f} deg {_sr:8.3f} {_y:19.3e} {_rate/_base:10.2f}")
print("  (a 20x20 cm block whose face is 5 cm from the source subtends ~63 deg;")
print("   the slab model is laterally infinite, so the large-angle rows are upper bounds)")

# ═══════════════════════════════════════════════════════════════════════════
#  §11  DIE-AWAY BY CHARGE BAND
# ═══════════════════════════════════════════════════════════════════════════
def fit_exp_flat(tc, y, yerr=None):
    """Least-squares A exp(-t/lam) + C by scanning lam (linear in A, C)."""
    best = (None, 1e99)
    w = 1.0 / np.maximum(yerr, 1e-9) ** 2 if yerr is not None else np.ones_like(y)
    for lam in np.arange(0.8, 12.0, 0.01):
        X = np.column_stack([np.exp(-tc / lam), np.ones_like(tc)])
        WX = X * w[:, None]
        try:
            beta = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            continue
        if beta[0] < 0: continue
        chi2 = float(np.sum(w * (y - X @ beta) ** 2))
        if chi2 < best[1]: best = (lam, chi2)
    return best[0]


TB = np.arange(0.5, 18.51, 0.5)
TC = 0.5 * (TB[1:] + TB[:-1])
BANDS = [('alpha ROI', 0.45, 0.85), ('478 Compton 1-2.5', 1.0, 2.5),
         ('above edge 3-4.4', 3.0, 4.4), ('clipped >4.4', 4.4, 1e9)]
print("\n" + "=" * 78)
print("DIE-AWAY BY CHARGE BAND — MC vs DATA")
print("=" * 78)
print(f"  {'band':22s} {'MC lambda':>10s} {'data lambda':>14s}")
mc_lam = {}
for nm, lo, hi in BANDS:
    hh = np.zeros(TC.size)
    for tag in ORDER:
        if tag not in EVT: continue
        e = EVT[tag]
        m = (e['q'] >= lo) & (e['q'] < hi)
        hh += np.histogram(e['tau'][m], bins=TB, weights=e['w'][m])[0]
    lam = fit_exp_flat(TC, hh, np.sqrt(np.maximum(hh, 1e-12)) * 0 + np.maximum(hh, 1e-12) * 0.05) \
        if hh.sum() > 0 else float('nan')
    mc_lam[nm] = lam
    d = DATA_LAMBDA[nm]
    print(f"  {nm:22s} {lam:10.2f} {d[0]:9.2f} +- {d[1]:.2f}")
_e = EVT['tile capture (alpha)']
_h = np.histogram(_e['tau'], bins=TB, weights=_e['w'])[0]
_lam_tile = fit_exp_flat(TC, _h, np.maximum(_h, 1e-12) * 0.03)
_e = EVT['B-10 478 keV']
_h = np.histogram(_e['tau'], bins=TB, weights=_e['w'])[0]
_lam_sh = fit_exp_flat(TC, _h, np.maximum(_h, 1e-12) * 0.03)
print(f"  {'(pure tile capture)':22s} {_lam_tile:10.2f}      -- data alpha ROI 3.33")
print(f"  {'(pure shield gamma)':22s} {_lam_sh:10.2f}      -- data 478 band  4.06")


# ═══════════════════════════════════════════════════════════════════════════
#  §12  FIGURES
# ═══════════════════════════════════════════════════════════════════════════
COL = {'B-10 478 keV': '#2e7d32', 'H 2.22 MeV': '#1565c0',
       'Fe 7.6 MeV cascade': '#c62828', 'tile capture (alpha)': '#f9a825',
       'tile H capture': '#6a1b9a', 'neutron recoil': '#546e7a'}

fig, axs = plt.subplots(1, 2, figsize=(16, 6.4))
for ax, logy in zip(axs, (False, True)):
    bot = np.zeros(QC.size)
    for tag in ORDER:
        if tag not in STACK: continue
        h = STACK[tag]
        if h.sum() <= 0: continue
        ax.fill_between(QC, bot, bot + h, step='mid', color=COL[tag], alpha=0.75,
                        label=LBL[tag] + f'  ({h.sum():.0f})')
        bot = bot + h
    if DAT is not None:
        db, dn, de = DAT['r14_bins'], DAT['r14_net'], DAT['r14_err']
        dc = 0.5 * (db[1:] + db[:-1])
        k = dc >= Q_THR
        ax.errorbar(dc[k], dn[k], de[k], fmt='ko', ms=3.2, lw=1,
                    label='DATA 20260814 net excess')
        db2, dn2, de2 = DAT['r20_bins'], DAT['r20_net'], DAT['r20_err']
        dc2 = 0.5 * (db2[1:] + db2[:-1])
        k2 = dc2 >= Q_THR
        s20 = dn[(dc >= 1) & (dc < 3)].sum() / max(dn2[(dc2 >= 1) & (dc2 < 3)].sum(), 1)
        ax.plot(dc2[k2], dn2[k2] * s20, 'x', color='dimgray', ms=4,
                label=f'DATA 20260820 (3-D print) x{s20:.2f}')
        # composite data curve: run 14 below its ceiling, run 20 (scaled) above
        dmerge = np.where(QC < Q_CLIP_14, np.interp(QC, dc, dn, left=0, right=0),
                          np.interp(QC, dc2, dn2 * s20, left=0, right=0))
        res = dmerge - tot_pred
        kk = QC >= Q_THR
        ax.plot(QC[kk], np.maximum(res[kk], 1e-3), 'r--', lw=1.5,
                label='DATA - MC = the missing component')
    ax.axvline(2.75, color='k', ls='--', lw=1)
    ax.axvline(Q_CLIP_14, color='k', ls=':', lw=1)
    ax.text(2.77, ax.get_ylim()[1] * 0.6, '478 keV\nCompton edge', fontsize=9)
    ax.set_xlabel('charge  [V.ns]   (113 keVee / V.ns)', fontsize=11)
    ax.set_ylabel(f'counts per {N_REC:,} gate-tagged records / 0.1 V.ns', fontsize=11)
    ax.set_xlim(0, 9.5); ax.grid(alpha=0.25)
    if logy:
        ax.set_yscale('log'); ax.set_ylim(0.3, None)
axs[0].legend(fontsize=8.5, loc='upper right')
axs[0].set_title('MC prediction (stacked) vs measured delayed excess — linear', fontsize=11)
axs[1].set_title('same, log', fontsize=11)
fig.suptitle('Predicted post-gate charge spectrum, normalised to the measured tile-capture rate',
             fontsize=12)
plt.tight_layout()
plt.savefig('gamma_mc_spectrum.png', dpi=120)
print("\nSaved -> gamma_mc_spectrum.png")

# --- capture vertices and gamma origin --------------------------------------
fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.8))
ax2[0].hist(vB[:, 2], bins=70, color=COL['B-10 478 keV'], alpha=0.8, label='B-10')
ax2[0].hist(vH[:, 2], bins=70, color=COL['H 2.22 MeV'], alpha=0.8,
            weights=np.full(vH.shape[0], vB.shape[0] / max(vH.shape[0], 1) * 0.05), label='H (x scaled)')
ax2[0].set_xlabel('capture depth z [cm]'); ax2[0].set_ylabel('captures')
ax2[0].legend(fontsize=9); ax2[0].grid(alpha=0.25)
ax2[0].set_title(f'capture vertices, {N_NEUTRON:,} neutrons', fontsize=10)
tb2 = np.logspace(-2, 2, 60)
ax2[1].hist(vB[:, 3] * 1e6, bins=tb2, color=COL['B-10 478 keV'], alpha=0.8, label='B-10 shield')
ax2[1].hist(vF[:, 3] * 1e6, bins=tb2, color=COL['Fe 7.6 MeV cascade'], alpha=0.8,
            weights=np.full(vF.shape[0], 30.0), label='Fe steel (x30)')
ax2[1].set_xscale('log'); ax2[1].set_xlabel('capture time [us]'); ax2[1].legend(fontsize=9)
ax2[1].grid(alpha=0.25)
ax2[1].set_title(f'B-10 median {np.median(vB[:,3])*1e6:.2f} us', fontsize=10)
for tag in ['B-10 478 keV', 'H 2.22 MeV', 'Fe 7.6 MeV cascade']:
    e = EVT[tag]
    m = e['q'] >= Q_THR
    h = np.histogram(e['dep'][m] * 1e3 / KEVEE_PER_VNS, bins=QB, weights=e['w'][m])[0]
    ax2[2].step(QC, h / max(h.sum(), 1e-30), where='mid', color=COL[tag], lw=1.6, label=LBL[tag])
ax2[2].set_yscale('log'); ax2[2].set_xlabel('charge [V.ns]'); ax2[2].set_ylabel('shape (area 1)')
ax2[2].set_xlim(0, 12); ax2[2].legend(fontsize=8); ax2[2].grid(alpha=0.25)
ax2[2].set_title('per-species deposit shape in the tile', fontsize=10)
plt.tight_layout(); plt.savefig('gamma_mc_sources.png', dpi=120)
print("Saved -> gamma_mc_sources.png")

# --- time profiles -----------------------------------------------------------
def reb(h, k=2):
    n = (h.size // k) * k
    return h[:n].reshape(-1, k).sum(1)


TB2 = TB[::2]
TC2 = 0.5 * (TB2[1:] + TB2[:-1])
BCOL = {'alpha ROI': '#f9a825', '478 Compton 1-2.5': '#2e7d32',
        'above edge 3-4.4': '#1565c0', 'clipped >4.4': '#c62828'}
DKEY = {'alpha ROI': 'alpha', '478 Compton 1-2.5': 'c478',
        'above edge 3-4.4': 'above', 'clipped >4.4': 'clip'}
fig3, ax3 = plt.subplots(1, 2, figsize=(14.5, 5.6))
for nm, lo, hi in BANDS:
    hh = np.zeros(TC.size)
    for tag in ORDER:
        if tag not in EVT: continue
        e = EVT[tag]
        m = (e['q'] >= lo) & (e['q'] < hi)
        hh += np.histogram(e['tau'][m], bins=TB, weights=e['w'][m])[0]
    if hh.sum() <= 0: continue
    h2 = reb(hh)
    nrm = h2[TC2 < 5].sum()
    ax3[0].step(TC2, h2 / max(nrm, 1e-30), where='mid', lw=1.9, color=BCOL[nm],
                label=f'MC  {nm}: {mc_lam[nm]:.2f} us')
    if DAT is not None:
        d2 = reb(DAT['r14_t_' + DKEY[nm]].astype(float))
        ax3[0].step(TC2, d2 / max(d2[TC2 < 5].sum(), 1), where='mid', lw=1.5, ls='--',
                    color=BCOL[nm],
                    label=f'data {nm}: {DATA_LAMBDA[nm][0]:.2f} us')
ax3[0].set_yscale('log'); ax3[0].set_xlabel('tau after gate end [us]', fontsize=11)
ax3[0].set_ylabel('normalised to the 0.5-5 us integral', fontsize=11)
ax3[0].legend(fontsize=7.6, ncol=2); ax3[0].grid(alpha=0.25)
ax3[0].set_title('post-gate time profile by charge band: MC (solid) vs data (dashed)',
                 fontsize=11)

for tag, col, lb in [('tile capture (alpha)', '#f9a825', 'MC tile capture (alpha)'),
                     ('B-10 478 keV', '#2e7d32', 'MC shield 478 keV gamma')]:
    e = EVT[tag]
    h2 = reb(np.histogram(e['tau'], bins=TB, weights=e['w'])[0])
    ax3[1].step(TC2, h2 / max(h2[TC2 < 5].sum(), 1e-30), where='mid', lw=2.0,
                color=col, label=lb)
if DAT is not None:
    d2 = reb(DAT['r14_t_alpha'].astype(float))
    ax3[1].step(TC2, d2 / max(d2[TC2 < 5].sum(), 1), where='mid', lw=1.6, ls='--',
                color='k', label='data alpha ROI (0.45-0.85 V.ns)')
    d2 = reb(DAT['r14_t_clip'].astype(float))
    ax3[1].step(TC2, d2 / max(d2[TC2 < 5].sum(), 1), where='mid', lw=1.6, ls=':',
                color='k', label='data clipped (>4.4 V.ns)')
ax3[1].set_yscale('log'); ax3[1].set_xlabel('tau after gate end [us]', fontsize=11)
ax3[1].legend(fontsize=8.5); ax3[1].grid(alpha=0.25)
ax3[1].set_title(f'the two MC clocks: shield gamma {_lam_sh:.1f} us, '
                 f'tile capture {_lam_tile:.1f} us', fontsize=11)
plt.tight_layout(); plt.savefig('gamma_mc_timing.png', dpi=120)
print("Saved -> gamma_mc_timing.png")

# ═══════════════════════════════════════════════════════════════════════════
#  §13  SUMMARY LEDGER
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("SHORTFALL LEDGER (data / MC), normalisation = measured tile captures")
print("=" * 78)
if DAT is not None:
    dn = DAT['r14_net']; db = DAT['r14_bins']
    jthr = np.searchsorted(db, Q_THR) - 1
    j275 = np.searchsorted(db, 2.75) - 1
    clipnet = float(DAT['r14_clipnet'][0])
    rows = [('total delayed excess >0.35 V.ns', dn[jthr:].sum() + clipnet, tot_pred.sum()),
            ('above the 478 edge (>2.75)', dn[j275:].sum() + clipnet, tot_pred[i275:].sum()),
            ('clipped (>4.4 V.ns)', clipnet, tot_pred[i440:].sum())]
    for nm, dv, mv in rows:
        print(f"  {nm:36s} data {dv:8.0f}   MC {mv:8.0f}   data/MC {dv/max(mv,1e-9):7.2f}")
# ═══════════════════════════════════════════════════════════════════════════
#  §14  SHAPE COMPARISON, BIN BY BIN
# ═══════════════════════════════════════════════════════════════════════════
if DAT is not None:
    print("\n" + "=" * 78)
    print("SHAPE: data vs MC per charge band  (MC normalised to tile captures)")
    print("=" * 78)
    db, dn = DAT['r14_bins'], DAT['r14_net']
    print(f"  {'band [V.ns]':>14s} {'data':>9s} {'MC':>9s} {'d/MC':>7s} {'MC uncoll':>10s}"
          f" {'d/unc':>6s}   MC composition")
    for lo, hi in [(0.35, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
                   (2.5, 2.75), (2.75, 3.5), (3.5, 4.4)]:
        j0, j1 = np.searchsorted(db, lo) - 1, np.searchsorted(db, hi) - 1
        k0, k1 = np.searchsorted(QB, lo) - 1, np.searchsorted(QB, hi) - 1
        dv = dn[j0:j1].sum(); mv = tot_pred[k0:k1].sum()
        cm = "  ".join(f"{LBL[t].split()[0]}:{100*STACK[t][k0:k1].sum()/max(mv,1e-9):.0f}%"
                       for t in ORDER if t in STACK and STACK[t][k0:k1].sum() > 0.02 * mv)
        uv = tot_unc[k0:k1].sum()
        print(f"  {lo:5.2f}-{hi:5.2f}   {dv:9.0f} {mv:9.0f} {dv/max(mv,1e-9):7.2f} {uv:10.0f}"
              f" {dv/max(uv,1e-9):6.2f}   {cm}")
    print(f"  {'>4.4 (clipped)':>14s} {float(DAT['r14_clipnet'][0]):9.0f} "
          f"{tot_pred[i440:].sum():9.0f} {float(DAT['r14_clipnet'][0])/max(tot_pred[i440:].sum(),1e-9):7.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  §15  WHERE COULD THE MISSING SIGNAL COME FROM?  (quantitative ranking)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("NEUTRON BOOK-KEEPING — populations available to feed unmodelled gammas")
print("=" * 78)
p_hit  = N_HIT / N_ARR
p_miss = (FAT_C['tran'] - N_HIT) / N_ARR
p_back = FAT_C['back'] / N_ARR
esc = WALK['Eesc']; mesc = np.isfinite(esc)
p_leak = mesc.sum() / NWALK * p_hit
p_leak_sub = ((esc < 1.0) & mesc).sum() / NWALK * p_hit
print(f"  per cone-emitted neutron:")
print(f"    reach the tile                       {p_hit:.3e}")
print(f"    leak back OUT of the tile            {p_leak:.3e}   "
      f"({100*p_leak/p_hit:.1f} % of arrivals)")
print(f"      ... of which below 1 eV            {p_leak_sub:.3e}")
print(f"    pass the shield but MISS the tile    {p_miss:.3e}   "
      f"({p_miss/p_hit:.0f} x the tile-hit flux)")
print(f"    backscattered out of the shield face {p_back:.3e}")

need_tot = (dtot - tot_pred.sum()) / (N_REC * NORM) if DAT is not None else float('nan')
need_hard = (float(DAT['r14_clipnet'][0]) - tot_pred[i440:].sum()) / (N_REC * NORM) \
    if DAT is not None else float('nan')
need_mid = ((dn[j275:].sum() - float(DAT['r14_clipnet'][0]) * 0)
            - tot_pred[i275:].sum()) / (N_REC * NORM) if DAT is not None else float('nan')
print(f"\n  MISSING rate, per cone-emitted neutron (normalisation = tile captures,")
print(f"  {NORM:.0f} cone-emitted neutrons per gate-tagged record):")
print(f"    total  >0.35 V.ns : {need_tot:.3e}")
print(f"    above the 478 edge: {need_mid:.3e}")
print(f"    hard   >4.4  V.ns : {need_hard:.3e}")

# ═══════════════════════════════════════════════════════════════════════════
#  §15b  PER-SPECIES BOOK-KEEPING: how many captures would be needed?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("HOW MANY CAPTURES WOULD IT TAKE?  (>4.4 V.ns excess = "
      f"{float(DAT['r14_clipnet'][0]):.0f} events)")
print("=" * 78)
_have = {'B-10 478 keV': FAT_C['B10'] / N_ARR, 'H 2.22 MeV': FAT_C['H'] / N_ARR,
         'Fe 7.6 MeV cascade': FAT_C['Fe'] / N_ARR}
print(f"  {'species (in the shield/steel)':32s} {'p_det(>4.4)':>12s} {'captures now':>13s} "
      f"{'needed':>10s} {'factor':>8s}")
for tag, nhave in _have.items():
    e = EVT[tag]
    m = e['q'] >= Q_CLIP_14
    y = float(net_w(e['tau'], e['w'])[m].sum())       # per cone-emitted neutron
    pdet = y / max(nhave, 1e-30)
    nneed = need_hard / max(pdet, 1e-30)
    print(f"  {LBL[tag]:32s} {pdet:12.2e} {nhave:13.2e} {nneed:10.2e} "
          f"{nneed/max(nhave,1e-30):8.0f}x")
print("  (a 'factor' above ~1.3 is excluded: only 100 % of the neutrons exist)")

# ---- probe: a capture sitting right against the tile face ------------------
print("\n  PROBE — detection probability for a capture 1 mm in FRONT of the tile")
print("  (i.e. in the tile's own support / frame / housing), by gamma energy:")
_npb = 40000
_zp = np.full(_npb, D_TILE - 0.1)
_xp = (rng.random(_npb) - 0.5) * 6.0
_yp = (rng.random(_npb) - 0.5) * 6.0
for _lbl, _Eg, _mult in [('B-10 478 keV', E_B478, Y_B478),
                         ('H 2223 keV', E_H2223, 1.0),
                         ('Fe 7.6 MeV cascade', 7.638, 2.0),
                         ('Al-27 7.72 MeV', 7.724, 2.0)]:
    _d, _w, _t, _g = shield_photons(_xp, _yp, _zp, np.zeros(_npb),
                                    np.full(_npb, _Eg), _mult / _npb, rng, 'probe')
    _q = to_charge(_d)
    _p_thr = float(_w[_q >= Q_THR].sum())
    _p_hard = float(_w[_q >= Q_CLIP_14].sum())   # prompt: no time weighting
    print(f"    {_lbl:22s} p(>0.35 V.ns) = {_p_thr:.4f}   p(>4.4 V.ns) = {_p_hard:.4f}")
    if _lbl.startswith('Fe'):
        _fe_probe = _p_hard
print(f"  -> captures per cone-emitted neutron needed on such adjacent material:")
print(f"     {need_hard/max(_fe_probe,1e-9):.2e}  = {100*need_hard/max(_fe_probe,1e-9):.2f} % of all")
print(f"     cone-emitted neutrons, against {100*p_leak:.2f} % that leak out of the tile and")
print(f"     {100*p_miss:.1f} % that pass the shield and miss the tile.")
print(f"     required capture fraction of the leak-out flux : "
      f"{need_hard/max(_fe_probe,1e-9)/max(p_leak,1e-30):.2f}")
print(f"     required capture fraction of the missed-tile flux: "
      f"{need_hard/max(_fe_probe,1e-9)/max(p_miss,1e-30):.4f}")


print("\n" + "=" * 78)
print("CANDIDATE RANKING for the missing signal")
print("=" * 78)
f_cone = None
for cd, sr, y, x in _cone_rows:
    if f_cone is None and x * (tot_pred.sum() - 3300) >= (dtot - 3300):
        f_cone = (cd, x)
_soft_mc = tot_pred.sum() - STACK['tile capture (alpha)'].sum()
print(f"  (A) shield illuminated beyond the 20 deg cone")
print(f"      needed multiplier on ALL shield gammas to close the TOTAL: "
      f"x{(dtot - 3300)/max(_soft_mc,1e-9):.2f}")
print(f"      -> illuminated half-angle ~ "
      f"{[f'{cd:.0f} deg (x{x:.1f})' for cd, sr, y, x in _cone_rows]}")
print(f"      supplies of the >4.4 V.ns excess: "
      f"{100*tot_pred[i440:].sum()*(dtot-3300)/max(_soft_mc,1e-9)/max(float(DAT['r14_clipnet'][0]),1):.1f} %"
      f"   (the 478 keV line CANNOT exceed 2.75 V.ns except by resolution)")

# (B) capture on structure adjacent to the tile, fed by tile leakage
det_per_cap = _fe_probe               # measured by the MC probe above
eff_needed = need_hard / max(p_leak, 1e-30) / det_per_cap
print(f"\n  (B) (n,gamma) on STRUCTURE next to the tile (support, frame, PMT can, table)")
print(f"      fed by the {100*p_leak/p_hit:.0f} % of tile arrivals that leak out again")
print(f"      detection prob. per adjacent Fe capture, from the MC probe: "
      f"{det_per_cap:.4f}")
print(f"      required  Omega/4pi x P(capture on that structure) = {eff_needed:.3f}")
print(f"      -> needs {100*eff_needed:.0f} % of the leak-out flux to capture on it "
      f"({'possible' if eff_needed < 1 else 'NOT ENOUGH FLUX'})")
eff_needed2 = need_hard / max(p_miss, 1e-30) / det_per_cap
print(f"      same, fed instead by the missed-tile flux: {eff_needed2:.4f}")

# (C) the modelled steel plate with more thermal flux
fe_now = STACK['Fe 7.6 MeV cascade'][i440:].sum()
print(f"\n  (C) the MODELLED 1 cm steel plate at z=20-21 cm")
print(f"      it now supplies {fe_now:.0f} of the {float(DAT['r14_clipnet'][0]):.0f} clipped events; "
      f"closing the gap needs x{float(DAT['r14_clipnet'][0])/max(fe_now,1e-9):.0f} more Fe captures")
print(f"      and Fe captures in the plate peak at tens of us (median "
      f"{np.median(vF[:, 3])*1e6:.0f} us) -- the wrong clock for a 3.2 us die-away")
print(f"      MC Fe capture rate {FAT_C['Fe']/N_ARR:.2e} /emitted "
      f"(0.10 %); x{float(DAT['r14_clipnet'][0])/max(fe_now,1e-9):.0f} would need "
      f"{100*FAT_C['Fe']/N_ARR*float(DAT['r14_clipnet'][0])/max(fe_now,1e-9):.0f} % of all neutrons "
      f"to capture on Fe -- excluded")

# (D) room return
print(f"\n  (D) room / floor return (H 2.22 MeV, concrete)")
print(f"      timing: 0.3-2 ms lifetime -> flat inside the 18.5 us record; the observed")
print(f"      hard band decays with lambda = {DATA_LAMBDA['clipped >4.4'][0]:.2f} us. Excluded as the")
print(f"      SHAPE of the excess (it is the flat pedestal that the late window removes).")

# (E) reflection behind the tile
print(f"\n  (E) neutron albedo from behind the tile (table, floor)")
print(f"      would raise TILE CAPTURES (alpha line) and lengthen the tile clock; it")
print(f"      cannot make deposits above 0.9 V.ns.  Supplies 0 % of the >4.4 excess.")

# (F) pile-up
print(f"\n  (F) pile-up inside the {COINC_NS:.0f} ns integration window")
print(f"      two 478 keV Compton events sum to <= 5.5 V.ns and would decay with")
print(f"      lambda/2 = {DATA_LAMBDA['478 Compton 1-2.5'][0]/2:.2f} us; the clipped band measures "
      f"{DATA_LAMBDA['clipped >4.4'][0]:.2f} +- {DATA_LAMBDA['clipped >4.4'][1]:.2f} us -> disfavoured but")
print(f"      not excluded for the 2.75-4.4 V.ns region.")
print("\ndone.")
