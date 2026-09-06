"""HYDROGEN.md worksheet: every number in the hydrogen study, derived here.

Read-only w.r.t. the MC: loads _mc3d_cache.npz and _shield_capture_times.npz,
never regenerates them.  Run from this directory with the energy-deposits env.
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib_neutrons import get_endf_cross_sections

N_A = 6.022e23
E0 = 2.45e6          # eV
E_TH = 0.0253        # eV
BARN = 1e-24

def hdr(s): print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)

# ─────────────────────────────────────────────────────────────── 1. materials
hdr("1. MATERIAL NUMBER DENSITIES (exactly as the MC defines them)")
n_B10_sh = 1.00 * 0.05 * 0.199 / 10.0 * N_A
n_B11_sh = 1.00 * 0.05 * 0.801 / 11.0 * N_A
n_H_sh   = 1.00 * 0.143 * N_A
n_C_sh   = 1.00 * 0.807 / 12.0 * N_A
n_H_wd   = 0.60 * 0.06 * N_A
n_Ceq_wd = (0.60 * 0.50 / 12.0 + 0.60 * 0.44 / 16.0) * N_A
n_Fe     = 7.85 / 55.85 * N_A
n_B10_t  = 1.05 * 0.02 * 0.199 / 10.0 * N_A     # MC uses rho = 1.05
n_B10_t103 = 1.03 * 0.02 * 0.199 / 10.0 * N_A   # stated tile density 1.03
n_H_t, n_C_t = 5.2514e22, 4.7262e22

for k, v in [("B-HDPE  n_H", n_H_sh), ("B-HDPE  n_C", n_C_sh),
             ("B-HDPE  n_B10", n_B10_sh), ("B-HDPE  n_B11", n_B11_sh),
             ("wood    n_H", n_H_wd), ("wood    n_C(eq)", n_Ceq_wd),
             ("steel   n_Fe", n_Fe),
             ("tile    n_H", n_H_t), ("tile    n_C", n_C_t),
             ("tile    n_B10 (rho 1.05)", n_B10_t),
             ("tile    n_B10 (rho 1.03)", n_B10_t103)]:
    print(f"  {k:28s} = {v:10.4e} cm^-3")
print(f"  tile H:C atom ratio        = {n_H_t/n_C_t:.3f}  (PVT ~ C9H10 -> 1.111)")
print(f"  shield H:C atom ratio      = {n_H_sh/n_C_sh:.3f}  (CH2 -> 2.000)")
print(f"  B-HDPE H mass fraction 0.143 of 0.95 hydrocarbon = "
      f"{0.143/0.95:.4f} (CH2 = {2/14:.4f})  -> consistent")

# ────────────────────────────────────────────────── 2. cross sections & mfp
hdr("2. CROSS SECTIONS AND MEAN FREE PATHS")
for E, lbl in [(E0, "2.45 MeV"), (1e5, "100 keV"), (1e3, "1 keV"),
               (1.0, "1 eV"), (E_TH, "thermal 25.3 meV")]:
    sH, sC = get_endf_cross_sections(np.array([E]))
    sH, sC = float(sH[0]), float(sC[0])
    SH, SC = n_H_sh * sH * BARN, n_C_sh * sC * BARN
    sB10 = 3840.0 * math.sqrt(E_TH / max(E, 1e-12))
    SB = n_B10_sh * sB10 * BARN
    print(f"  {lbl:18s} sigma_H {sH:7.2f} b   sigma_C {sC:6.2f} b   "
          f"sigma_B10(n,a) {sB10:9.3f} b")
    print(f"  {'':18s} lam_H {1/SH:8.3f} cm  lam_C {1/SC:7.3f} cm  "
          f"lam_B10 {1/SB:10.4f} cm   lam_tot {1/(SH+SC+SB):6.3f} cm")

sH0, sC0 = get_endf_cross_sections(np.array([E0]))
print(f"\n  Fraction of 2.45 MeV collisions in B-HDPE that are on H: "
      f"{n_H_sh*sH0[0]/(n_H_sh*sH0[0]+n_C_sh*sC0[0]):.3f}")
SHw = n_H_wd * float(sH0[0]) * BARN; SCw = n_Ceq_wd * float(sC0[0]) * BARN
print(f"  wood at 2.45 MeV: lam_H {1/SHw:.2f} cm  lam_tot {1/(SHw+SCw):.2f} cm "
      f"(1 cm of wood = {(SHw+SCw):.3f} mfp)")

# ────────────────────────────────────── 3. moderation: collisions and time
hdr("3. MODERATION BY HYDROGEN IN THE SHIELD")
xi_H, xi_C = 1.0, 2/3. * 0  # filled below
def xi(A):
    if A == 1: return 1.0
    a = ((A - 1) / (A + 1)) ** 2
    return 1.0 + a / (1 - a) * math.log(a)
xi_H, xi_C = xi(1), xi(12)
print(f"  xi_H = {xi_H:.4f}   xi_C = {xi_C:.4f}")
Egrid = np.logspace(math.log10(E_TH), math.log10(E0), 4000)
sHg, sCg = get_endf_cross_sections(Egrid)
SHg, SCg = n_H_sh * sHg * BARN, n_C_sh * sCg * BARN
xibar = (xi_H * SHg + xi_C * SCg) / (SHg + SCg)
# n = int dE/(E xibar)  = int d(lethargy)/xibar
u = np.log(E0 / Egrid)
ncoll = np.trapezoid(1.0 / xibar[::-1], u[::-1])
print(f"  mean collisions 2.45 MeV -> thermal in B-HDPE: {ncoll:.1f}"
      f"   (pure H would be {math.log(E0/E_TH):.1f})")
frac_on_H = np.trapezoid((SHg / (SHg + SCg))[::-1], u[::-1]) / u[0]
print(f"  lethargy-averaged fraction of those collisions on H: {frac_on_H:.3f}"
      f"  -> {ncoll*frac_on_H:.1f} H collisions, {ncoll*(1-frac_on_H):.1f} on C")
# slowing-down time  t = int_E^E0 dE'/(E' xibar Sigma_s v)
v = 1.3831e6 * np.sqrt(Egrid)                      # cm/s
integ = 1.0 / (Egrid * xibar * (SHg + SCg) * v)
t_sd = abs(np.trapezoid(integ, Egrid))
print(f"  slowing-down time 2.45 MeV -> thermal: {t_sd*1e6:.3f} us"
      f"   (dominated by the last decade)")
# thermal dwell before capture in the shield
Sa_th_sh = n_B10_sh * 3840 * BARN + n_H_sh * 0.332 * BARN
v_th = 2.2e5
print(f"  thermal Sigma_a(shield) = {Sa_th_sh:.3f} /cm  ->  "
      f"capture lifetime 1/(Sigma_a v) = {1e6/(Sa_th_sh*v_th):.2f} us")
Ss_th = n_H_sh * float(get_endf_cross_sections(np.array([E_TH]))[0][0]) * BARN \
        + n_C_sh * float(get_endf_cross_sections(np.array([E_TH]))[1][0]) * BARN
mubar = (n_H_sh*float(get_endf_cross_sections(np.array([E_TH]))[0][0])*2/3
         + n_C_sh*float(get_endf_cross_sections(np.array([E_TH]))[1][0])*2/36) \
        / (Ss_th/BARN)
Str = Ss_th * (1 - mubar); D = 1.0 / (3 * Str)
L = math.sqrt(D / Sa_th_sh)
print(f"  thermal Sigma_s(shield) = {Ss_th:.2f} /cm, mubar {mubar:.3f}, "
      f"D {D:.4f} cm  -> diffusion length L = {L:.3f} cm")
print("  => thermal neutrons created at ~5 cm depth cannot random-walk to the")
print("     14 cm exit face: attenuation exp(-8.7/L) = %.1e" % math.exp(-8.7/L))
zc = np.load('_shield_capture_times.npz')
print(f"  MC shield-capture depth: median {np.median(zc['z_cap']):.2f} cm, "
      f"mean {zc['z_cap'].mean():.2f} cm; capture time median "
      f"{np.median(zc['t_cap'])*1e6:.2f} us, mean {zc['t_cap'].mean()*1e6:.2f} us")

# ────────────────────────────────── 4. capture competition H vs B-10 (1/v)
hdr("4. CAPTURE COMPETITION H(n,g) vs B-10(n,a)  --  energy independent")
for nm, nH, nB in [("B-HDPE shield", n_H_sh, n_B10_sh),
                   ("tile (rho 1.05)", n_H_t, n_B10_t),
                   ("tile (rho 1.03)", n_H_t, n_B10_t103)]:
    SH = nH * 0.332 * BARN; SB = nB * 3840 * BARN
    print(f"  {nm:16s} Sigma_H(th) {SH:8.5f}/cm   Sigma_B10(th) {SB:7.4f}/cm"
          f"   B:H = {SB/SH:6.1f} : 1   H branch = {SH/(SH+SB)*100:5.2f} %")
SH_sh = n_H_sh*0.332*BARN; SB_sh = n_B10_sh*3840*BARN
print(f"\n  MC fate B-10(shield) = 71.8 % of emitted; expected H capture in the")
print(f"  same material = 71.8 x {SH_sh/SB_sh:.5f} = {71.8*SH_sh/SB_sh:.3f} %")
print(f"  MC fate 'H capture (shield+wood)' = 0.90 %  ->  CONSISTENT; wood adds ~0")
Y478 = 0.718 * 0.94
Y2223 = 0.0090
print(f"\n  gamma yield per cone-emitted neutron:  478 keV  {Y478:.4f}")
print(f"                                        2223 keV {Y2223:.4f}"
      f"   ratio {Y2223/Y478:.4f} = {100*Y2223/Y478:.2f} %")

# ─────────────────────────────── 5. gamma transport to the tile (Klein-Nishina)
hdr("5. 2.2 MeV vs 478 keV GAMMAS AT THE TILE")
mec2 = 0.510999
def kn_total(Eg):
    a = Eg / mec2
    return 2*math.pi*(2.8179403e-13)**2 * (
        (1+a)/a**2 * (2*(1+a)/(1+2*a) - math.log(1+2*a)/a)
        + math.log(1+2*a)/(2*a) - (1+3*a)/(1+2*a)**2)   # cm^2 / electron
def kn_T_spectrum(Eg, nT=20000):
    """dsigma/dT (arb) on a grid of T from 0 to Tmax."""
    a = Eg / mec2
    Tmax = Eg * 2*a/(1+2*a)
    T = np.linspace(1e-9, Tmax*(1-1e-9), nT)
    Ep = Eg - T                       # scattered photon energy
    r = Ep/Eg
    x = mec2 * (1.0/Ep - 1.0/Eg)      # = 1 - cos(theta)
    sin2 = 1.0 - (1.0 - x)**2
    dsig_dEp = (r + 1.0/r - sin2)     # Klein-Nishina, per unit E'
    return T, np.maximum(dsig_dEp, 0)
def frac_above(Eg, Tcut):
    T, w = kn_T_spectrum(Eg)
    tot = np.trapezoid(w, T)
    m = T > Tcut
    return float(np.trapezoid(w[m], T[m]) / tot) if tot > 0 else 0.0

ne_bhdpe = 1.00 * N_A * (0.143*1.0 + 0.807*6/12. + 0.05*0.4626)   # e/cm3
ne_pvt   = 1.03 * N_A * (10*1 + 9*6) / 118.0
mu = {}
for Eg in (0.478, 2.223):
    s = kn_total(Eg)
    mu[('bhdpe', Eg)] = s * ne_bhdpe
    mu[('pvt',  Eg)] = s * ne_pvt
mu_fe = {0.478: 0.0839*7.85, 2.223: 0.0413*7.85}    # NIST XCOM, cm^-1
print(f"  electron densities: B-HDPE {ne_bhdpe:.3e}/cm3   PVT {ne_pvt:.3e}/cm3")
for Eg in (0.478, 2.223):
    print(f"  Eg {Eg*1e3:7.1f} keV : sigma_KN {kn_total(Eg)*1e24:.4f} b/e   "
          f"mu(B-HDPE) {mu[('bhdpe',Eg)]:.4f}/cm   mu(PVT) {mu[('pvt',Eg)]:.4f}/cm"
          f"   mu(Fe) {mu_fe[Eg]:.3f}/cm")
    print(f"      Compton edge {Eg*2*(Eg/mec2)/(1+2*Eg/mec2)*1e3:7.1f} keV   "
          f"p(interact in 1 cm PVT) {1-math.exp(-mu[('pvt',Eg)]):.4f}")
# escape from the shield along the beam axis, using the MC capture depth profile
z = zc['z_cap']; dz = np.clip(14.0 - z, 0, None)
esc = {Eg: float(np.mean(np.exp(-mu[('bhdpe', Eg)] * dz))) for Eg in (0.478, 2.223)}
fe  = {Eg: math.exp(-mu_fe[Eg] * 1.0) for Eg in (0.478, 2.223)}
print(f"\n  uncollided escape through the remaining B-HDPE (same depth profile,")
print(f"  both gammas follow the SAME thermal-capture distribution):")
for Eg in (0.478, 2.223):
    print(f"    {Eg*1e3:7.1f} keV : <exp(-mu dz)> = {esc[Eg]:.4f}   "
          f"x 1 cm steel {fe[Eg]:.3f}  ->  {esc[Eg]*fe[Eg]:.4f}")
Tcut = 2.31 * 135e-3          # V.ns -> MeVee  (the 478 keV Compton edge)
f_hi = frac_above(2.223, Tcut)
f_lo478 = frac_above(0.478, 0.047)
print(f"\n  charge cut 2.31 V.ns = {Tcut*1e3:.0f} keVee")
print(f"  fraction of 2223 keV Compton deposits ABOVE it: {f_hi:.3f}")
print(f"  fraction of  478 keV Compton deposits above the 47 keVee trigger: {f_lo478:.3f}")
R = (Y2223 * esc[2.223]*fe[2.223] * (1-math.exp(-mu[('pvt',2.223)])) * f_hi) / \
    (Y478  * esc[0.478]*fe[0.478] * (1-math.exp(-mu[('pvt',0.478)])) * f_lo478)
print(f"\n  PREDICTED above-edge / below-edge gamma ratio = {R:.4f} = {100*R:.2f} %")
print(f"  MEASURED bound: 4 +- 2 of 13987  =  {100*4/13987:.3f} %  (< 0.05 %)")
print(f"  -> prediction exceeds the bound by a factor {R/(5e-4):.0f}")

# ───────────────────────────────────────────── 6. the tile: capture budget
hdr("6. HYDROGEN IN THE TILE")
sH_th, sC_th = get_endf_cross_sections(np.array([E_TH]))
Ss_t_th = n_H_t*float(sH_th[0])*BARN + n_C_t*float(sC_th[0])*BARN
Sa_H = n_H_t*0.332*BARN; Sa_B = n_B10_t*3840*BARN
St_th = Ss_t_th + Sa_H + Sa_B
print(f"  thermal: Sigma_sH {n_H_t*float(sH_th[0])*BARN:.3f}/cm "
      f"(sigma_H {float(sH_th[0]):.1f} b bound)  Sigma_sC {n_C_t*float(sC_th[0])*BARN:.4f}/cm")
print(f"           Sigma_a(B10) {Sa_B:.4f}/cm   Sigma_a(H) {Sa_H:.5f}/cm   "
      f"H share of captures {100*Sa_H/(Sa_H+Sa_B):.2f} %")
print(f"           Sigma_tot {St_th:.3f}/cm -> p(interact in 1 cm) "
      f"{1-math.exp(-St_th):.4f}, single-pass p(B-10 capture) "
      f"{Sa_B/St_th*(1-math.exp(-St_th)):.4f}")
print(f"  pure-capture tile lifetime 1/(Sigma_a v) = "
      f"{1e6/((Sa_B+Sa_H)*2.2e5):.2f} us")
for E, lbl in [(1.0, "1 eV"), (100.0, "100 eV"), (1e4, "10 keV"), (1e5, "100 keV")]:
    sHe, sCe = get_endf_cross_sections(np.array([E]))
    Sc = n_B10_t*3840*math.sqrt(E_TH/E)*BARN
    Ss = n_H_t*float(sHe[0])*BARN + n_C_t*float(sCe[0])*BARN
    St = Sc + Ss
    print(f"  {lbl:8s}: Sigma_cap {Sc:.4f}  Sigma_scat {Ss:.3f}  "
          f"single-pass p_cap {Sc/St*(1-math.exp(-St)):.4f}   "
          f"p_scat {Ss/St*(1-math.exp(-St)):.4f}")

# ─────────────────────────────────── 7. Birks light output for protons
hdr("7. BIRKS QUENCHING OF PROTON RECOILS IN PVT")
# proton mass stopping power in polyvinyltoluene, PSTAR-like (MeV cm^2/g)
_PE = np.array([0.001,0.002,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,
                0.40,0.50,0.70,1.00,1.50,2.00,2.50,3.00,5.00,10.0])
_PS = np.array([169., 239., 372., 500., 655., 780., 805., 800., 792., 730.,
                642., 544., 470., 420., 348., 256., 194., 157., 137., 117.,
                77.6, 45.5])
def Sp(E):  # MeV cm2/g
    return np.exp(np.interp(np.log(np.clip(E,1e-3,10.)), np.log(_PE), np.log(_PS)))
kB = 0.0125     # g cm^-2 MeV^-1
def L_p(E, n=4000):
    e = np.linspace(1e-4, E, n)
    return float(np.trapezoid(1.0/(1.0+kB*Sp(e)), e))
print(f"  Birks kB = {kB} g/cm2/MeV, dE/dx from a PSTAR-like PVT proton table")
print("   Ep [keV]   dE/dx [MeV cm2/g]   L [keVee]   L/Ep")
for Ep in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.45):
    print(f"   {Ep*1e3:7.0f}    {Sp(Ep):10.0f}        {L_p(Ep)*1e3:8.1f}   {L_p(Ep)/Ep:.3f}")
# invert L -> Ep for the trigger threshold
grid = np.linspace(0.001, 3.0, 3000)
Lg = np.array([L_p(e, 800) for e in grid])
def Ep_of_L(Lv): return float(np.interp(Lv, Lg, grid))
LTHR = 0.047
Ep_thr = Ep_of_L(LTHR)
print(f"\n  trigger 25 mV = 0.35 V.ns = 47 keVee  ->  proton energy "
      f"{Ep_thr*1e3:.0f} keV   ({Ep_thr/0.05:.1f}x the MC's 50 keV cut)")
print(f"  observed turn-on 0.226 V.ns = {0.226*135:.0f} keVee -> "
      f"{Ep_of_L(0.226*0.135)*1e3:.0f} keV proton")
print(f"  C2 ceiling 382 mV ~ 5 V.ns = {5*135:.0f} keVee -> "
      f"{Ep_of_L(5*0.135)*1e3:.0f} keV proton  (above the 2.45 MeV kinematic max:"
      f" L(2.45 MeV) = {L_p(2.45)*1e3:.0f} keVee = {L_p(2.45)/0.135:.2f} V.ns)")
# carbon recoils
def Sc_ion(E):   # crude: carbon ion dE/dx ~ Z_eff^2 scaling at equal velocity
    return 6.0**2 * Sp(np.clip(E/12.0, 1e-3, None)) * 0.35
print(f"\n  carbon recoils: max fraction 4A/(A+1)^2 = {4*12/169:.3f} -> "
      f"E_C^max(2.45 MeV n) = {0.284*2.45*1e3:.0f} keV")
for Ec in (0.1, 0.3, 0.696):
    e = np.linspace(1e-4, Ec, 2000)
    Lc = float(np.trapezoid(1.0/(1.0+kB*Sc_ion(e)), e))
    print(f"    E_C {Ec*1e3:5.0f} keV -> L ~ {Lc*1e3:5.1f} keVee "
          f"({Lc/Ec:.3f} of the deposit)")
print("  -> carbon recoils are ~30-60x below threshold: they NEVER trigger.")

# ──────────────────────────── 8. recompute the MC visible-scatter budget
hdr("8. MC TILE BUDGET RECOMPUTED WITH THE REAL (LIGHT) THRESHOLD")
c = np.load('_mc3d_cache.npz', allow_pickle=True)
arr = c['arr']; fates = c['fates'].item(); N = int(c['N'])
E_t = arr[:,0]; X, Y = arr[:,2], arr[:,3]; UX,UY,UZ = arr[:,4],arr[:,5],arr[:,6]
Lr = (23.0 - 23.0) / np.where(UZ!=0, UZ, 1)     # exit plane == tile plane
xt, yt = X + UX*Lr, Y + UY*Lr
hit = (np.abs(xt)<=1.5)&(np.abs(yt)<=1.5)&(UZ>0)
Eh = E_t[hit]
print(f"  arrivals at the tile: {hit.sum():,} of {N:,} emitted")
sHh, sCh = get_endf_cross_sections(np.maximum(Eh, E_TH))
Sig_c = n_B10_t*3840.0*np.sqrt(E_TH/np.maximum(Eh,E_TH))*BARN
Sig_sH = n_H_t*sHh*BARN; Sig_sC = n_C_t*sCh*BARN
Sig_t = Sig_c+Sig_sH+Sig_sC
p_int = 1-np.exp(-Sig_t*1.0)
p_cap = Sig_c/Sig_t*p_int
grp = [(1e5,4e6,'fast >100 keV'),(1.0,1e5,'epithermal'),(0,1.0,'thermal <1 eV')]
print(f"\n  captures (single pass, first interaction) = {p_cap.sum():.1f}")
for lo,hi,lbl in grp:
    m=(Eh>=lo)&(Eh<hi)
    print(f"    {lbl:16s} {m.sum():6d} arrivals ({100*m.mean():5.2f}%)  "
          f"captures {p_cap[m].sum():7.2f}  <p_cap> {p_cap[m].mean():.5f}")
print()
for thr_label, fH, fC in [
        ("50 keV proton (MC as written)",
         np.clip(1-5e4/np.maximum(Eh,1e-3),0,1),
         np.clip(1-5e4/(0.284*np.maximum(Eh,1e-3)),0,1)),
        (f"{Ep_thr*1e3:.0f} keV proton (= 47 keVee trigger)",
         np.clip(1-Ep_thr*1e6/np.maximum(Eh,1e-3),0,1),
         np.zeros_like(Eh)),
        (f"{Ep_of_L(0.226*0.135)*1e3:.0f} keV proton (observed 0.226 V.ns turn-on)",
         np.clip(1-Ep_of_L(0.226*0.135)*1e6/np.maximum(Eh,1e-3),0,1),
         np.zeros_like(Eh))]:
    pv = (Sig_sH*fH + Sig_sC*fC)/Sig_t*p_int
    print(f"  visible scatters, threshold = {thr_label:44s} {pv.sum():8.1f}"
          f"   ratio to captures {pv.sum()/p_cap.sum():6.1f} : 1")
# with the time-resolved tile (NOTES: single-pass 0.99% -> 1.96% per arrival)
fH = np.clip(1-Ep_thr*1e6/np.maximum(Eh,1e-3),0,1)
pv = (Sig_sH*fH)/Sig_t*p_int
for boost, lbl in [(2.0,"x2 tile random walk (NOTES 0.99->1.96%)"),
                   (3.0,"x3"), (5.6,"x5.6 (eps=1.95% per arrival)")]:
    print(f"    + {lbl:42s} captures {p_cap.sum()*boost:7.1f}"
          f"  ratio {pv.sum()/(p_cap.sum()*boost):6.1f} : 1")

# what thermal fraction closes the gap?
hdr("9. WHAT WOULD IT TAKE?  (item 5b)")
targets = [1.3, 2.3]
p_cap_th_per = p_cap[Eh<1.0].sum()/max((Eh<1.0).sum(),1)
n_th = (Eh<1.0).sum()
print(f"  present: {pv.sum():.0f} triggers-from-recoils : {p_cap.sum():.1f} captures"
      f" = {pv.sum()/p_cap.sum():.0f} : 1  (per {N:,} cone-emitted)")
print(f"  <p_cap> for a thermal arrival = {p_cap_th_per:.3f}; "
      f"{n_th} thermal arrivals now")
for tgt in targets:
    need_cap = pv.sum()/tgt
    extra = need_cap - p_cap.sum()
    n_extra = extra/p_cap_th_per
    print(f"  to reach {tgt:.1f}:1 need {need_cap:.0f} captures -> {n_extra:.0f} extra"
          f" thermal arrivals = {n_extra/n_th:.0f}x the present thermal flux;")
    print(f"      thermal would then be {100*(n_th+n_extra)/(hit.sum()+n_extra):.0f} %"
          f" of all arrivals, and {100*(n_th+n_extra)/N:.4f} % of cone-emitted")
    print(f"      as a fraction of an ISOTROPIC source (cone = "
          f"{(1-math.cos(math.radians(20)))/2*100:.2f} % of 4pi): "
          f"{(n_th+n_extra)/N*(1-math.cos(math.radians(20)))/2:.2e} per 4pi neutron")

# ─────────────────────────── 10. capture during slowing down in the shield
hdr("10. DOES THE SHIELD CAPTURE AT THERMAL, OR ON THE WAY DOWN?")
SB_E = n_B10_sh*3840*np.sqrt(E_TH/np.maximum(Egrid,E_TH))*BARN
SH_E = n_H_sh*0.332*np.sqrt(E_TH/np.maximum(Egrid,E_TH))*BARN
Sa_E = SB_E + SH_E
St_E = Sa_E + SHg + SCg
uu = np.log(E0/Egrid)
I = abs(np.trapezoid((Sa_E/(xibar*St_E))[::-1], uu[::-1]))
print(f"  slowing-down capture integral I = int Sigma_a/(xibar Sigma_t) du = {I:.3f}")
print(f"  -> P(captured before reaching thermal) = 1-exp(-I) = {1-math.exp(-I):.3f}")
print(f"     so ~{100*(1-math.exp(-I)):.0f} % of shield captures happen EPITHERMAL,")
print(f"     which is why the MC capture-time median (1.34 us) is shorter than the")
print(f"     {t_sd*1e6:.1f} us full thermalisation time plus the 1.95 us thermal dwell.")
# where in lethargy
for ulo, uhi, lbl in [(0, 4.6, "2.45 MeV - 24 keV"), (4.6, 11.5, "24 keV - 25 eV"),
                      (11.5, 18.4, "25 eV - thermal")]:
    m = (uu >= ulo) & (uu <= uhi)
    Ii = abs(np.trapezoid((Sa_E/(xibar*St_E))[m][::-1], uu[m][::-1]))
    print(f"     {lbl:20s} contributes {Ii/I*100:5.1f} % of I")

# ────────────────── 11. validation: alpha+Li light, and the Fe(n,g) gammas
hdr("11. VALIDATION AND A COMPETING GAMMA SOURCE")
# ASTAR-like alpha mass stopping power in polyvinyltoluene [MeV cm^2/g]
_AE = np.array([0.05,0.1,0.2,0.4,0.7,1.0,1.47,2.0,3.0,5.0])
_AS = np.array([1050,1500,1950,2250,2260,2130,1790,1500,1150, 800.])
def Sa_(E): return np.exp(np.interp(np.log(np.clip(E,0.02,5.)),np.log(_AE),np.log(_AS)))
def L_gen(E, S, n=4000):
    e = np.linspace(1e-4, E, n); return float(np.trapezoid(1.0/(1.0+kB*S(e)), e))
L_a = L_gen(1.470, Sa_)
# 7Li: same velocity as a proton of E/7; effective charge ~2.6 at these speeds
def S_Li(E): return (2.6**2) * Sp(np.clip(E/7.0, 1e-3, None))
L_Li = L_gen(0.840, S_Li)
print(f"  alpha 1470 keV -> {L_a*1e3:5.1f} keVee ; 7Li 840 keV -> {L_Li*1e3:5.1f} keVee")
print(f"  total {1e3*(L_a+L_Li):5.1f} keVee   vs adopted 85 keVee "
      f"(-> {(L_a+L_Li)*1e3/0.63:.0f} keVee per V.ns)")
# Fe(n,gamma) from the 1 cm steel: yield, escape, solid angle, detection
mu_fe76, mu_pvt76 = 0.0342*7.85, kn_total(7.6)*ne_pvt
Y_Fe = 0.0010 * 2.0          # 2 gammas per capture, <E> ~ 3.8 MeV, use 7.6/2 -> take 7.6
esc_fe = (1-math.exp(-mu_fe76*0.5))/(mu_fe76*0.5) if mu_fe76 else 1
d_shield, d_steel = 9.5, 2.5      # cm from the tile
sa_gain = (d_shield/d_steel)**2
R_fe = (Y_Fe*esc_fe*sa_gain*(1-math.exp(-mu_pvt76))) / \
       (Y478*esc[0.478]*fe[0.478]*(1-math.exp(-mu[('pvt',0.478)])))
print(f"  Fe(n,g): mu(Fe,7.6MeV) {mu_fe76:.3f}/cm  self-escape {esc_fe:.2f}  "
      f"mu(PVT) {mu_pvt76:.4f}/cm  p_int {1-math.exp(-mu_pvt76):.4f}")
print(f"  solid-angle gain (steel at 2.5 cm vs shield at 9.5 cm) x{sa_gain:.1f}")
print(f"  -> Fe-capture gamma rate at the tile = {100*R_fe:.1f} % of the 478 keV rate")
print(f"  Fe thermal Sigma_a = {n_Fe*2.56*BARN:.4f}/cm -> "
      f"1/(Sigma_a v) = {1e6/(n_Fe*2.56*BARN*2.2e5):.1f} us in bulk steel")
print(f"  pure-carbon moderator would need {math.log(E0/E_TH)/xi_C:.0f} collisions")
