"""Is the 'in-gate' sample really prompt recoils, or the ambient floor?"""
import numpy as np
from pnsana.extract import Feat, cache_path
RUN = "20260814_sample02_gate1us_period50us_finalConfig_merged"
f = Feat(cache_path(RUN, "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"))
hg = np.isfinite(f.tau_end_us) & np.isfinite(f.tau_rise_us)
unc = ~f.clip
te = f.tau_end_us
print("gate widths:", np.unique(np.round(f.gate_w_us[hg],2))[:10])
# tau_end distribution (beam-referred, after gate end)
bins = np.array([-1,0,.5,1,2,3,5,8,12,16,20,25,30,35,40,45])
h,_ = np.histogram(te[hg & unc], bins=bins)
print("tau_end hist (unclipped), counts and per-us:")
for a,b,c in zip(bins[:-1],bins[1:],h):
    print(f"  {a:6.1f}..{b:6.1f} us : {c:7d}   {c/(b-a):8.1f}/us")
late = hg & unc & (te>20) & (te<45)
rate_flat = late.sum()/25.0
print(f"\nflat floor from 20-45us: {rate_flat:.1f} /us")
ing = hg & unc & (f.tau_rise_us>0) & (te<0)
w = np.median(f.gate_w_us[ing])
print(f"in-gate: {ing.sum()} in gate width {w:.2f} us -> {ing.sum()/w:.1f}/us ;"
      f" excess over flat = {ing.sum()-rate_flat*w:.0f}")
dly = hg & unc & (te>0.5) & (te<15)
print(f"delayed 0.5-15us: {dly.sum()} ; excess over flat = {dly.sum()-rate_flat*14.5:.0f}")

# charge-spectrum shape comparison
edges = np.array([0.2,.35,.5,.75,1.,1.5,2.,2.31,3.,5.,10.,50.])
for lbl,m in [("in-gate",ing),("delayed0.5-15",dly),("late20-45",late)]:
    h,_ = np.histogram(f.charge[m], bins=edges)
    print(f"{lbl:15s} N={m.sum():6d}  frac:", " ".join(f"{x:.3f}" for x in h/h.sum()))
# median psd
for lbl,m in [("in-gate",ing),("delayed0.5-15",dly),("late20-45",late)]:
    print(f"{lbl:15s} median charge {np.median(f.charge[m]):.2f} psd {np.median(f.psd[m]):.4f}")
# tau_rise structure inside the gate
tr = f.tau_rise_us[ing]
hb = np.linspace(0,1.2,13)
h,_ = np.histogram(tr,bins=hb)
print("tau_rise inside gate:", h)
