"""The in-gate window: is any of it lost?  tau_rise = t(pulse) - t(gate rise)."""
import numpy as np
from pnsana.extract import Feat, cache_path
C="/Users/virgolaema/Software/3det/pns-waveform-ana/cache"
f=Feat(cache_path("20260814_sample02_gate1us_period50us_finalConfig_merged",C))
hg=np.isfinite(f.tau_rise_us)&np.isfinite(f.tau_end_us)&~f.clip
tr=f.tau_rise_us[hg]
b=np.arange(-2.0,3.01,0.1)
h,_=np.histogram(tr,bins=b)
for lo,c in zip(b[:-1],h):
    print(f"  tau_rise {lo:+5.2f}..{lo+0.1:+5.2f}: {c:6d}")
print("gate width median", np.median(f.gate_w_us[hg]))
print("min tau_rise", tr.min(), " frac tau_rise<0:", (tr<0).mean())
