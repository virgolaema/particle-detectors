"""Is there a beam-correlated delayed component ABOVE the 478 keV Compton edge?
Tests the <0.05% H-capture-gamma bound directly on the cache."""
import numpy as np
from pnsana.extract import Feat, cache_path
RUN = "20260814_sample02_gate1us_period50us_finalConfig_merged"
f = Feat(cache_path(RUN, "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"))
hg = np.isfinite(f.tau_end_us) & np.isfinite(f.tau_rise_us) & ~f.clip
te, q = f.tau_end_us, f.charge
EDGE = 2.31
bins = [(0.5,2),(2,4),(4,6),(6,9),(9,13),(13,20)]
print(" tau_end [us]      N<edge   N>edge   frac>edge")
for a,b in bins:
    m = hg & (te>a) & (te<b)
    lo = int((m & (q<EDGE)).sum()); hi = int((m & (q>=EDGE)).sum())
    print(f"  {a:5.1f}-{b:5.1f}   {lo:8d} {hi:8d}     {hi/(lo+hi):.4f}")
# fit each class to A exp(-t/lam) + C over 0.5-20 us with a common lam
from scipy.optimize import curve_fit
edges = np.arange(0.5, 20.01, 0.5); ctr = 0.5*(edges[:-1]+edges[1:])
def hist(m): return np.histogram(te[m], bins=edges)[0].astype(float)
hlo, hhi = hist(hg & (q<EDGE)), hist(hg & (q>=EDGE))
# record-window acceptance: fraction of records whose gate allows tau_end = t
acc = np.array([(hg & (te>t)).sum() for t in ctr], float); acc/=acc[0]
def mdl(t, A, lam, C): return A*np.exp(-t/lam) + C
for nm, h in [("below edge", hlo), ("above edge", hhi)]:
    try:
        p,cv = curve_fit(mdl, ctr, h/np.maximum(acc,1e-9),
                         p0=[h[0], 3.6, h[-1]], sigma=np.sqrt(np.maximum(h,1))/np.maximum(acc,1e-9),
                         maxfev=20000)
        e=np.sqrt(np.diag(cv))
        print(f"{nm}: A={p[0]:.0f}+-{e[0]:.0f}  lam={p[1]:.2f}+-{e[1]:.2f} us  C={p[2]:.0f}+-{e[2]:.0f}"
              f"   integral of decay 0.5-20us = {p[0]*p[1]*(np.exp(-0.5/p[1])-np.exp(-20/p[1]))/0.5:.0f} counts")
    except Exception as ex:
        print(nm, "fit failed", ex)
