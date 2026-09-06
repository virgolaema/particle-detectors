"""Ambient-subtracted delayed spectrum above vs below the 478 keV Compton edge."""
import numpy as np
from pnsana.extract import Feat, cache_path
from pnsana import alphareport as AR
C = "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"
f = Feat(cache_path("20260814_sample02_gate1us_period50us_finalConfig_merged", C))
b = Feat(cache_path("20260817_sample02_background", C))
sc = AR.ambient_scale(f, b)
print("ambient_scale =", sc, " n_bkg =", int(b.ok.sum()))
EDGE = AR.EDGE_478_VNS
print("EDGE_478_VNS =", round(EDGE,3), " keVee/Vns =", round(AR.TILE_KEVEE_PER_VNS,1))
dm = AR.delayed_mask(f)
bm = AR.base_mask(b)
for lbl, sel in [("below edge", lambda m,x: m & (x < EDGE)),
                 ("above edge", lambda m,x: m & (x >= EDGE))]:
    n = int(sel(dm, f.charge).sum()); nb = int(sel(bm, b.charge).sum())
    exp = sc*nb
    print(f"  {lbl:11s} beam {n:7d}   ambient expected {exp:9.1f}   "
          f"NET {n-exp:+9.1f} +- {np.sqrt(n+ (sc**2)*nb):.1f}")

print("\nmeta beam:", {k: f.meta[k] for k in f.meta if k in
      ("clip_b_mv","rate_hz","dt_ns","thr_mv","lsb_mv","n","run")})
print("meta bkg :", {k: b.meta[k] for k in b.meta if k in
      ("clip_b_mv","rate_hz","dt_ns","thr_mv","lsb_mv","n","run")})
print("beam meta keys:", sorted(f.meta))
import numpy as np
print("beam  amp_mv p99.9 %.1f  max %.1f  charge p99 %.2f" %
      (np.nanpercentile(f.amp_mv,99.9), np.nanmax(f.amp_mv), np.nanpercentile(f.charge,99)))
print("bkg   amp_mv p99.9 %.1f  max %.1f  charge p99 %.2f" %
      (np.nanpercentile(b.amp_mv,99.9), np.nanmax(b.amp_mv), np.nanpercentile(b.charge,99)))
print("bkg clip frac %.3f  beam clip frac %.3f" % (b.clip.mean(), f.clip.mean()))
