"""Where does '4 +- 2 above the edge vs 13,987 below' come from?"""
import numpy as np
from pnsana.extract import Feat, cache_path
from pnsana import alphareport as AR
C = "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"
f = Feat(cache_path("20260814_sample02_gate1us_period50us_finalConfig_merged", C))
b = Feat(cache_path("20260817_sample02_background", C))
sc = AR.ambient_scale(f, b); E = AR.EDGE_478_VNS
d, mb = AR.delayed_mask(f), AR.base_mask(b)
def net(lo, hi):
    n = int((d & (f.charge>=lo) & (f.charge<hi)).sum())
    nb = int((mb & (b.charge>=lo) & (b.charge<hi)).sum())
    return n, sc*nb, n-sc*nb, np.sqrt(n + sc*sc*nb)
for lo, hi, lbl in [(0.0, E, "below edge (0 - 2.31)"),
                    (E, 3.0, "2.31 - 3.0  (the plotted range of fig_integral)"),
                    (E, 5.0, "2.31 - 5.0"),
                    (E, 1e9, "above edge, ALL")]:
    n, e_, s, err = net(lo, hi)
    print(f"  {lbl:44s} N {n:7d}  ambient {e_:9.1f}  NET {s:+9.1f} +- {err:.1f}")
print("\nper-0.1 V.ns bin around the edge (beam / ambient / net):")
edges = np.arange(2.0, 3.51, 0.1)
for lo, hi in zip(edges[:-1], edges[1:]):
    n, e_, s, err = net(lo, hi)
    print(f"   {lo:4.1f}-{hi:4.1f}  {n:6d} {e_:8.1f} {s:+8.1f} +- {err:.1f}")

print("\n--- alpha-tagged (PSD non-gamma) delayed sample ---")
e_, q90 = AR.q90_by_charge(b)
tag = AR.tag_alpha(f, e_, q90); btag = AR.tag_alpha(b, e_, q90)
def net2(lo, hi):
    n = int((d & tag & (f.charge>=lo) & (f.charge<hi)).sum())
    nb = int((mb & btag & (b.charge>=lo) & (b.charge<hi)).sum())
    return n, sc*nb, n-sc*nb, np.sqrt(n + sc*sc*nb)
for lo, hi, lbl in [(0.0, E, "below edge"), (E, 3.0, "2.31-3.0"),
                    (E, 1e9, "above edge ALL"), (0.44, 0.83, "alpha ROI")]:
    n, ee, s, err = net2(lo, hi)
    print(f"  {lbl:16s} N {n:7d}  ambient {ee:8.1f}  NET {s:+8.1f} +- {err:.1f}")
print("psd band validity: edges", e_[:3], "...", e_[-3:])
