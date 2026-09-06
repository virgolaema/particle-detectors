"""Data-side cross-check for the HYDROGEN study: how much of the in-gate
recoil-trigger population is removed by the C2 ceiling (clip) and by the
25 mV / 0.35 V.ns trigger turn-on.  Read-only; loads the feature cache."""
import numpy as np
from pnsana.extract import Feat, cache_path

RUN = "20260814_sample02_gate1us_period50us_finalConfig_merged"
f = Feat(cache_path(RUN, "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"))
print("n =", len(f.charge), " meta clip_b_mv =", f.meta.get("clip_b_mv"),
      " dt_ns =", f.meta.get("dt_ns"))
print("keys:", sorted(k for k in f._z))

has_gate = np.isfinite(f.tau_end_us) & np.isfinite(f.tau_rise_us)
print("records with a gate:", int(has_gate.sum()))
ing = has_gate & (f.tau_rise_us > 0) & (f.tau_end_us < 0)
print("in-gate (any clip):", int(ing.sum()),
      "  unclipped:", int((ing & ~f.clip).sum()),
      "  clipped:", int((ing & f.clip).sum()))
dly = has_gate & (f.tau_end_us > 0.5) & (f.tau_end_us < 15)
print("delayed 0.5-15us (any clip):", int(dly.sum()),
      "  unclipped:", int((dly & ~f.clip).sum()),
      "  clipped:", int((dly & f.clip).sum()))

q = f.charge
for lbl, m in [("in-gate", ing), ("delayed", dly)]:
    qq = q[m]
    print(f"\n{lbl}: charge V.ns  min {qq.min():.3f} med {np.median(qq):.3f} "
          f"p90 {np.percentile(qq,90):.3f} max {qq.max():.3f}")
    print("   amp_mv  min %.1f med %.1f p90 %.1f max %.1f"
          % (f.amp_mv[m].min(), np.median(f.amp_mv[m]),
             np.percentile(f.amp_mv[m],90), f.amp_mv[m].max()))
    edges = np.array([0,.2,.35,.5,.75,1.,1.5,2.,2.31,3.,5.,10.,50.])
    h,_ = np.histogram(qq, bins=edges)
    print("   Q hist:", " ".join(f"{a:g}-{b:g}:{c}" for a,b,c in zip(edges[:-1],edges[1:],h)))

# amplitude ceiling: where does amp_mv pile up?
a = f.amp_mv[np.isfinite(f.amp_mv)]
print("\namp_mv percentiles:", np.percentile(a,[50,90,99,99.9,100]).round(2))
print("frac clipped overall: %.4f" % f.clip.mean())
