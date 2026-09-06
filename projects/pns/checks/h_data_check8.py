"""The in-gate trigger count is ~2x incomplete: events in the first 0.39 us of
the beam gate have tau_end = NaN (gate not fully reconstructed) and are dropped
by the (tau_rise>0 & tau_end<0) definition."""
import numpy as np
from pnsana.extract import Feat, cache_path
C = "/Users/virgolaema/Software/3det/pns-waveform-ana/cache"
f = Feat(cache_path("20260814_sample02_gate1us_period50us_finalConfig_merged", C))
tr, te = f.tau_rise_us, f.tau_end_us
fin = np.isfinite(tr)
GW = float(np.median(f.gate_w_us[fin]))
kept = fin & np.isfinite(te) & (tr > 0) & (te < 0)
lost = fin & ~np.isfinite(te) & (tr >= 0) & (tr <= GW)
print(f"gate width {GW:.3f} us")
for lbl, m in [("kept  (tau_rise>0 & tau_end<0)", kept),
               ("lost  (tau_rise in gate, tau_end NaN)", lost)]:
    print(f"  {lbl:40s} all {int(m.sum()):6d}  unclipped {int((m&~f.clip).sum()):6d}"
          f"  tau_rise range {tr[m].min():.3f}-{tr[m].max():.3f}")
tot = int(kept.sum()) + int(lost.sum())
totu = int((kept & ~f.clip).sum()) + int((lost & ~f.clip).sum())
ng = int(np.isfinite(te).sum())
print(f"\n  true in-gate population  all {tot}   unclipped {totu}")
print(f"  quoted in-gate count 4369 is {100*4369/totu:.0f}% of the unclipped total")
print(f"  per beam pulse (60,356 gated records): quoted {4369/ng:.4f}"
      f"  ->  corrected {totu/ng:.4f}")
print(f"  vs ~3300 true tile captures -> {3300/ng:.4f}/pulse ;"
      f"  recoil:capture = {4369/3300:.2f} (quoted)  {totu/3300:.2f} (corrected)")
