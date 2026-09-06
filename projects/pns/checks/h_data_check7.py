"""Is the empty first ~0.39 us of the gate a fixed offset or a dead time?
Compare runs with different gate widths."""
import numpy as np, glob, os
from pnsana.extract import Feat
for p in sorted(glob.glob("/Users/virgolaema/Software/3det/pns-waveform-ana/cache/feat_v3_*.npz")):
    try:
        f = Feat(p)
    except Exception as e:
        continue
    tr = f.tau_rise_us[np.isfinite(f.tau_rise_us)]
    if tr.size < 200: continue
    gw = np.nanmedian(f.gate_w_us[np.isfinite(f.gate_w_us)])
    ing = tr[(tr>=0)&(tr<=gw)]
    print(f"{os.path.basename(p)[8:-4][:56]:58s} gate {gw:6.2f} us  "
          f"n_tau {tr.size:6d}  min(tau_rise) {tr.min():+7.3f}  "
          f"p1 {np.percentile(tr,1):+7.3f}  in-gate {ing.size:6d}")
