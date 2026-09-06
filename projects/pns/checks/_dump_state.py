import json, os, sys, pickle, contextlib, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None   # neutralise any blocking show() in cells
sys.path.insert(0, os.path.abspath('../..'))
nb=json.load(open('pns.ipynb'))
g={'__name__':'__main__'}
REPLAY='1bc91a56'
with contextlib.redirect_stdout(io.StringIO()):
    for c in nb['cells']:
        if c['cell_type']!='code': continue
        src=''.join(c['source'])
        if src.strip(): exec(compile(src,f"<{c.get('id')}>",'exec'),g)
        if c.get('id')==REPLAY: break
out={}
for k in ['RES','SAMPLES','n_H_scint','n_C_scint','SCINT_THICKNESS_CM','E_THERMAL_EV']:
    if k in g: out[k]=g[k]
for k,v in g.items():
    if k.isupper() and isinstance(v,(int,float,str)): out.setdefault(k,v)
pickle.dump(out, open('_mc_state.pkl','wb'))
print("keys:",list(out.keys()))
r=list(out['RES'].values())[0]
print("RES sub-keys:",list(r.keys()))
