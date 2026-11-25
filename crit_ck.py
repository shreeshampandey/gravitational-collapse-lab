import sys,time,os; sys.path.insert(0,'.')
import numpy as np, json
from crit import *
dr=float(sys.argv[1]); mode=sys.argv[2]; budget=float(sys.argv[3]) if len(sys.argv)>3 else 240
fn='ck_dr%g.json'%dr; t0=time.time()
st=json.load(open(fn)) if os.path.exists(fn) else dict(lo=0.015,hi=0.025,it=0,rows=[])
if mode=='bisect':
    while st['it']<34 and time.time()-t0<budget:
        pm=0.5*(st['lo']+st['hi'])
        if run(pm,dr)['bh']: st['hi']=pm
        else: st['lo']=pm
        st['it']+=1; json.dump(st,open(fn,'w'))
    print("dr=%g it=%d lo=%.14f hi-lo=%.2e"%(dr,st['it'],st['lo'],st['hi']-st['lo']))
else:
    ps=st['lo']; done={r['delta'] for r in st['rows']}
    for d in np.logspace(-0.6,-2.0,15):
        if float(d) in done: continue
        if time.time()-t0>budget: break
        o=run(ps*(1+d),dr)
        st['rows'].append(dict(delta=float(d),bh=bool(o['bh']),mbh=(None if o['mbh'] is None else float(o['mbh'])),t_bh=o['t_bh']))
        json.dump(st,open(fn,'w'))
    print("rows done",len(st['rows']),"of 15")
