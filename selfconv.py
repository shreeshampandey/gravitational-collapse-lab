import sys; sys.path.insert(0,'.')
import numpy as np, json
from collapse import *
def avg2(u,k=1):
    for _ in range(k): u=0.5*(u[0::2]+u[1::2])
    return u
out={}
for p,T in ((0.012,10.0),(0.018,9.0)):
    res={}
    for dr in (0.04,0.02,0.01):
        o=evolve(*gaussian_shell(p),R=16,dr=dr,tmax=T,sponge_frac=0.25,stop_bh=False); res[dr]=o
    r=res[0.04]['r']; m=r<10
    row={}
    for f in ('Phi','Pi','al'):
        u1=res[0.04][f]; u2=avg2(res[0.02][f]); u4=avg2(res[0.01][f],2)
        n=min(u1.size,u2.size,u4.size); mm=m[:n]
        d1=np.linalg.norm((u1-u2)[:n][mm]); d2=np.linalg.norm((u2-u4)[:n][mm]); row[f]=d1/d2
        print("p=%.3f t=%.1f field %-3s Q = |u_h-u_h/2|/|u_h/2-u_h/4| = %.3f (expect 4 for 2nd order)"%(p,T,f,d1/d2))
    out["%.3f"%p]=dict(T=T,**row)
json.dump(out,open('selfconv.json','w'))
