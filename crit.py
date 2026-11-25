import sys,time; sys.path.insert(0,'.')
import numpy as np, json
from collapse import *
def run(p,dr,R=16.0,tmax=30.0,thresh=0.9):
    Phi0,Pi0=gaussian_shell(p)
    return evolve(Phi0,Pi0,R=R,dr=dr,tmax=tmax,sponge_frac=0.25,bh_thresh=thresh)
def bisect(dr,plo=0.005,phi=0.02,nit=40,**kw):
    assert not run(plo,dr,**kw)['bh'] and run(phi,dr,**kw)['bh']
    for k in range(nit):
        pm=0.5*(plo+phi)
        if run(pm,dr,**kw)['bh']: phi=pm
        else: plo=pm
    return plo,phi
if __name__=="__main__":
    for dr in (0.04,0.02):
        t0=time.time(); lo,hi=bisect(dr,nit=30); print("dr=%.3f  p*=%.12f  (%.0fs)"%(dr,0.5*(lo+hi),time.time()-t0))
