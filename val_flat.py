import sys; sys.path.insert(0,'.')
import numpy as np, json
from scipy.integrate import quad
from collapse import *
p,r0,w=1e-4,6.0,1.0
phi0=lambda s: p*np.exp(-((abs(s)-r0)/w)**2)
dphi0=lambda s: -2*(abs(s)-r0)/w**2*phi0(s)            # d/dr for r>0 (even extension -> handled by |s|)
u0=lambda s: s*phi0(s)
du0=lambda s: phi0(s)+abs(s)*dphi0(s)
ut0=lambda s: s*dphi0(s)*1.0                              # s*phi_t0(|s|), phi_t0=phi0'(|s|)
def G(s): return quad(ut0,0,s,limit=200,epsabs=1e-16,epsrel=1e-13)[0]
def exact(r,t):
    u=0.5*(u0(r+t)+u0(r-t))+0.5*(np.array([G(x) for x in r+t])-np.array([G(x) for x in r-t]))
    ur=0.5*(du0(r+t)+du0(r-t))+0.5*(ut0(r+t)-ut0(r-t))
    return ur/r-u/r**2, u/r
T=10.0; out=[]
for dr in (0.08,0.04,0.02,0.01):
    Phi0=lambda r: dphi0(r); Pi0=Phi0
    o=evolve(Phi0,Pi0,R=20,dr=dr,tmax=T,sponge_frac=0.25,stop_bh=False)
    Pe,ph=exact(o['r'],o['t'])
    mask=o['r']<11
    err=np.sqrt(np.mean((o['Phi'][mask]-Pe[mask])**2))/np.sqrt(np.mean(Pe[mask]**2))
    out.append((dr,err,o['t'])); print("dr=%.3f  rel L2 error of Phi at t=%.2f: %.3e"%(dr,o['t'],err))
e=np.array(out); print("orders:",np.log2(e[:-1,1]/e[1:,1]))
json.dump(dict(dr=e[:,0].tolist(),err=e[:,1].tolist(),T=float(T)),open('val_flat.json','w'))
