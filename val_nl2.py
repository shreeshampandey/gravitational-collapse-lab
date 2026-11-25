import sys,time; sys.path.insert(0,'.')
import numpy as np, json
from collapse import *
out=json.load(open('val_nl.json'))
p=0.012; Phi0,Pi0=gaussian_shell(p); mom=[]
for dr in (0.04,0.02,0.01):
    dt=0.25*dr; k=4; T0=7.0; n0=int(round(T0/dt))
    st=lambda ns: evolve(Phi0,Pi0,R=16,dr=dr,tmax=ns*dt-1e-12,sponge_frac=0.25,stop_bh=False)
    om,oc,op=st(n0-k),st(n0),st(n0+k)
    adot=(op['a']-om['a'])/(2*k*dt); rhs_=4*np.pi*oc['r']*oc['al']*oc['Phi']*oc['Pi']; m=oc['r']<10
    res=np.sqrt(np.mean((adot[m]-rhs_[m])**2))/np.sqrt(np.mean(rhs_[m]**2)); mom.append((dr,res))
    print("dr=%.3f rel. momentum-constraint residual %.3e"%(dr,res),flush=True)
mom=np.array(mom); print("orders",np.log2(mom[:-1,1]/mom[1:,1])); out['mom']=mom.tolist()
# Birkhoff / mass conservation with bigger domain
dr=0.01; times=[0.0,6.0,8.0,10.0,14.0,18.0]; Mt=[]; snaps={}
for T in times:
    o=evolve(Phi0,Pi0,R=40,dr=dr,tmax=max(T,1e-9),sponge_frac=0.25,stop_bh=False)
    r=o['r']; m=o['m']; al=o['al']; S=o['Phi']**2+o['Pi']**2
    i=np.argmin(abs(r-28)); M=m[i]; Mt.append((T,float(M)))
    vac=(r>r[np.max(np.where(S>1e-14)[0])]+0.1)&(r<29)
    dev=np.max(abs(al[vac]**2-(1-2*M/r[vac]))) if vac.sum()>10 else np.nan
    print("t=%.1f M(r=28)=%.6f  vac pts=%d  max|alpha^2-(1-2M/r)|=%.2e"%(T,M,vac.sum(),dev),flush=True)
    snaps[str(T)]=dict(r=r[::20].tolist(),m=m[::20].tolist(),al=al[::20].tolist(),dev=float(dev))
out['Mt']=Mt; out['snaps']=snaps; json.dump(out,open('val_nl.json','w'))
