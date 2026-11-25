import sys; sys.path.insert(0,'.')
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collapse import *
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'figure.dpi':150,'axes.grid':True,'grid.alpha':0.25,'lines.linewidth':1.3,'axes.titlesize':9})
Phi0,Pi0=gaussian_shell(0.012)
cfl=json.load(open('fail_cfl.json'))
fig,ax=plt.subplots(1,3,figsize=(7.3,2.6))
c=np.array(cfl); ax[0].semilogy(c[:,0],c[:,1],'o-'); ax[0].axvline(1.9,color='k',ls=':',lw=0.9); ax[0].text(1.03,0.03,'unstable\nbeyond\n$\\approx1.9$',fontsize=7)
ax[0].set_xlabel('time step / grid spacing ($\\Delta t/\\Delta r$)'); ax[0].set_ylabel('$\\max|\\Phi|$ at $t=20$'); ax[0].set_title('(a) time step too large')
# (b) wrong parity at the origin
for par,lab,col,ls in ((-1.0,'correct (odd)','C0','-'),(1.0,'wrong (even)','C3','--')):
    o=evolve(Phi0,Pi0,R=16,dr=0.02,tmax=10,sponge_frac=0.25,stop_bh=False,par=par,record_every=8)
    ax[1].plot(o['rec'][:,0],o['rec'][:,4]*1e3,color=col,ls=ls,label=lab)
ax[1].set_xlabel('time $t$'); ax[1].set_ylabel('$\\Phi$ in first cell ($\\times10^{-3}$)'); ax[1].set_title('(b) wrong symmetry at $r=0$'); ax[1].legend(loc='lower left'); ax[1].set_xlim(3,10)
# (c) outer boundary
for kw,lab,col,ls in ((dict(sponge_frac=0.25),'absorbing layer','C0','-'),(dict(sponge_frac=0.0,sponge_rate=0.0),'no layer','C3','--')):
    o=evolve(Phi0,Pi0,R=16,dr=0.02,tmax=40,stop_bh=False,record_every=40,**kw)
    ax[2].semilogy(o['rec'][:,0],np.maximum(o['rec'][:,5],1e-9),color=col,ls=ls,label=lab)
ax[2].set_xlabel('time $t$'); ax[2].set_ylabel('mass inside $r=8$'); ax[2].set_title('(c) reflecting outer edge'); ax[2].legend(loc='upper right'); ax[2].set_ylim(1e-7,1)
plt.tight_layout(); plt.savefig('figs/fig5_gallery.pdf'); plt.close(); print('ok')
