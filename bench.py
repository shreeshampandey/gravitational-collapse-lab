import sys,time; sys.path.insert(0,'.')
import numpy as np, json
import collapse as C
def constraints_np(r,Phi,Pi):
    S=Phi**2+Pi**2
    rp=np.concatenate([[0.0],r[:-1]]); Sp=np.concatenate([[S[0]],S[:-1]]); h=r-rp
    rm=0.5*(r+rp); Sm=0.5*(S+Sp)
    Am=2*np.pi*rm**2*Sm; Bm=4*np.pi*rm*Sm
    e=np.exp(-Bm*h); small=Bm*h<=1e-12
    c=np.where(small,Am*h,(Am/np.where(small,1,Bm))*(1-e))
    P=np.cumprod(e); m=P*np.cumsum(c/P)
    x=1-2*m/r; a=np.where(x>1e-12,1/np.sqrt(np.where(x>1e-12,x,1)),1e6)
    f=(a**2-1)/r; seg=0.5*(f[:-1]+f[1:])*np.diff(r)
    lam=np.empty_like(r); lam[-1]=-2*np.log(a[-1]); lam[:-1]=lam[-1]-np.cumsum(seg[::-1])[::-1]
    return m,a,a*np.exp(lam)
def rhs_np(r,Phi,Pi,dr,sponge,eps):
    m,a,al=constraints_np(r,Phi,Pi); g=2; n=r.size
    ext=lambda f,sgn: np.concatenate([sgn*f[g-1::-1][:g][::-1] if False else sgn*f[:g][::-1], f, 2*f[-1]-f[-2]+0*f[-g:], ])
    def ex(f,sgn):
        left=sgn*f[:g][::-1]; right=np.array([2*f[-1]-f[-2], 3*f[-1]-2*f[-2]])
        return np.concatenate([left,f,right])
    P=ex(Phi,-1.0); Q=ex(Pi,1.0); A=ex(a,1.0); L=ex(al,1.0)
    rr=np.concatenate([-r[:g][::-1],r,r[-1]+dr*np.array([1.0,2.0])])
    F1=L*Q/A; F2=rr**2*L*P/A
    dPhi=(F1[g+1:g+n+1]-F1[g-1:g+n-1])/(2*dr); dPi=(F2[g+1:g+n+1]-F2[g-1:g+n-1])/(2*dr)/rr[g:g+n]**2
    d4=lambda X:(X[g+2:g+n+2]-4*X[g+1:g+n+1]+6*X[g:g+n]-4*X[g-1:g+n-1]+X[g-2:g+n-2])
    dPhi-=eps/(16*dr)*d4(P)+sponge*Phi; dPi-=eps/(16*dr)*d4(Q)+sponge*Pi
    return dPhi,dPi
n=1600; dr=0.01; r=(np.arange(n)+0.5)*dr; Phi0,Pi0=C.gaussian_shell(0.015); Phi=Phi0(r); Pi=Pi0(r); sp=np.zeros(n)
a1=C.rhs(r,Phi,Pi,dr,16.0,sp,0.5,-1.0); a2=rhs_np(r,Phi,Pi,dr,sp,0.5)
print("max |numba-numpy| rhs:",max(np.abs(a1[0]-a2[0]).max(),np.abs(a1[1]-a2[1]).max()),"  scale",np.abs(a1[0]).max())
def timeit(f,rep): 
    f(); t=time.perf_counter(); [f() for _ in range(rep)]; return (time.perf_counter()-t)/rep
tn=timeit(lambda:C.rhs(r,Phi,Pi,dr,16.0,sp,0.5,-1.0),400); tv=timeit(lambda:rhs_np(r,Phi,Pi,dr,sp,0.5),400)
tp=timeit(lambda:C.rhs.py_func(r,Phi,Pi,dr,16.0,sp,0.5,-1.0),3)
print("time per RHS call, n=%d:  numba %.1f us   numpy-vectorised %.1f us   pure-python loops %.1f ms"%(n,tn*1e6,tv*1e6,tp*1e3))
res=dict(n=n,numba_us=tn*1e6,numpy_us=tv*1e6,python_ms=tp*1e3)
json.dump(res,open('bench.json','w'))
# wall-clock of a complete sub-critical run (dr=0.01, t=20) with numba
t=time.perf_counter(); o=C.evolve(*C.gaussian_shell(0.012),R=16,dr=0.01,tmax=20,sponge_frac=0.25,stop_bh=False); w=time.perf_counter()-t
print("full run dr=0.01, t=0..20 (%d steps): %.2f s with numba"%(int(np.ceil(20/(0.25*0.01))),w)); res['run_dr0.01_s']=w
# estimated for numpy/python
steps=int(np.ceil(20/(0.25*0.01))); print("estimated: numpy %.0f s ; pure python %.0f min"%(steps*4*tv,steps*4*tp/60)); res['est_numpy_s']=steps*4*tv; res['est_python_min']=steps*4*tp/60
json.dump(res,open('bench.json','w'))
