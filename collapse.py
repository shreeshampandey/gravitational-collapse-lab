"""Spherical Einstein--Klein-Gordon collapse in polar-areal gauge (G=c=1, 8*pi*G convention: G_ab=8 pi T_ab).
ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega^2 ;  Phi=phi_r, Pi=(a/alpha) phi_t.
Cell-centred grid r_i=(i+1/2)dr, parity-extended ghost cells at the origin (Phi odd, Pi even)."""
import numpy as np, numba as nb

@nb.njit(cache=True)
def constraints(r, Phi, Pi, dr, R):
    """mass function m(r) from  m' = 2 pi r^2 S - 4 pi r S m  (S=Phi^2+Pi^2), then a and alpha."""
    n = r.size; S = Phi*Phi + Pi*Pi
    m = np.empty(n); a = np.empty(n); al = np.empty(n)
    # integrate m' = A - B m with midpoint-exponential step (second order); m(0)=0, first cell centre at dr/2
    mm = 0.0; rprev = 0.0; Sprev = S[0]
    for i in range(n):
        h = r[i] - rprev
        Am = 2*np.pi*(0.5*(r[i]+rprev))**2 * 0.5*(S[i]+Sprev)
        Bm = 4*np.pi*(0.5*(r[i]+rprev)) * 0.5*(S[i]+Sprev)
        e = np.exp(-Bm*h)
        mm = mm*e + (Am/Bm)*(1-e) if Bm*h > 1e-12 else mm + Am*h
        m[i] = mm; rprev = r[i]; Sprev = S[i]
    for i in range(n):
        x = 1.0 - 2*m[i]/r[i]
        a[i] = 1.0/np.sqrt(x) if x > 1e-12 else 1e6
    # ln(alpha/a)' = (a^2-1)/r ; boundary alpha(R)=1/a(R)
    lam = np.empty(n)   # ln(alpha/a)
    lam[n-1] = -2*np.log(a[n-1])
    for i in range(n-2, -1, -1):
        f1 = (a[i]**2-1)/r[i]; f2 = (a[i+1]**2-1)/r[i+1]
        lam[i] = lam[i+1] - 0.5*(f1+f2)*(r[i+1]-r[i])
    for i in range(n):
        al[i] = a[i]*np.exp(lam[i])
    return m, a, al

@nb.njit(cache=True)
def rhs(r, Phi, Pi, dr, R, sponge, eps_diss, par=-1.0):
    n = r.size
    m, a, al = constraints(r, Phi, Pi, dr, R)
    # ghost-extended arrays (2 ghost cells each side)
    g = 2
    P = np.empty(n+2*g); Q = np.empty(n+2*g); A = np.empty(n+2*g); L = np.empty(n+2*g); rr = np.empty(n+2*g)
    for i in range(n):
        P[g+i]=Phi[i]; Q[g+i]=Pi[i]; A[g+i]=a[i]; L[g+i]=al[i]; rr[g+i]=r[i]
    for k in range(g):      # origin: Phi odd, Pi even, a,alpha even
        P[g-1-k] = par*Phi[k]; Q[g-1-k] = Pi[k]; A[g-1-k]=a[k]; L[g-1-k]=al[k]; rr[g-1-k] = -r[k]
    for k in range(g):      # outer: linear extrapolation
        for arr in (P,Q,A,L):
            arr[g+n+k] = 2*arr[g+n-1+k] - arr[g+n-2+k]
        rr[g+n+k] = r[n-1] + (k+1)*dr
    F1 = L*Q/A            # alpha Pi / a
    F2 = rr*rr*L*P/A      # r^2 alpha Phi / a
    dPhi = np.empty(n); dPi = np.empty(n)
    for i in range(n):
        j = g+i
        dPhi[i] = (F1[j+1]-F1[j-1])/(2*dr)
        dPi[i] = (F2[j+1]-F2[j-1])/(2*dr)/(rr[j]*rr[j])
        # Kreiss-Oliger 4th-derivative dissipation
        dPhi[i] += -eps_diss/(16*dr)*(P[j+2]-4*P[j+1]+6*P[j]-4*P[j-1]+P[j-2])
        dPi[i]  += -eps_diss/(16*dr)*(Q[j+2]-4*Q[j+1]+6*Q[j]-4*Q[j-1]+Q[j-2])
        dPhi[i] -= sponge[i]*Phi[i]; dPi[i] -= sponge[i]*Pi[i]
    return dPhi, dPi

def evolve(Phi0, Pi0, R=20.0, dr=0.02, tmax=15.0, cfl=0.25, eps_diss=0.5, sponge_frac=0.2, sponge_rate=2.0,
           stop_bh=True, record_every=None, bh_thresh=0.97, tol_alpha=None, par=-1.0):
    n = int(round(R/dr)); r = (np.arange(n)+0.5)*dr
    Phi = Phi0(r).copy(); Pi = Pi0(r).copy()
    sponge = np.where(r > R*(1-sponge_frac), sponge_rate*((r-R*(1-sponge_frac))/(R*sponge_frac))**2, 0.0)
    dt = cfl*dr; nsteps = int(np.ceil(tmax/dt)); t = 0.0
    rec = []; bh = False; t_bh = None; mbh=None
    for s in range(nsteps):
        if record_every and s % record_every == 0:
            m, a, al = constraints(r, Phi, Pi, dr, R); rec.append((t, al[0], np.max(2*m/r), m[-1], Phi[0], m[min(int(8.0/dr), m.size-1)]))
        k1 = rhs(r, Phi, Pi, dr, R, sponge, eps_diss, par)
        k2 = rhs(r, Phi+0.5*dt*k1[0], Pi+0.5*dt*k1[1], dr, R, sponge, eps_diss, par)
        k3 = rhs(r, Phi+0.5*dt*k2[0], Pi+0.5*dt*k2[1], dr, R, sponge, eps_diss, par)
        k4 = rhs(r, Phi+dt*k3[0], Pi+dt*k3[1], dr, R, sponge, eps_diss, par)
        Phi = Phi + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); Pi = Pi + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); t += dt
        if stop_bh and (s % 4 == 0):
            m, a, al = constraints(r, Phi, Pi, dr, R)
            comp = 2*m/r
            if np.max(comp) > bh_thresh:
                bh = True; t_bh = t; i = np.argmax(comp); mbh = m[i]; break
    m, a, al = constraints(r, Phi, Pi, dr, R)
    return dict(r=r, Phi=Phi, Pi=Pi, m=m, a=a, al=al, t=t, bh=bh, t_bh=t_bh, mbh=mbh, rec=np.array(rec), dr=dr)

def gaussian_shell(p, r0=6.0, w=1.0):
    """initially ingoing Gaussian pulse: phi = p exp(-((r-r0)/w)^2);  Pi = -(Phi) approximately ingoing (a~1)"""
    phi = lambda r: p*np.exp(-((r-r0)/w)**2)
    Phi0 = lambda r: -2*(r-r0)/w**2*phi(r)
    Pi0 = lambda r: Phi0(r)       # ingoing: phi_t = +phi_r  ->  Pi=+Phi
    return Phi0, Pi0
