"""nr_lab.py -- student version (pure NumPy): spherical Einstein--scalar-field collapse.
Units G=c=1.  Metric  ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega^2  (r = areal radius).
Fields: Phi = d(phi)/dr,  Pi = (a/alpha) d(phi)/dt.   Grid: cell centres r_i = (i+1/2) dr."""
import numpy as np

def solve_constraints(r, Phi, Pi):
    """Step 1: enclosed mass m(r) ('Gauss law'), then a(r) and the lapse alpha(r)."""
    S  = Phi**2 + Pi**2                                   # field energy ~ S/(2 a^2)
    rp = np.r_[0.0, r[:-1]]; h = r - rp                   # cell edges
    rm = 0.5*(r + rp);  Sm = 0.5*(S + np.r_[S[0], S[:-1]])
    A, B = 2*np.pi*rm**2*Sm, 4*np.pi*rm*Sm                # m' = A - B m   (A: source, B: gravity's self-coupling)
    e = np.exp(-B*h); c = np.where(B*h > 1e-12, A/np.where(B > 0, B, 1)*(1 - e), A*h)
    P = np.cumprod(e); m = P*np.cumsum(c/P)               # exact solution of the linear ODE cell by cell
    a = 1/np.sqrt(1 - 2*m/r)                              # 2m/r < 1 outside a black hole
    f = (a**2 - 1)/r                                      # d/dr ln(alpha/a) = (a^2-1)/r   ("redshift potential")
    lam = np.empty_like(r); lam[-1] = -2*np.log(a[-1])    # alpha(R) = 1/a(R): t = proper time at infinity
    lam[:-1] = lam[-1] - np.cumsum((0.5*(f[:-1] + f[1:])*np.diff(r))[::-1])[::-1]
    return m, a, a*np.exp(lam)

def rhs(r, Phi, Pi, dr, sponge, eps=0.5):
    """Step 2: wave equation on the curved background (flat space: alpha=a=1)."""
    m, a, al = solve_constraints(r, Phi, Pi); g = 2
    ext = lambda f, s: np.r_[s*f[:g][::-1], f, 2*f[-1]-f[-2], 3*f[-1]-2*f[-2]]   # parity at r=0, linear extrapolation outside
    P, Q, A, L = ext(Phi, -1.), ext(Pi, 1.), ext(a, 1.), ext(al, 1.)
    rr = np.r_[-r[:g][::-1], r, r[-1] + dr, r[-1] + 2*dr]; n = r.size
    F1, F2 = L*Q/A, rr**2*L*P/A                            # alpha*Pi/a  and  r^2 alpha Phi/a
    ctr = slice(g, g + n); up = slice(g + 1, g + n + 1); dn = slice(g - 1, g + n - 1)
    dPhi = (F1[up] - F1[dn])/(2*dr)                        # Phi_t = (alpha Pi/a)'
    dPi  = (F2[up] - F2[dn])/(2*dr)/rr[ctr]**2             # Pi_t  = r^-2 (r^2 alpha Phi/a)'
    k4 = lambda X: X[g+2:g+n+2] - 4*X[g+1:g+n+1] + 6*X[ctr] - 4*X[g-1:g+n-1] + X[g-2:g+n-2]
    return dPhi - eps/(16*dr)*k4(P) - sponge*Phi, dPi - eps/(16*dr)*k4(Q) - sponge*Pi

def evolve(phi0, R=16., dr=0.02, tmax=20., cfl=0.25, eps=0.5, stop_bh=True, thresh=0.9):
    """RK4 time stepping. phi0(r): initial profile of an ingoing pulse. Returns a dict of the final state."""
    n = int(round(R/dr)); r = (np.arange(n) + 0.5)*dr
    h = 1e-6; Phi = (phi0(r + h) - phi0(r - h))/(2*h); Pi = Phi.copy()           # ingoing: phi_t = +phi_r
    x = np.clip((r - 0.75*R)/(0.25*R), 0, None); sponge = 2.0*x**2                 # absorbing layer near r=R
    dt = cfl*dr; t = 0.; bh = False; mbh = None
    F = lambda U, V: rhs(r, U, V, dr, sponge, eps)
    for _ in range(int(np.ceil(tmax/dt))):
        k1 = F(Phi, Pi); k2 = F(Phi + .5*dt*k1[0], Pi + .5*dt*k1[1])
        k3 = F(Phi + .5*dt*k2[0], Pi + .5*dt*k2[1]); k4 = F(Phi + dt*k3[0], Pi + dt*k3[1])
        Phi = Phi + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); Pi = Pi + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]); t += dt
        if stop_bh:
            m = solve_constraints(r, Phi, Pi)[0]; comp = 2*m/r
            if comp.max() > thresh: bh = True; mbh = m[comp.argmax()]; break      # apparent-horizon proxy
    m, a, al = solve_constraints(r, Phi, Pi)
    return dict(r=r, Phi=Phi, Pi=Pi, m=m, a=a, al=al, t=t, bh=bh, mbh=mbh)

def gaussian(p, r0=6., w=1.):
    return lambda r: p*np.exp(-((r - r0)/w)**2)

def bisect(make_profile, lo, hi, n=30, **kw):
    """Find the critical amplitude p* between 'disperses' (lo) and 'collapses' (hi)."""
    for _ in range(n):
        mid = 0.5*(lo + hi)
        if evolve(make_profile(mid), **kw)['bh']: hi = mid
        else: lo = mid
    return lo, hi


# ---------------------------------------------------------------- self-checks (verification before physics)
def flat_exact_Phi(r, t, p=1e-4, r0=6., w=1.):
    """Exact d(phi)/dr of the flat-space spherical wave whose data are phi=p exp(-((r-r0)/w)^2), phi_t = phi_r at t=0
    (d'Alembert for u = r*phi with odd extension)."""
    from scipy.integrate import quad
    phi0 = lambda s: p*np.exp(-((abs(s) - r0)/w)**2); dphi0 = lambda s: -2*(abs(s) - r0)/w**2*phi0(s)
    u0 = lambda s: s*phi0(s); du0 = lambda s: phi0(s) + abs(s)*dphi0(s); ut0 = lambda s: s*dphi0(s)
    G = np.vectorize(lambda s: quad(ut0, 0, s, limit=200, epsabs=1e-16, epsrel=1e-13)[0])
    u = 0.5*(u0(r + t) + u0(r - t)) + 0.5*(G(r + t) - G(r - t))
    ur = 0.5*(du0(r + t) + du0(r - t)) + 0.5*(ut0(r + t) - ut0(r - t))
    return ur/r - u/r**2

def selfcheck(verbose=True):
    """Three automatic checks; every stage of the lab must pass them before any physics is interpreted."""
    rep = []; ok = True
    # 1. flat space: second-order convergence against the exact solution
    errs = []
    for dr in (0.04, 0.02):
        o = evolve(gaussian(1e-4), R=20., dr=dr, tmax=10., stop_bh=False); msk = o['r'] < 11
        ex = flat_exact_Phi(o['r'][msk], o['t']); errs.append(np.linalg.norm(o['Phi'][msk] - ex)/np.linalg.norm(ex))
    order = np.log2(errs[0]/errs[1]); p1 = 1.8 < order < 2.2; ok &= p1
    rep.append(f"[{'PASS' if p1 else 'FAIL'}] flat-space error vs exact: {errs[0]:.2e} -> {errs[1]:.2e}, observed order {order:.2f} (expect 2)")
    # 2. Birkhoff: outside the matter the mass is constant and alpha^2 = 1 - 2M/r
    o = evolve(gaussian(0.012), R=30., dr=0.02, tmax=10., stop_bh=False)
    S = o['Phi']**2 + o['Pi']**2; edge = o['r'][np.max(np.where(S > 1e-14)[0])] + 0.1
    vac = (o['r'] > edge) & (o['r'] < 26); M = o['m'][vac][0]
    spread = (o['m'][vac].max() - o['m'][vac].min())/M; dev = np.max(np.abs(o['al'][vac]**2 - (1 - 2*M/o['r'][vac])))
    p2 = spread < 1e-8 and dev < 1e-6; ok &= p2
    rep.append(f"[{'PASS' if p2 else 'FAIL'}] Birkhoff (t=10): M={M:.6f}, spread of m(r) in vacuum {spread:.1e}, max|alpha^2-(1-2M/r)| {dev:.1e}")
    # 3. critical amplitude at coarse resolution lies where expected
    lo, hi = bisect(gaussian, 0.015, 0.025, n=18, dr=0.04); p3 = 0.0185 < lo < 0.0205; ok &= p3
    rep.append(f"[{'PASS' if p3 else 'FAIL'}] critical amplitude p* = {lo:.5f} (dr=0.04; reference window 0.0185-0.0205)")
    if verbose: print("\n".join(rep)); print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok, rep
