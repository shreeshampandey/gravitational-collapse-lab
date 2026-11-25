import sys; sys.path.insert(0,'.')
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from collapse import *

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.25,
    'lines.linewidth': 1.3, 'axes.titlesize': 9
})

# ------------- Fig 1: dispersal vs collapse ---------------
cases = [
    (0.012, 'disperses ($p=0.012$)', 'C0', '-'),
    (0.0188, '$p=0.0188<p^*$', 'C2', '--'),
    (0.0190, '$p=0.0190>p^*$', 'C1', '-.'),
    (0.03, 'collapses ($p=0.03$)', 'C3', ':')
]
fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.6))
for p, lab, c, ls in cases:
    o = evolve(*gaussian_shell(p), R=16, dr=0.01, tmax=20, sponge_frac=0.25, bh_thresh=0.9, record_every=20)
    rec = o['rec']
    ax[0].plot(rec[:, 0], rec[:, 1], color=c, ls=ls, label=lab)
    ax[1].plot(rec[:, 0], rec[:, 2], color=c, ls=ls)
    if o['bh']:
        ax[0].plot(o['t'], rec[-1, 1], '*', color=c, ms=8)
        ax[1].plot(o['t'], rec[-1, 2], '*', color=c, ms=8)
ax[0].set_yscale('log'); ax[0].set_ylim(3e-3, 400)
ax[0].set_xlabel('time $t$ (units with $G=c=1$)'); ax[0].set_ylabel('central lapse $\\alpha(t,0)$')
ax[0].legend(loc='upper right', ncol=1, fontsize=6.3, framealpha=0.9)
ax[0].set_title('(a) time dilation at the centre')
ax[1].axhline(0.9, color='k', lw=0.7); ax[1].text(0.3, 0.92, 'threshold 0.9', fontsize=7)
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel('time $t$ (units with $G=c=1$)')
ax[1].set_ylabel('max compactness $2m/r$'); ax[1].set_title('(b) approach to a horizon')
for T, ls in ((0.0, '-'), (6.0, '--'), (8.0, '-.')):
    o = evolve(*gaussian_shell(0.03), R=16, dr=0.01, tmax=max(T, 1e-9), sponge_frac=0.25, stop_bh=False)
    ax[2].plot(o['r'], o['m'], ls, label='$t=%.0f$' % T)
ax[2].set_xlim(0, 14); ax[2].set_xlabel('areal radius $r$'); ax[2].set_ylabel('enclosed mass $m(r)$')
ax[2].legend(loc='center right'); ax[2].set_title('(c) mass function, $p=0.03$')
plt.tight_layout(); plt.savefig('figs/fig1_collapse.pdf'); plt.close()

# ------------- Fig 2: verification convergence (revised non-overlapping x-ticks) ------------
vf = json.load(open('val_flat.json'))
vn = json.load(open('val_nl.json'))
mom = np.array(vn['mom'])
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
ax[0].loglog(vf['dr'], vf['err'], 'o-', label='measured')
d = np.array(vf['dr'])
ax[0].loglog(d, vf['err'][0] * (d / d[0])**2, 'k--', label='slope 2')
ax[0].set_xlabel(r'grid spacing $\Delta r$'); ax[0].set_ylabel(r'relative $L_2$ error of $\Phi$')
ax[0].set_title('(a) flat space vs exact solution'); ax[0].legend(loc='lower right')
ax[1].loglog(mom[:, 0], mom[:, 1], 's-', color='C3', label='measured')
ax[1].loglog(mom[:, 0], mom[0, 1] * (mom[:, 0] / mom[0, 0])**2, 'k--', label='slope 2')
ax[1].set_xlabel(r'grid spacing $\Delta r$'); ax[1].set_ylabel('momentum-constraint residual')
ax[1].set_title('(b) strong-field consistency check'); ax[1].legend(loc='lower right')
for a, tk in ((ax[0], [0.01, 0.02, 0.04, 0.08]), (ax[1], [0.01, 0.02, 0.04])):
    a.xaxis.set_major_locator(FixedLocator(tk))
    a.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in tk]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xlim(tk[0] * 0.88, tk[-1] * 1.14)
plt.tight_layout(); plt.savefig('figs/fig2_convergence.pdf'); plt.close()

# ------------- Fig 3: Birkhoff ------------------------
sn = vn['snaps']
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
for T, ls in (('0.0', '-'), ('6.0', '--'), ('10.0', '-.'), ('14.0', ':'), ('18.0', (0, (5, 1)))):
    ax[0].plot(sn[T]['r'], sn[T]['m'], ls=ls, label='$t=%s$' % T.rstrip('0').rstrip('.'))
ax[0].set_xlim(0, 34); ax[0].set_xlabel('areal radius $r$'); ax[0].set_ylabel('enclosed mass $m(r)$')
ax[0].legend(loc='center right'); ax[0].set_title('(a) plateau = Birkhoff exterior')
Ts = [float(k) for k in sn]; dev = [sn[k]['dev'] for k in sn]
ax[1].semilogy(Ts, dev, 'o-'); ax[1].set_xlabel('time $t$')
ax[1].set_ylabel(r'$\max|\alpha^2-(1-2M/r)|$ in vacuum')
ax[1].set_title('(b) exterior lapse vs Schwarzschild')
plt.tight_layout(); plt.savefig('figs/fig3_birkhoff.pdf'); plt.close()

# ------------- Fig 4: critical scaling with OLS fit uncertainties (revised) ------------
def fit_powerlaw(x, y):
    X = np.log(x); Y = np.log(y); n = len(X)
    A = np.vstack([X, np.ones(n)]).T
    c = np.linalg.lstsq(A, Y, rcond=None)[0]
    s2 = ((Y - A @ c)**2).sum() / (n - 2)
    se = np.sqrt(s2 / ((X - X.mean())**2).sum())
    return c[0], c[1], se

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={'width_ratios': [1.15, 1.0]})
sym = {'0.02': ('^', 'C2'), '0.01': ('s', 'C1'), '0.005': ('o', 'C0')}
G = {}
for dr in ('0.02', '0.01', '0.005'):
    rows = [r for r in json.load(open(f'ck_dr{dr}.json'))['rows'] if r['bh']]
    x = np.array([r['delta'] for r in rows])
    y = np.array([r['mbh'] for r in rows])
    ax[0].loglog(x, y, sym[dr][0], color=sym[dr][1], ms=4, label=rf'$\Delta r={dr}$')
    G[dr] = {}
    for lo in (0.01, 0.02):
        m = x >= lo - 1e-9
        G[dr][lo] = fit_powerlaw(x[m], y[m])
    if dr == '0.005':
        g, c0, se = G[dr][0.01]
        xx = np.logspace(-2, np.log10(0.26), 40)
        ax[0].loglog(xx, np.exp(c0) * xx**g, '-', color='C0', lw=1.0, alpha=0.85, label=r'fit, $\Delta r=0.005$ ($\pm1\sigma$ band)')
        xm = np.exp(np.log(x).mean()); ym = np.exp(np.log(y).mean())
        ax[0].fill_between(xx, ym * (xx / xm)**(g - se), ym * (xx / xm)**(g + se),
                           color='C0', alpha=0.18, lw=0)

xx = np.logspace(-2, np.log10(0.30), 10)
ax[0].loglog(xx, 0.258 * (xx / 0.251)**0.374, 'k--', lw=1.0, label=r'published $\gamma=0.374$')
ax[0].set_xlabel(r'$(p-p^*)/p^*$'); ax[0].set_ylabel(r'black-hole mass $M_{\rm BH}$')
ax[0].legend(loc='upper left', fontsize=6.6, framealpha=0.9)
ax[0].set_title(r'(a) mass scaling above threshold')

for k, dr in enumerate(('0.02', '0.01', '0.005')):
    for j, lo in enumerate((0.01, 0.02)):
        g, _, se = G[dr][lo]
        dx = -0.10 if j == 0 else 0.10
        ax[1].errorbar(k + dx, g, yerr=se, fmt=('o' if j == 0 else 's'),
                       color=sym[dr][1], mfc=(sym[dr][1] if j == 0 else 'w'),
                       capsize=3.0, ms=4.5)

ax[1].axhline(0.374, color='k', ls='--', lw=1.0, label=r'published $\gamma=0.374$')
ax[1].plot([], [], 'ko', ms=4.5, label=r'fit from $0.01$')
ax[1].plot([], [], 'ks', mfc='w', ms=4.5, label=r'fit from $0.02$')
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(['0.02', '0.01', '0.005'])
ax[1].set_xlim(-0.5, 2.5); ax[1].set_ylim(0.23, 0.43)
ax[1].set_xlabel(r'grid spacing $\Delta r$ (coarser $\rightarrow$ finer)'); ax[1].set_ylabel(r'fitted exponent $\gamma$ ($\pm1\sigma$)')
ax[1].set_title(r'(b) fitted exponent with $1\sigma$ uncertainties')
ax[1].legend(loc='lower right', fontsize=7.0)
plt.tight_layout(); plt.savefig('figs/fig4_critical.pdf'); plt.close()
print("figs 1-4 done")
