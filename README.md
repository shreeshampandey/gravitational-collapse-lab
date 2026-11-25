# Verification-first computational lab on gravitational collapse

Python code and data for a teaching lab in which students evolve a spherically symmetric
Einstein--massless-scalar-field spacetime: a scalar pulse that disperses or collapses to a black hole,
a test of Birkhoff's theorem, and a search for the critical threshold. Three automatic self-checks
must pass before any physics is interpreted.

## Files
| File | Purpose |
|---|---|
| `nr_lab.py` | Student module (NumPy only, about 100 lines), including `selfcheck()` |
| `student_lab.ipynb` | Blank student notebook (three sessions); keep `nr_lab.py` in the same folder |
| `collapse.py` | Faster Numba solver for higher-resolution runs |
| `val_flat.py`, `val_nl2.py`, `selfconv.py` | Verification: flat-space error, Birkhoff and momentum constraint, self-convergence |
| `crit.py`, `crit_ck.py` | Critical-amplitude bisection and the scaling data |
| `bench.py`, `fig5.py` | Timing benchmark and the bug-gallery experiments |
| `figs_nr.py` | Produces Figures 1-4 of the paper |
| `*.json` | Raw numerical results behind the tables and figures |
| `figs/` | The five figures (PDF) |
| `docs/student_notebook_rendered.pdf` | The notebook rendered as a PDF |

## Quick start
```
pip install -r requirements.txt
python -c "import nr_lab; nr_lab.selfcheck()"
```
Expected: three `[PASS]` lines and `ALL CHECKS PASSED` (about 25 s on one core).
Run all scripts from this folder. Create an empty `figs/` folder if it is missing.

## Requirements
Python 3.10 or later; NumPy, SciPy, Matplotlib, Jupyter; Numba (only for `collapse.py`).

## License
MIT, see `LICENSE`.
