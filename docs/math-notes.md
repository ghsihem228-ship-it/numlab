# ODE methods

For y' = f(t, y), initial state y(t0) = y0, and step h:

- **Euler**: y_next = y + h f(t, y). Global error is O(h).
- **Heun**: predict with Euler, then average the initial and predicted slopes.
  Global error is O(h²).
- **RK4**: combine four slopes with weights 1, 2, 2, 1 divided by 6.
  Global error is O(h⁴).
- **RK45**: SciPy's Dormand–Prince embedded 5(4) method estimates local error
  and adjusts its step to meet relative and absolute tolerances. Accepted
  samples are nonuniform; the supplied step is a maximum.

Convergence tests halve h and measure the endpoint error against exp(-t).
The expected log₂(error(h) / error(h/2)) is 1, 2, and 4 respectively.
Tolerance controls for adaptive integration do not guarantee a global error bound.

The solver accepts scalar or vector problems and integrates forward in time.
It shortens the final fixed step to reach the endpoint. Step, dimension,
function evaluation, and cooperative time budgets limit computation.
