# NumLab

An interactive numerical methods visualizer for learning ODEs, Fourier series,
and the heat equation through solution, error, and convergence plots.

## Current module

The ODE library is implemented: Euler, Heun, RK4, and adaptive RK45, including
scalar/vector states, work limits, timing, and analytical/convergence tests.
The web app, expression parser, and REST API are planned next. No browser demo
or GIF is available until the UI is implemented.

## Quick start

Requires Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-dev.lock
.venv/bin/python -m pip install --no-deps -e './backend[dev]'
cd backend
../.venv/bin/python -m pytest
../.venv/bin/ruff check .
```

Run a problem from the `backend` directory:

```python
from app.solvers import solve_ode

result = solve_ode(lambda t, y: -y, y0=1.0, t_span=(0.0, 1.0), method="rk4", step=0.01)
print(result.t[-1], result.y[-1, 0], result.runtime_ms)
```

Results contain `t`, `y` (samples × state variables), `method`, `runtime_ms`,
`nfev`, and `steps`. RK45 uses adaptive sample times. Analytical error is
computed by the caller when a reference solution exists; it is not fabricated
for arbitrary problems. Callables must be trusted; visitor expression strings
will use a restricted parser in the API module. No environment variables are
required for this module; see `.env.example`.

## Mathematics

Euler, Heun, and RK4 have global convergence orders 1, 2, and 4. RK45 uses
an embedded error estimate to adapt its steps. See [math notes](docs/math-notes.md)
for formulas, error interpretation, and limits.

## Roadmap

1. ODE solver library and analytical tests (implemented).
2. Validated REST API with a safe mathematical expression parser.
3. React/TypeScript ODE page, solution and convergence plots, comparison.
4. Fourier/Gibbs lab and explicit/Crank–Nicolson heat equation modules.
5. PNG/CSV exports, shareable links, Docker, CI, deployment, and GIF demos.

The REST API table will be documented when endpoints are implemented.
The intended deployment is FastAPI on Render and React/Vite on Vercel,
without a database or paid APIs; free-tier quotas and availability apply.

## License

See [LICENSE](LICENSE).
