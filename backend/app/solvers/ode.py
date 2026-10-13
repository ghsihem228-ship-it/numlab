"""Scalar and vector initial-value ODE solvers with bounded work.

Right-hand sides are Python callables supplied by trusted application code.
Parsing visitor expressions belongs to the API layer; this module never evaluates strings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import RK45

Method = Literal["euler", "heun", "rk4", "rk45"]
RHS = Callable[[float, NDArray[np.float64]], ArrayLike]
MAX_STEPS = 100_000
MAX_DIMENSION = 32


@dataclass(frozen=True)
class ODEResult:
    """Samples, elapsed solver time, and work counters.

    ``y`` always has shape (number of samples, number of state variables).
    RK45 samples follow its accepted adaptive steps, rather than a uniform grid.
    """

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    method: Method
    runtime_ms: float
    nfev: int
    steps: int


def solve_ode(
    rhs: RHS,
    y0: ArrayLike,
    t_span: tuple[float, float],
    *,
    method: Method = "rk4",
    step: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 10_000,
    timeout_s: float = 2.0,
) -> ODEResult:
    """Solve y'=rhs(t,y) forward in time.

    Fixed methods shorten their final step to hit the requested endpoint.
    For RK45, ``step`` is the maximum step size and tolerances control accuracy.
    The deadline is cooperative: it cannot interrupt a blocking RHS callback.
    Invalid inputs, nonfinite states, exhausted budgets, and failed integration
    raise ValueError, RuntimeError, or TimeoutError instead of returning partial results.
    """
    if method not in {"euler", "heun", "rk4", "rk45"}:
        raise ValueError("Unknown ODE method")
    t0, t1 = map(float, t_span)
    if not np.isfinite([t0, t1, step, rtol, atol, timeout_s]).all():
        raise ValueError("Times, step, tolerances, and timeout must be finite")
    if t1 <= t0 or step <= 0 or rtol <= 0 or atol <= 0 or timeout_s <= 0:
        raise ValueError("Require t1 > t0 and positive step, tolerances, and timeout")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise ValueError("max_steps must be an integer")
    if not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
    state = np.atleast_1d(np.asarray(y0, dtype=float)).copy()
    if state.ndim != 1 or not 1 <= state.size <= MAX_DIMENSION:
        raise ValueError(f"y0 must be a scalar or vector of at most {MAX_DIMENSION} values")
    if not np.isfinite(state).all():
        raise ValueError("y0 must be finite")
    duration = t1 - t0
    ratio = duration / step
    if not np.isfinite(duration) or not np.isfinite(ratio) or ratio > max_steps:
        raise ValueError("Requested maximum step size exceeds the step budget")
    if t0 + min(step, duration) == t0:
        raise ValueError("Step is too small to advance time")
    started = perf_counter()
    nfev = 0

    def check_deadline() -> None:
        if perf_counter() - started > timeout_s:
            raise TimeoutError("ODE solve exceeded its time budget")

    def evaluate(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal nfev
        check_deadline()
        # Also bound rejected adaptive steps, whose RHS calls do not produce samples.
        if nfev >= 20 * max_steps + 2:
            raise RuntimeError("ODE solve exceeded its function evaluation budget")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = np.atleast_1d(np.asarray(rhs(t, y.copy()), dtype=float))
        nfev += 1
        check_deadline()
        if value.shape != state.shape or not np.isfinite(value).all():
            raise ValueError("RHS must return a finite derivative matching y0's shape")
        return value

    times = [t0]
    states = [state.copy()]
    if method == "rk45":
        integrator = RK45(evaluate, t0, state, t1, max_step=step, rtol=rtol, atol=atol)
        while integrator.status == "running":
            if len(times) - 1 >= max_steps:
                raise RuntimeError("RK45 exceeded its accepted-step budget")
            message = integrator.step()
            if integrator.status == "failed":
                raise RuntimeError(f"RK45 failed: {message}")
            check_deadline()
            times.append(float(integrator.t))
            states.append(integrator.y.copy())
    else:
        count = ceil(ratio)
        for index in range(count):
            t = times[-1]
            next_t = t1 if index == count - 1 else min(t0 + (index + 1) * step, t1)
            h = next_t - t
            if h <= 0:
                raise ValueError("Step is too small to advance time")
            with np.errstate(over="ignore", invalid="ignore"):
                k1 = evaluate(t, state)
                if method == "euler":
                    updated = state + h * k1
                elif method == "heun":
                    k2 = evaluate(next_t, state + h * k1)
                    updated = state + h * (k1 + k2) / 2
                else:
                    k2 = evaluate(t + h / 2, state + h * k1 / 2)
                    k3 = evaluate(t + h / 2, state + h * k2 / 2)
                    k4 = evaluate(next_t, state + h * k3)
                    updated = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not np.isfinite(updated).all():
                raise ValueError("Integration produced a nonfinite state")
            state = updated
            times.append(next_t)
            states.append(state.copy())
    check_deadline()
    return ODEResult(
        t=np.asarray(times),
        y=np.asarray(states),
        method=method,
        runtime_ms=(perf_counter() - started) * 1000,
        nfev=nfev,
        steps=len(times) - 1,
    )
