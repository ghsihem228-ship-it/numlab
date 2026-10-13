"""Accuracy, convergence, and failure behavior against known initial-value problems."""

import time

import numpy as np
import pytest

from app.solvers import solve_ode


@pytest.mark.parametrize(
    "method,tolerance",
    [
        ("euler", 0.015),
        ("heun", 5e-5),
        ("rk4", 1e-9),
        ("rk45", 1e-9),
    ],
)
def test_exponential_growth(method, tolerance):
    result = solve_ode(lambda t, y: y, 1.0, (0, 1), method=method, step=0.01)
    assert np.max(np.abs(result.y[:, 0] - np.exp(result.t))) < tolerance
    assert result.t[0] == 0 and result.t[-1] == 1
    assert result.y.shape == (len(result.t), 1)
    assert result.steps == len(result.t) - 1
    assert result.nfev > 0 and result.runtime_ms >= 0


@pytest.mark.parametrize("method,order", [("euler", 1), ("heun", 2), ("rk4", 4)])
def test_expected_convergence_order(method, order):
    errors = []
    for step in (0.1, 0.05, 0.025):
        result = solve_ode(lambda t, y: -y, 1, (0, 1), method=method, step=step)
        errors.append(abs(result.y[-1, 0] - np.exp(-1)))
    observed = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.all(np.abs(observed - order) < 0.2)


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "rk45"])
def test_vector_harmonic_oscillator(method):
    result = solve_ode(
        lambda t, y: [y[1], -y[0]],
        [1, 0],
        (0, 2 * np.pi),
        method=method,
        step=0.01,
    )
    expected = np.column_stack([np.cos(result.t), -np.sin(result.t)])
    assert np.max(np.abs(result.y - expected)) < 0.04


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "rk45"])
def test_non_autonomous_rhs_and_exact_endpoint(method):
    result = solve_ode(lambda t, y: [2 * t], 0, (0, 1), method=method, step=0.3)
    assert result.t[-1] == 1
    assert np.all(np.diff(result.t) > 0)
    assert np.max(np.diff(result.t)) <= 0.3 + 1e-15
    if method != "euler":
        np.testing.assert_allclose(result.y[:, 0], result.t**2, atol=1e-12)


def test_adaptive_tolerances_improve_accuracy():
    loose = solve_ode(lambda t, y: y, 1, (0, 5), method="rk45", step=5, rtol=1e-3)
    tight = solve_ode(lambda t, y: y, 1, (0, 5), method="rk45", step=5, rtol=1e-9)
    assert tight.steps > loose.steps
    assert abs(tight.y[-1, 0] - np.exp(5)) < abs(loose.y[-1, 0] - np.exp(5)) / 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": 0},
        {"step": -1},
        {"step": np.nan},
        {"rtol": 0},
        {"atol": -1},
        {"method": "unknown"},
        {"max_steps": True},
        {"max_steps": 0},
        {"max_steps": 100_001},
        {"max_steps": 1.5},
        {"timeout_s": 0},
        {"step": 1e-9},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        solve_ode(lambda t, y: y, 1, (0, 1), **kwargs)


@pytest.mark.parametrize(
    "y0,span",
    [
        (np.nan, (0, 1)),
        ([], (0, 1)),
        ([[1]], (0, 1)),
        ([0] * 33, (0, 1)),
        (1, (1, 0)),
        (1, (0, 0)),
        (1, (0, np.inf)),
    ],
)
def test_invalid_initial_problem(y0, span):
    with pytest.raises(ValueError):
        solve_ode(lambda t, y: y, y0, span)


@pytest.mark.parametrize("rhs", [lambda t, y: [1, 2], lambda t, y: [np.inf]])
def test_invalid_derivative(rhs):
    with pytest.raises(ValueError, match="finite derivative"):
        solve_ode(rhs, 1, (0, 1))


def test_timeout():
    def slow_rhs(t, y):
        time.sleep(0.01)
        return y

    with pytest.raises(TimeoutError):
        solve_ode(slow_rhs, 1, (0, 1), timeout_s=0.001)


def test_adaptive_step_budget():
    with pytest.raises(RuntimeError, match="budget"):
        solve_ode(lambda t, y: y, 1, (0, 1), method="rk45", step=1, max_steps=1)


def test_initial_state_is_not_mutated():
    initial = np.array([1.0])
    solve_ode(lambda t, y: -y, initial, (0, 1))
    np.testing.assert_array_equal(initial, [1])
