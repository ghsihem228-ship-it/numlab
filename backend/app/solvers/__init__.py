"""Public numerical solver interface."""

from .ode import ODEResult, solve_ode

__all__ = ["ODEResult", "solve_ode"]
