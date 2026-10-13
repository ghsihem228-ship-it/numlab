# Contributing

Use Python 3.12 and follow the installation commands in README.md.
Run the backend tests and Ruff before proposing changes. New solvers should
include tests against analytical solutions and their expected convergence rates,
as well as validation and work-budget failure tests.

Keep changes focused on one module and use conventional commit messages such
as `feat: add ODE solvers` or `test: verify RK4 convergence`. Include the problem,
implementation, validation commands, and known limitations in pull requests.

Do not commit credentials or local virtual environments. Untrusted mathematical
expressions must never be passed to Python evaluation functions.
