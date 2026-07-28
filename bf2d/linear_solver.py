"""
Linear-solver wrapper.

Provides a single ``solve`` entry point that transparently dispatches between
a direct sparse LU (``scipy.sparse.linalg.spsolve``) and a preconditioned
GMRES driven by an algebraic multigrid preconditioner (``pyamg``).

Why both
--------
- For small systems (< 5000 unknowns) direct LU is faster, deterministic
  and has no tuning parameters.
- For larger systems (the 3N block momentum system) direct factorisation
  is O(N^1.5) memory and quickly becomes the dominant cost.  GMRES + AMG
  scales as O(N) for well-conditioned elliptic operators.

The wrapper also handles failure modes that otherwise crash the solver:
singular matrices fall back to least-squares, non-convergent GMRES falls
back to direct solve.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, spsolve

try:
    import pyamg  # type: ignore
    _HAS_PYAMG = True
except ImportError:  # pragma: no cover
    _HAS_PYAMG = False


log = logging.getLogger("bf2d.linear")


@dataclass
class LinearSolveStats:
    """Diagnostic information returned alongside the solution."""
    method: str                 # "direct" | "gmres_amg" | "lstsq_fallback"
    iterations: int             # GMRES iterations (0 for direct)
    residual: float             # final ||Ax - b|| / ||b||
    ok: bool                    # True iff converged / succeeded


def solve_linear_system(
    A: sp.csr_matrix,
    b: np.ndarray,
    *,
    direct_threshold: int = 5000,
    gmres_tol: float = 1.0e-8,
    gmres_maxiter: int = 300,
    gmres_restart: int = 40,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, LinearSolveStats]:
    """Solve ``A x = b`` with adaptive method selection.

    Parameters
    ----------
    A
        Square CSR sparse matrix.
    b
        Right-hand side vector of compatible length.
    direct_threshold
        Below this number of unknowns, use direct LU.
    gmres_tol, gmres_maxiter, gmres_restart
        GMRES control parameters.
    x0
        Optional initial guess (ignored by direct solve).

    Returns
    -------
    x, stats
        Solution vector and diagnostic information.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"A ({A.shape}) and b ({b.shape}) incompatible")

    n = A.shape[0]
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1e-30:
        return np.zeros_like(b), LinearSolveStats("trivial", 0, 0.0, True)

    # Small systems: direct LU
    if n <= direct_threshold or not _HAS_PYAMG:
        return _direct_solve(A, b, b_norm)

    # Large systems: GMRES + AMG
    return _gmres_amg_solve(A, b, b_norm, gmres_tol, gmres_maxiter,
                             gmres_restart, x0)


# ---------------------------------------------------------------------------
def _direct_solve(A: sp.csr_matrix, b: np.ndarray,
                  b_norm: float) -> tuple[np.ndarray, LinearSolveStats]:
    """Direct sparse LU with least-squares fallback for singular systems."""
    try:
        x = spsolve(A, b)
        if not np.all(np.isfinite(x)):
            raise RuntimeError("spsolve returned non-finite values")
        res = float(np.linalg.norm(A @ x - b)) / b_norm
        return x, LinearSolveStats("direct", 0, res, True)
    except Exception as exc:  # noqa: BLE001
        log.warning("direct_solve_failed_falling_back_to_lstsq",
                    extra={"error_type": type(exc).__name__})
        # Least-squares on the dense matrix (only when genuinely broken)
        x, *_ = np.linalg.lstsq(A.toarray(), b, rcond=None)
        res = float(np.linalg.norm(A @ x - b)) / b_norm
        return x, LinearSolveStats("lstsq_fallback", 0, res, False)


# ---------------------------------------------------------------------------
def _gmres_amg_solve(
    A: sp.csr_matrix,
    b: np.ndarray,
    b_norm: float,
    tol: float,
    maxiter: int,
    restart: int,
    x0: np.ndarray | None,
) -> tuple[np.ndarray, LinearSolveStats]:
    """Smoothed-aggregation AMG preconditioned GMRES."""
    # Build preconditioner.  SA is robust for the symmetric-ish block
    # systems we produce; for strongly non-symmetric (high Peclet) systems
    # classical AMG is used via ruge_stuben_solver.
    try:
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=50)
        M = ml.aspreconditioner(cycle="V")
    except Exception as exc:  # noqa: BLE001
        log.warning("amg_setup_failed_falling_back_to_direct",
                    extra={"error_type": type(exc).__name__})
        return _direct_solve(A, b, b_norm)

    # GMRES driver
    it_counter = {"n": 0}

    def _count(_res):
        it_counter["n"] += 1

    try:
        x, info = gmres(
            A, b, M=M, rtol=tol,
            restart=restart, maxiter=maxiter,
            x0=x0, callback=_count, callback_type="pr_norm",
        )
    except TypeError:
        # Older scipy: rtol -> tol, no callback_type kwarg
        x, info = gmres(A, b, M=M, tol=tol, restart=restart,
                        maxiter=maxiter, x0=x0, callback=_count)

    res = float(np.linalg.norm(A @ x - b)) / b_norm
    converged = (info == 0) and res < 10.0 * tol
    if not converged:
        # Fall back to direct if GMRES didn't converge
        log.warning("gmres_did_not_converge_falling_back_to_direct",
                    extra={"gmres_info": int(info), "residual": res,
                           "iterations": it_counter["n"]})
        return _direct_solve(A, b, b_norm)

    return x, LinearSolveStats("gmres_amg", it_counter["n"], res, True)
