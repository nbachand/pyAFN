"""Solver functions for AFN optimization problems.

This module contains functions for solving airflow network equations
using optimization techniques.
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from .constants import rho
from .flow import flowField, getWindBuoyantP, flowFromP


def qObjective(p_0, flowParams):
    """Objective function for airflow network optimization.
    
    Minimizes the sum of squared room flow imbalances (continuity residual).
    
    Args:
        p_0: Indoor pressure(s) to optimize
        flowParams: Dictionary containing flow parameters
        
    Returns:
        Sum of squared flow residuals
    """
    qs = flowField(p_0, flowParams)
    rooms = flowParams["rooms"]
    qRooms = np.matmul(rooms.T, qs)
    return np.sum(qRooms**2)

def checkTwoOpeningAnalyticalP0(flowParams):
    if flowParams["rooms"].shape[1] != 1 or flowParams["C_d"].shape[0] != 2:
        return None  # Not applicable

    A = flowParams["A"]
    C_d = flowParams["C_d"]
    P = getWindBuoyantP(flowParams)
    C1 = A[0]*C_d[0]
    C2 = A[1]*C_d[1]
    return (C1**2 * P[0] + C2**2 * P[1]) / (C1**2 + C2**2)


def findOptimalP0(flowParams, checkAnalytical=True):
    """Find optimal indoor pressures that satisfy continuity.
    
    Solves for the indoor pressure(s) that minimize mass flow imbalances
    in all rooms of the airflow network.
    
    Args:
        flowParams: Dictionary containing:
            - rooms: Room connectivity matrix
            - Other parameters for flow calculations
            
    Returns:
        Optimization result object from scipy.optimize.minimize
    """
    if checkAnalytical:
        analyticalP0 = checkTwoOpeningAnalyticalP0(flowParams)
        if analyticalP0 is not None:
            return OptimizeResult(
                x=np.array([analyticalP0]),
                success=True,
                message="Analytical solution found for 2-opening single-room case."
            )
    bounds = np.array([
        np.min(getWindBuoyantP(flowParams)),
        np.max(getWindBuoyantP(flowParams))
    ])
    x0 = np.mean(bounds)
    NRooms = flowParams["rooms"].shape[1]
    bounds = np.tile(bounds, (NRooms, 1))
    x0 = np.tile(x0, NRooms)
    return minimize(qObjective, x0=x0, bounds=bounds, args=(flowParams,))


def matchObjective(x, flowParams, weight):
    """Combined objective function for matching flows and regularizing C_d.
    
    Minimizes:
      1) Sum of squared errors between predicted and target opening flows
      2) Plus weight × variance(C_d) for discharge coefficient uniformity
    
    Args:
        x: Decision vector [p0_1, ..., p0_N, Cd_1, ..., Cd_M]
        flowParams: Dictionary containing flow parameters including:
            - rooms: Room connectivity matrix (M × N)
            - A: Opening areas
            - q: Target flow rates
        weight: Regularization weight for C_d variance penalty
        
    Returns:
        Combined objective value (flow error + regularization)
    """
    rooms = flowParams["rooms"]
    N = rooms.shape[1]    # Number of rooms
    M = rooms.shape[0]    # Number of openings

    # Unpack decision vector
    p_0 = x[:N]
    Cd = x[N:]

    # Compute driving pressures
    params = flowParams.copy()
    params["C_d"] = Cd
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(flowParams)

    # 1) Predicted opening flows
    qs_pred = flowFromP(Cd, params["A"], delP)

    # 2) Flow-matching error (per opening)
    q_target = params["q"]
    f1 = np.sum((qs_pred - q_target)**2)

    # 3) Uniform-C penalty (variance)
    meanCd = np.mean(Cd)
    f2 = np.sum((Cd - meanCd)**2)

    return f1 + weight * f2


def findOptimalP0AndC(flowParams, weight=1e-1, disp=False):
    """Find optimal indoor pressures and discharge coefficients.
    
    Jointly optimizes room pressures (p0) and discharge coefficients (Cd)
    to match target flow measurements while encouraging uniform Cd values.
    
    Args:
        flowParams: Dictionary containing:
            - rooms: Room connectivity matrix (M × N)
            - A: Opening areas
            - q: Target/measured flow rates
            - C_d: Initial discharge coefficients
            - Other parameters for flow calculations
        weight: Regularization weight for C_d uniformity (default: 0.1)
        disp: Whether to display optimization progress (default: False)
        
    Returns:
        Optimization result with .x = [p0_1, ..., p0_N, Cd_1, ..., Cd_M]
    """
    rooms = flowParams["rooms"]
    N = rooms.shape[1]  # Number of rooms
    M = rooms.shape[0]  # Number of openings

    # Bounds for p0: between min/max wind-buoyancy pressures
    WBP = getWindBuoyantP(flowParams)
    p_bounds = [(np.min(WBP), np.max(WBP))] * N

    # Bounds for Cd: reasonable physical range
    C_bounds = [(1e-3, 5.0)] * M
    bounds = p_bounds + C_bounds

    # Initial guess: mid-range p0, mean(C_d)
    x0 = np.concatenate([
        np.full(N, np.mean(WBP)),
        np.full(M, np.mean(flowParams["C_d"]))
    ])

    res = minimize(
        matchObjective,
        x0=x0,
        args=(flowParams, weight),
        bounds=bounds,
        method="L-BFGS-B",
        options={"disp": disp}
    )
    return res

