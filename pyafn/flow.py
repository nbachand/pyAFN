"""Flow and pressure calculation functions for AFN models.

This module contains functions for calculating flow rates, pressures, and
discharge coefficients in airflow network models.
"""

import numpy as np
from .constants import g, beta, rho


def createFlowParams(C_d=None, A=None, p_w=None, z=None, delT=None, q=None, rooms=None, hr=None):
    """Create a flowParams dictionary with automatic numpy array conversion.
    
    This helper function converts list inputs to numpy arrays and organizes them
    into the proper flowParams structure used by other pyAFN functions.
    
    Args:
        C_d: Discharge coefficients (list or array)
        A: Opening areas in m² (list or array)
        p_w: Wind pressure in Pa (list or array)
        z: Height of opening in m (list or array)
        delT: Temperature difference(s) in K (list or 2D array for multi-layer)
        q: Measured flow rates in m³/s (list or array)
        rooms: Room connectivity matrix (list or 2D array)
              Rows = openings, Columns = rooms
              Entry = 1 if opening connects room (inlet), 
                    = -1 if opening leaves room (outlet),
                    = 0 if opening doesn't connect to room
        hr: Reference/room height in m (scalar)
        
    Returns:
        flowParams: Dictionary with numpy array values ready for use
        
    Example:
        >>> flowParams = createFlowParams(
        ...     C_d=[1, 1, 1],
        ...     A=[1, 1, 2],
        ...     p_w=[1, 3, 0],
        ...     z=[3, 3, 3],
        ...     delT=[-3, 0, 2],
        ...     q=[1, 2, -3],
        ...     rooms=[[1], [1], [1]],
        ...     hr=3
        ... )
    """
    return {
        "C_d": np.array(C_d) if C_d is not None else None,
        "A": np.array(A) if A is not None else None,
        "p_w": np.array(p_w) if p_w is not None else None,
        "z": np.array(z) if z is not None else None,
        "delT": np.array(delT) if delT is not None else None,
        "q": np.array(q) if q is not None else None,
        "rooms": np.array(rooms) if rooms is not None else None,
        "hr": hr if hr is not None else None
    }


def getWindBuoyantP(flowParams):
    """Calculate wind and buoyancy-driven pressure.
    
    Args:
        flowParams: Dictionary containing:
            - p_w: Wind pressure
            - z: Height
            - delT: Temperature difference
            - hr: Reference height (m)
            
    Returns:
        Total pressure difference (wind + buoyancy)
    """
    p_w = flowParams["p_w"]
    z = flowParams["z"]
    delT = flowParams["delT"]
    hr = flowParams["hr"]
    
    if len(delT.shape) == 2:
        n_levels = delT.shape[1]
        delz = hr / n_levels
        z_levels = delz * (np.arange(n_levels) + 0.45)  # Vectorized center calculation
        z_below = (z_levels < z[:, None])
        delT = np.sum(delT * z_below, axis=1) / np.maximum(np.sum(z_below, axis=1), 1)
        sl_length = z - hr
        sls = sl_length > 0
        delT[sls] = hr / z[sls] * delT[sls] + sl_length[sls] / z[sls] * flowParams["delT"][sls, -1]
    
    delrho = -rho * beta * delT
    # delP is outdoor minus indoor, while p0/rho is indoor minus outdoor,
    # driving positive flow into the room (opposite textbook)
    return (delrho * g * z) + p_w


def flowFromP(C_d, A, delp):
    """Calculate flow rate from pressure difference.
    
    Args:
        C_d: Discharge coefficient
        A: Opening area (m²)
        delp: Pressure difference (Pa)
        
    Returns:
        Flow rate (m³/s)
    """
    delp = np.array(delp)
    S = np.sign(delp)
    return S * C_d * A * np.sqrt(2 * abs(delp) / rho)


def pFromFlow(C_d, A, q):
    """Calculate pressure difference from flow rate.
    
    Args:
        C_d: Discharge coefficient
        A: Opening area (m²)
        q: Flow rate (m³/s)
        
    Returns:
        Pressure difference (Pa)
    """
    q = np.array(q)
    S = np.sign(q)
    return S * (q / (C_d * A))**2 * (rho / 2)


def CFromFlow(q, A, delp):
    """Calculate discharge coefficient from flow and pressure measurements.
    
    Args:
        q: Flow rate (m³/s)
        A: Opening area (m²)
        delp: Pressure difference (Pa)
        
    Returns:
        Discharge coefficient (C_d)
    """
    delp = np.array(delp, dtype=float)
    # Prepare output filled with NaNs
    C = np.full_like(delp, np.nan, dtype=float)
    # Mask non-NaN, non-zero delp
    mask = ~np.isnan(delp) & (delp != 0)
    S = np.sign(delp[mask])
    C[mask] = q[mask] / (S * A[mask] * np.sqrt(2 * np.abs(delp[mask]) / rho))
    return C


def flowField(p_0, flowParams):
    """Calculate flow field for all openings in the network.
    
    Args:
        p_0: Indoor pressure(s)
        flowParams: Dictionary containing:
            - C_d: Discharge coefficients
            - A: Opening areas
            - rooms: Room connectivity matrix
            - Other parameters for getWindBuoyantP
            
    Returns:
        Flow rates for all openings
    """
    C_d = flowParams["C_d"]
    A = flowParams["A"]
    rooms = flowParams["rooms"]
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(flowParams)
    return flowFromP(C_d, A, delP)


def getC(p_0, flowParams):
    """Calculate discharge coefficients from measured flows.
    
    Args:
        p_0: Indoor pressure(s)
        flowParams: Dictionary containing:
            - A: Opening areas
            - q: Measured flow rates
            - rooms: Room connectivity matrix
            - Other parameters for getWindBuoyantP
            
    Returns:
        Calculated discharge coefficients
    """
    A = flowParams["A"]
    q = flowParams["q"]
    rooms = flowParams["rooms"]
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(flowParams)
    return CFromFlow(q, A, delP)
