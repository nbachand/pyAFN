"""Flow and pressure calculation functions for AFN models.

This module contains functions for calculating flow rates, pressures, and
discharge coefficients in airflow network models.
"""

import numpy as np
from .constants import g, beta


def getWindBuoyantP(rho, flowParams):
    """Calculate wind and buoyancy-driven pressure.
    
    Args:
        rho: Air density (kg/m³)
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


def flowFromP(rho, C_d, A, delp):
    """Calculate flow rate from pressure difference.
    
    Args:
        rho: Air density (kg/m³)
        C_d: Discharge coefficient
        A: Opening area (m²)
        delp: Pressure difference (Pa)
        
    Returns:
        Flow rate (m³/s)
    """
    delp = np.array(delp)
    S = np.sign(delp)
    return S * C_d * A * np.sqrt(2 * abs(delp) / rho)


def pFromFlow(rho, C_d, A, q):
    """Calculate pressure difference from flow rate.
    
    Args:
        rho: Air density (kg/m³)
        C_d: Discharge coefficient
        A: Opening area (m²)
        q: Flow rate (m³/s)
        
    Returns:
        Pressure difference (Pa)
    """
    q = np.array(q)
    S = np.sign(q)
    return S * (q / (C_d * A))**2 * (rho / 2)


def CFromFlow(rho, q, A, delp):
    """Calculate discharge coefficient from flow and pressure measurements.
    
    Args:
        rho: Air density (kg/m³)
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


def flowField(p_0, rho, flowParams):
    """Calculate flow field for all openings in the network.
    
    Args:
        p_0: Indoor pressure(s)
        rho: Air density (kg/m³)
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
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(rho, flowParams)
    return flowFromP(rho, C_d, A, delP)


def getC(p_0, rho, flowParams):
    """Calculate discharge coefficients from measured flows.
    
    Args:
        p_0: Indoor pressure(s)
        rho: Air density (kg/m³)
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
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(rho, flowParams)
    return CFromFlow(rho, q, A, delP)
