"""Ventilation scaling functions for AFN calculations.

This module contains functions for calculating ventilation scaling factors
based on various Richardson numbers and pressure/velocity relationships.
"""

import numpy as np


def ventilationLowerScaling(Rh):
    """Calculate lower bound ventilation scaling.
    
    Args:
        Rh: Richardson number (horizontal)
        
    Returns:
        Lower bound scaling factor
    """
    return Rh / 2


def ventilationUpperScaling(Rq):
    """Calculate upper bound ventilation scaling with binomial approximation.
    
    Args:
        Rq: Richardson number (flux-based)
        
    Returns:
        Upper bound scaling factor using binomial approximation
    """
    scaling = np.zeros_like(Rq)
    mask = Rq > 1
    scaling[mask] = np.sqrt(1 - Rq[mask]**-2)
    # # Binomial approximation: 1 - Rq^-2 / 2 + Rq^-4 / 8
    # scaling[mask] = 1 - Rq[mask]**-2 / 2 + Rq[mask]**-4 / 8
    return scaling


def ventilationScalingSwitch(Rh, Rq, Rq_crit):
    """Switch between lower and upper scaling based on critical Richardson number.
    
    Args:
        Rh: Richardson number (horizontal)
        Rq: Richardson number (flux-based)
        Rq_crit: Critical Richardson number for switching
        
    Returns:
        Blended scaling factor
    """
    scaling = np.empty_like(Rq)
    mask = Rq < Rq_crit
    scaling[mask] = ventilationLowerScaling(Rh[mask])
    scaling[~mask] = ventilationUpperScaling(Rq[~mask])
    return scaling


def ventilationBlendedScaling_q(Rq):
    """Calculate blended ventilation scaling for velocity-based Richardson number.
    
    Args:
        Rq: Richardson number (flux-based)
        
    Returns:
        Blended scaling factor for velocity-based calculations
    """
    alpha = 1
    Rh_bound = Rq
    Rh_tangent = alpha * Rh_bound
    Rq_crit = np.sqrt(2)
    scaling = ventilationScalingSwitch(Rh_tangent, Rq, Rq_crit)
    return scaling


def ventilationBlendedScaling_p(Rq):
    """Calculate blended ventilation scaling for pressure-based Richardson number.
    
    Args:
        Rq: Richardson number (flux-based)
        
    Returns:
        Blended scaling factor for pressure-based calculations
    """
    # Scale factor to make the tangent line match the lower bound at the critical point
    alpha = 16 / (3 * np.sqrt(3))
    Rh_bound = np.sqrt(Rq / 2)
    Rh_tangent = np.sqrt(alpha) * Rh_bound
    Rq_crit = np.sqrt(3)
    scaling = ventilationScalingSwitch(Rh_tangent, Rq, Rq_crit)
    return scaling


def ventilationReDecomp_q(u_model, a, u_rms):
    """Recompose ventilation velocity with scaling.
    
    Args:
        u_model: Model velocity
        a: Scaling factor
        u_rms: RMS velocity
        
    Returns:
        Scaled ventilation velocity
    """
    u_model_scaled = u_model * a
    Rq = u_model_scaled / u_rms
    scaling = ventilationBlendedScaling_q(Rq)
    return u_model_scaled * scaling


def ventilationReDecomp_p(u_model, a, p_rms, A_param, rho_param):
    """Recompose ventilation pressure with scaling.
    
    Args:
        u_model: Model velocity
        a: Scaling factor
        p_rms: RMS pressure
        A_param: Opening area (m²)
        rho_param: Air density (kg/m³)
        
    Returns:
        Scaled ventilation pressure-equivalent velocity
    """
    k = A_param * np.sqrt(2 / rho_param)
    delP = u_model**2 / k  # Not u_model_scaled because we want the original delta P
    Rq = delP / p_rms
    scaling = ventilationBlendedScaling_p(Rq)
    return a * u_model * scaling
