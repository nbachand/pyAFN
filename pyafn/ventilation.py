"""Ventilation scaling functions for AFN calculations.

This module contains functions for calculating ventilation scaling factors
based on fluctuation intensity and harmonic intensity relationships.
"""

import numpy as np
from .constants import g, beta, rho, Cd


def ventilationLowerScaling(H):
    """Calculate fluctuation-dominated ventilation scaling.
    
    Args:
        H: Harmonic intensity
        
    Returns:
        Lower bound scaling factor
    """
    return 0.5 / H


def ventilationUpperScaling(I):
    """Calculate mean-dominated ventilation scaling.
    
    Args:
        I: Fluctuation intensity
        
    Returns:
        Upper bound scaling factor using binomial approximation
    """
    scaling = np.zeros_like(I, dtype=float)
    mask = I < 1
    scaling[mask] = np.sqrt(1 - I[mask] ** 2)
    return scaling


def ventilationScalingSwitch(H, I, I_crit):
    """Switch between fluctuation- and mean-dominated scaling.
    
    Args:
        H: Harmonic intensity
        I: Fluctuation intensity
        I_crit: Critical fluctuation intensity for switching
        
    Returns:
        Blended scaling factor
    """
    scaling = np.empty_like(I, dtype=float)
    mask = I >= I_crit
    scaling[mask] = ventilationLowerScaling(H[mask])
    scaling[~mask] = ventilationUpperScaling(I[~mask])
    return scaling


def ventilationBlendedScaling_q(I, I_crit=1 / np.sqrt(2)):
    """Calculate blended ventilation scaling for velocity-based fluctuation intensity.
    
    Args:
        I: Fluctuation intensity
        
    Returns:
        Blended scaling factor for velocity-based calculations
    """
    alpha = 1.0
    H_bound = I
    H_tangent = alpha * H_bound
    scaling = ventilationScalingSwitch(H_tangent, I, I_crit)
    return scaling


def ventilationBlendedScaling_p(I, I_crit =1 / np.sqrt(3)):
    """Calculate blended ventilation scaling for pressure-based fluctuation intensity.
    
    Args:
        I: Fluctuation intensity
        
    Returns:
        Blended scaling factor for pressure-based calculations
    """
    # Tangential-intersect model written in terms of harmonic intensity.
    alpha = 3 * np.sqrt(3) / 16
    H_bound = np.sqrt(2 * I)
    H_tangent = H_bound * np.sqrt(alpha)
    scaling = ventilationScalingSwitch(H_tangent, I, I_crit)
    return scaling

def getI_q(u_model_scaled, u_rms):
    mask = u_model_scaled != 0
    I = np.zeros_like(u_model_scaled)
    if np.isscalar(u_rms):
        I[mask] = u_rms / np.abs(u_model_scaled[mask])
    else:
        I[mask] = u_rms[mask] / np.abs(u_model_scaled[mask])
    return I

def getI_p(delP, p_rms):
    mask = delP != 0
    I = np.zeros_like(delP)
    if np.isscalar(p_rms):
        I[mask] = p_rms / (2 * np.abs(delP[mask]))
    else:
        I[mask] = p_rms[mask] / (2 * np.abs(delP[mask]))
    return I

def uModelToI_p(u_model, p_rms, A_param=1):
    k = A_param * Cd * np.sqrt(2 / rho)
    delP = (u_model/k)**2
    return getI_p(delP, p_rms)

def ventilationReDecomp_q(u_model, a, u_rms, I_crit=1 / np.sqrt(2)):
    """Recompose ventilation velocity with scaling.
    
    Args:
        u_model: Model velocity
        a: Scaling factor
        u_rms: RMS velocity
        
    Returns:
        Scaled ventilation velocity
    """
    I = getI_q(a * u_model, u_rms)
    scaling = ventilationBlendedScaling_q(I, I_crit=I_crit)
    return a * u_model * scaling


def ventilationReDecomp_p(u_model, a, p_rms, A_param=1, I_crit=1 / np.sqrt(3)):
    """Recompose ventilation pressure with scaling.
    
    Args:
        u_model: Model velocity
        a: Scaling factor
        p_rms: RMS pressure
        A_param: Opening area (m²)
        
    Returns:
        Scaled ventilation pressure-equivalent velocity
    """
    I = uModelToI_p(u_model, p_rms, A_param=A_param) # note a is not applied here to preserve original pressure first
    scaling = ventilationBlendedScaling_p(I, I_crit=I_crit)
    return a * u_model * scaling