"""Ventilation scaling functions for AFN calculations.

This module contains functions for calculating ventilation scaling factors
from the fluctuation intensity I and the fluctuation measure H.

H is a normalized lower-order mean of the fluctuations, used in the
fluctuation-dominated correction q_bar ~= q_MP / (2H):
  - Ventilation form (symmetric flow fluctuations):
        H(q) = mean(|q'|) / |q_MP|            (arithmetic mean of |q'|)
  - Pressure form (symmetric pressure fluctuations):
        H(dp) = M_-1(sqrt|dp'|) / sqrt|dp_bar| (harmonic mean of sqrt|dp'|)
Power-mean monotonicity gives the upper bounds H_U(q) = I(q) (M_1 <= M_2) and
H_U(dp) = sqrt(2 I(dp)). The blended (piecewise) models replace H by a
tangent model H_T so that the fluctuation- and mean-dominated branches meet
tangentially at I*:
  - H_T(q) = I,                     I* = 1/sqrt(2)
  - H_T(dp) = sqrt((3 sqrt(3)/8) I), I* = 1/sqrt(3)
"""

import numpy as np
from .constants import g, beta, rho, Cd


def ventilationLowerScaling(H):
    """Calculate fluctuation-dominated ventilation scaling, q_bar / q_MP = 1 / (2H).
    
    Args:
        H: Normalized fluctuation mean (mean(|q'|)/|q_MP| in the ventilation
           form; harmonic mean of sqrt|dp'| over sqrt|dp_bar| in the pressure form)
        
    Returns:
        Fluctuation-dominated scaling factor
    """
    return 0.5 / H


def ventilationUpperScaling(I):
    """Calculate mean-dominated ventilation scaling, q_bar / q_MP = sqrt(1 - I^2).
    
    Args:
        I: Fluctuation intensity (array)
        
    Returns:
        Mean-dominated scaling factor (binomial approximation); 0 for I >= 1
    """
    scaling = np.zeros_like(I, dtype=float)
    mask = I < 1
    scaling[mask] = np.sqrt(1 - I[mask] ** 2)
    return scaling


def ventilationScalingSwitch(H, I, I_crit):
    """Switch between fluctuation- and mean-dominated scaling.
    
    Args:
        H: Normalized fluctuation mean used for I >= I_crit
        I: Fluctuation intensity
        I_crit: Critical fluctuation intensity for switching
        
    Returns:
        Blended scaling factor: 1/(2H) for I >= I_crit, sqrt(1 - I^2) otherwise
    """
    scaling = np.empty_like(I, dtype=float)
    mask = I >= I_crit
    scaling[mask] = ventilationLowerScaling(H[mask])
    scaling[~mask] = ventilationUpperScaling(I[~mask])
    return scaling


def ventilationBlendedScaling_q(I, I_crit=1 / np.sqrt(2)):
    """Calculate blended ventilation scaling for ventilation-based fluctuation intensity.
    
    Uses the tangent model H_T(q) = H_U(q) = I, which meets sqrt(1 - I^2)
    tangentially at I* = 1/sqrt(2).
    
    Args:
        I: Fluctuation intensity I(q) = sigma_q / |q_MP|
        
    Returns:
        Blended scaling factor for ventilation-based calculations
    """
    alpha = 1.0
    H_bound = I
    H_tangent = alpha * H_bound
    scaling = ventilationScalingSwitch(H_tangent, I, I_crit)
    return scaling


def ventilationBlendedScaling_p(I, I_crit =1 / np.sqrt(3)):
    """Calculate blended ventilation scaling for pressure-based fluctuation intensity.
    
    Uses the tangent model H_T(dp) = sqrt((3 sqrt(3)/8) I), i.e. the bound
    H_U(dp) = sqrt(2I) scaled by sqrt(3 sqrt(3)/16), which meets
    sqrt(1 - I^2) tangentially at I* = 1/sqrt(3).
    
    Args:
        I: Fluctuation intensity I(dp) = sigma_dp / (2 |dp_bar|)
        
    Returns:
        Blended scaling factor for pressure-based calculations
    """
    # Tangential-intersect model: H_T = sqrt(alpha) * H_U.
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