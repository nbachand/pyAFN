import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

g = 10
beta = 0.0034
rho = 1.225
# window_dim = hr/4
# A = window_dim ** 2 
A = 1 # predicting per area flux, so A is already included in flux


def ventilationLowerScaling(Rh):
    return Rh / 2

def ventilationUpperScaling(Rq):
    scaling = np.zeros_like(Rq)
    mask = Rq > 0
    # scaling[mask] = np.sqrt(1 - Rq[mask]**-2)
    scaling[mask] = 1 - Rq[mask]**-2 / 2  + Rq[mask]**-4 / 8 # Binomial approximation
    return scaling

def ventilationScalingSwitch(Rh, Rq, Rq_crit):
    scaling = np.empty_like(Rq)
    mask = Rq < Rq_crit
    scaling[mask] = ventilationLowerScaling(Rh[mask])
    scaling[~mask] = ventilationUpperScaling(Rq[~mask])
    return scaling

def ventilationBlendedScaling_q(Rq):
    alpha = 1
    Rh_bound = Rq
    Rh_tangent = alpha * Rh_bound
    Rq_crit = np.sqrt(2)
    scaling = ventilationScalingSwitch(Rh_tangent, Rq, Rq_crit)
    return scaling

def ventilationBlendedScaling_p(Rq):
    alpha = 16 / (3 * np.sqrt(3))  # Scale factor to make the tangent line match the lower bound at the critical point
    Rh_bound = np.sqrt(Rq / 2)
    Rh_tangent = np.sqrt(alpha) * Rh_bound
    Rq_crit = np.sqrt(3)
    scaling = ventilationScalingSwitch(Rh_tangent, Rq, Rq_crit)
    return scaling

    
def ventilationReDecomp_q(u_model, a, u_rms):
    u_model_scaled = u_model * a
    Rq = u_model_scaled / u_rms
    scaling = ventilationBlendedScaling_q(Rq)
    return u_model_scaled * scaling

def ventilationReDecomp_p(u_model, a, p_rms, A_param, rho_param):
    k = A_param * np.sqrt(2 / rho_param)
    delP = u_model**2 / k # not u_model_scaled because we want the original delta P
    Rq = delP / p_rms
    scaling = ventilationBlendedScaling_p(Rq)
    return a * u_model * scaling

def getWindBuoyantP(rho, flowParams):
    p_w = flowParams["p_w"]
    z = flowParams["z"]
    delT = flowParams["delT"]
    hr = flowParams["hr"]
    if len(delT.shape) == 2:
        n_levels = delT.shape[1]
        delz = hr / n_levels
        z_levels = delz * (np.arange(n_levels) + 0.45)  # vectorized center calculation
        z_below = (z_levels < z[:, None])
        delT = np.sum(delT * z_below, axis=1) / np.maximum(np.sum(z_below, axis=1), 1)
        sl_length = z - hr
        sls = sl_length > 0
        delT[sls] = hr / z[sls] * delT[sls] + sl_length[sls] / z[sls] * flowParams["delT"][sls,-1]

    delrho = -rho * beta * delT
    return (delrho * g * z) + p_w # delP is outdoor minus indoor, while p0/rho is indoor minus outdoor, driving positive flow into the room (oppiste textbook)

def flowFromP(rho, C_d, A, delp):
    delp=np.array(delp)
    S = np.sign(delp)
    return S * C_d * A * np.sqrt(2 * abs(delp) / rho)

def pFromFlow(rho, C_d, A, q):
    q = np.array(q)
    S = np.sign(q)
    return S * (q / (C_d * A))**2 * (rho / 2)

def CFromFlow(rho, q, A, delp):
    delp = np.array(delp, dtype=float)
    # prepare output filled with NaNs
    C = np.full_like(delp, np.nan, dtype=float)
    # mask non‐NaN, non‐zero delp
    mask = ~np.isnan(delp) & (delp != 0)
    S = np.sign(delp[mask])
    C[mask] = q[mask] / (S * A[mask] * np.sqrt(2 * np.abs(delp[mask]) / rho))
    return C

def flowField(p_0, rho, flowParams):
    C_d = flowParams["C_d"]
    A = flowParams["A"]
    rooms = flowParams["rooms"]
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(rho, flowParams) 
    return flowFromP(rho, C_d, A, delP)

def getC(p_0, rho, flowParams):
    A = flowParams["A"]
    q = flowParams["q"]
    rooms = flowParams["rooms"]
    delP = -np.matmul(rooms, p_0) + getWindBuoyantP(rho, flowParams)
    return CFromFlow(rho, q, A, delP)

def qObjective(p_0, rho, flowParams):
    qs = flowField(p_0, rho, flowParams)
    rooms = flowParams["rooms"]
    qRooms = np.matmul(rooms.T, qs)
    return np.sum(qRooms**2)

def findOptimalP0(rho, flowParams):
    bounds = np.array([np.min(getWindBuoyantP(rho, flowParams)), np.max(getWindBuoyantP(rho, flowParams))])
    x0 = np.mean(bounds)
    NRooms = flowParams["rooms"].shape[1]
    bounds = np.tile(bounds, (NRooms, 1))
    x0 = np.tile(x0, NRooms)
    return minimize(qObjective, x0=x0, bounds=bounds, args=(rho, flowParams))