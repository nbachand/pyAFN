"""pyAFN: Airflow Network (AFN) Modeling Package

A Python package for airflow network modeling with ventilation scaling,
flow calculations, and pressure optimization.
"""

__version__ = "0.1.0"

# Import constants
from .constants import g, beta, rho, A

# Import ventilation functions
from .ventilation import (
    ventilationLowerScaling,
    ventilationUpperScaling,
    ventilationScalingSwitch,
    ventilationBlendedScaling_q,
    ventilationBlendedScaling_p,
    ventilationReDecomp_q,
    ventilationReDecomp_p,
)

# Import flow functions
from .flow import (
    createFlowParams,
    getWindBuoyantP,
    flowFromP,
    pFromFlow,
    CFromFlow,
    flowField,
    getC,
)

# Import solver functions
from .solver import (
    qObjective,
    findOptimalP0,
    matchObjective,
    findOptimalP0AndC,
)

__all__ = [
    # Constants
    "g", "beta", "rho", "A",
    # Ventilation functions
    "ventilationLowerScaling",
    "ventilationUpperScaling",
    "ventilationScalingSwitch",
    "ventilationBlendedScaling_q",
    "ventilationBlendedScaling_p",
    "ventilationReDecomp_q",
    "ventilationReDecomp_p",
    # Flow functions
    "createFlowParams",
    "getWindBuoyantP",
    "flowFromP",
    "pFromFlow",
    "CFromFlow",
    "flowField",
    "getC",
    # Solver functions
    "qObjective",
    "findOptimalP0",
    "matchObjective",
    "findOptimalP0AndC",
]
