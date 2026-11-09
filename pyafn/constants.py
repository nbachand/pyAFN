"""Physical constants used in AFN calculations.

This module defines the physical constants used throughout the pyAFN package.
"""

# Physical constants
# g is set to 10 m/s² (not 9.81) to match the LES used to calibrate the model.
g = 10  # Gravitational acceleration (m/s²)
beta = 0.0034  # Thermal expansion coefficient (1/K)
rho = 1.225  # Air density (kg/m³)
Cd = 0.611  # Discharge coefficient for openings