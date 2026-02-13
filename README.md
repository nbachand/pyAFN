# pyAFN: Airflow Network Modeling Package

A Python package for airflow network (AFN) modeling with ventilation scaling, flow calculations, and pressure optimization.

## Features

- **Ventilation Scaling**: Functions for calculating ventilation scaling factors based on Richardson numbers
- **Flow Calculations**: Convert between flow rates and pressure differences using orifice equations
- **Pressure Optimization**: Solve for indoor pressures that satisfy mass balance in airflow networks
- **Physical Constants**: Predefined constants for air properties and building dimensions

## Installation

### From source

```bash
git clone https://github.com/yourusername/pyAFN.git
cd pyAFN
pip install -e .
```

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- tqdm >= 4.60.0

## Package Structure

```
pyAFN/
├── pyafn/
│   ├── __init__.py          # Package initialization and exports
│   ├── constants.py         # Physical constants and parameters
│   ├── ventilation.py       # Ventilation scaling functions
│   ├── flow.py             # Flow and pressure calculations
│   └── solver.py           # Optimization solver functions
├── setup.py                 # Package setup configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Usage

### Basic Import

```python
import pyafn

# Access constants
print(pyafn.rho)  # Air density
print(pyafn.g)    # Gravitational acceleration

# Use flow functions
flow_rate = pyafn.flowFromP(rho=1.225, C_d=0.6, A=1.0, delp=10.0)

# Use solver
result = pyafn.findOptimalP0(rho=1.225, flowParams)
```

### Module-level Import

```python
from pyafn import flow, ventilation, solver, constants

# Use specific functions
pressure = flow.pFromFlow(constants.rho, C_d=0.6, A=1.0, q=0.5)
scaling = ventilation.ventilationBlendedScaling_q(Rq)
result = solver.findOptimalP0(constants.rho, flowParams)
```

## Modules

### constants

Physical constants used throughout the package:
- `g`: Gravitational acceleration (m/s²)
- `beta`: Thermal expansion coefficient (1/K)
- `rho`: Air density (kg/m³)
- `A`: Reference area (m²) - default value

### ventilation

Functions for ventilation scaling:
- `ventilationBlendedScaling_q()`: Velocity-based scaling
- `ventilationBlendedScaling_p()`: Pressure-based scaling
- `ventilationReDecomp_q()`: Recompose velocity with scaling
- `ventilationReDecomp_p(u_model, a, p_rms, A_param, rho_param)`: Recompose pressure with scaling (requires A and rho as parameters)

### flow

Flow and pressure calculations:
- `flowFromP()`: Calculate flow from pressure difference
- `pFromFlow()`: Calculate pressure from flow rate
- `CFromFlow()`: Calculate discharge coefficient
- `getWindBuoyantP()`: Calculate wind and buoyancy pressures
- `flowField()`: Calculate flow field for network
- `getC()`: Get discharge coefficients from measurements

### solver

Optimization functions:
- `findOptimalP0()`: Find optimal indoor pressures
- `qObjective()`: Objective function for optimization
- `matchObjective()`: Combined objective for matching flows and regularizing C_d
- `findOptimalP0AndC()`: Jointly optimize pressures and discharge coefficients

## Flow Parameters Dictionary

Several functions require a `flowParams` dictionary with the following keys:

**Required:**
- `C_d`: Discharge coefficients (array)
- `A`: Opening areas (m², array)
- `rooms`: Room connectivity matrix
- `p_w`: Wind pressure (Pa, array)
- `z`: Height (m, array)
- `delT`: Temperature difference (K, array or 2D array)
- `hr`: Reference height (m)

**Optional:**
- `q`: Measured flow rates (for `getC()` function)

## Example

```python
import numpy as np
import pyafn

# Define flow parameters
flowParams = {
    "C_d": np.array([0.6, 0.6, 0.6]),
    "A": np.array([1.0, 1.0, 1.0]),
    "rooms": np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]]),
    "p_w": np.array([5.0, 0.0, -5.0]),
    "z": np.array([3.0, 3.0, 3.0]),
    "delT": np.array([2.0, 0.0, -2.0]),
    "hr": 3.0  # Reference height (m)
}

# Solve for optimal pressures
result = pyafn.findOptimalP0(pyafn.rho, flowParams)
print(f"Optimal pressures: {result.x}")

# Calculate resulting flows
flows = pyafn.flowField(result.x, pyafn.rho, flowParams)
print(f"Flow rates: {flows}")
```

### Advanced: Joint Optimization of Pressures and Discharge Coefficients

```python
# When you have target/measured flows, you can jointly optimize
# pressures and discharge coefficients
flowParams_with_targets = flowParams.copy()
flowParams_with_targets["q"] = np.array([0.5, -0.3, -0.2])  # Measured flows

# Jointly optimize pressures and C_d values
result = pyafn.findOptimalP0AndC(pyafn.rho, flowParams_with_targets, weight=0.1)

# Extract optimized values
n_rooms = flowParams["rooms"].shape[1]
optimal_pressures = result.x[:n_rooms]
optimal_Cd = result.x[n_rooms:]

print(f"Optimal pressures: {optimal_pressures}")
print(f"Optimal discharge coefficients: {optimal_Cd}")
```

## Development

To install in development mode with testing dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License

## Author

Nicholas Bachand

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
