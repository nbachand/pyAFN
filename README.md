# pyAFN: Airflow Network Modeling with Fluctuation Corrections

pyAFN is a Python package for airflow network (AFN) modeling of naturally
ventilated buildings. It solves the standard orifice-equation network for room
pressures and opening flows. It also corrects those mean flows for turbulent
fluctuations, using RMS pressure (`pRMS`) or RMS flow (`qRMS`) at each opening.

## Features

- **Fluctuation corrections**: scale orifice-equation flows by a factor that
  depends on the fluctuation intensity `I` at each opening. The factor comes
  from either RMS pressure or RMS flow, and blends a mean-dominated branch with
  a fluctuation-dominated branch.
- **Network solver**: find the room pressures that satisfy mass balance in
  single- or multi-room networks, with wind and buoyancy (stack) forcing.
- **Calibration**: back-calculate discharge coefficients from measured flows,
  or jointly fit room pressures and discharge coefficients.
- **Flow/pressure conversions**: orifice-equation helpers for flow from
  pressure, pressure from flow, and discharge coefficient from both.

## Installation

```bash
git clone https://github.com/nbachand/pyAFN.git
cd pyAFN
pip install -e .
```

Dependencies: `numpy >= 1.20.0`, `scipy >= 1.7.0`.

## Constants

`pyafn.constants` defines the following. All are re-exported at package level
(`pyafn.g`, etc.):

| Name   | Value  | Meaning                                     |
| ------ | ------ | ------------------------------------------- |
| `g`    | 10     | Gravitational acceleration (m/s²)           |
| `beta` | 0.0034 | Thermal expansion coefficient of air (1/K)  |
| `rho`  | 1.225  | Air density (kg/m³)                         |
| `Cd`   | 0.611  | Default discharge coefficient for openings  |

> **Note:** `g` is 10 m/s², not 9.81 m/s². This matches the large-eddy
> simulations (LES) the fluctuation model was developed against. Every module
> copies the constants when `pyafn` is imported, so reassigning `pyafn.g` at
> runtime has no effect. To use standard gravity, edit `g` in
> `pyafn/constants.py`.

The flow functions do not take density as an argument; they use `rho` from
`pyafn.constants`.

## Quick start: steady network

```python
import numpy as np
import pyafn

# Single room, three openings
flowParams = pyafn.createFlowParams(
    C_d=[pyafn.Cd, pyafn.Cd, pyafn.Cd],
    A=[1.0, 0.5, 1.5],          # opening areas (m²)
    p_w=[5.0, 2.0, -4.0],       # wind pressure at each opening (Pa)
    z=[1.0, 2.5, 1.0],          # opening heights (m)
    delT=[2.0, 2.0, 2.0],       # T_indoor - T_outdoor (K)
    rooms=[[1], [1], [1]],      # connectivity: openings x rooms
    hr=3.0,                     # room height (m)
)

result = pyafn.findOptimalP0(flowParams)
p_0 = result.x                            # indoor pressure(s), Pa
q = pyafn.flowField(p_0, flowParams)      # flow through each opening, m³/s
print(p_0, q, q.sum())                    # q.sum() ~ 0 (mass balance)
```

Orifice-equation helpers:

```python
q  = pyafn.flowFromP(C_d=pyafn.Cd, A=1.0, delp=10.0)   # m³/s
dp = pyafn.pFromFlow(C_d=pyafn.Cd, A=1.0, q=0.5)       # Pa
```

## Fluctuation corrections (pRMS / qRMS)

The orifice equation applied to the mean pressure difference gives a
mean-pressure flow `q_MP`. When the flow fluctuates strongly, the true mean flow
differs from `q_MP`. pyAFN corrects it as `q̄ = q_MP · S(I)`, where `I` is a
fluctuation intensity at each opening.

Pass **one** of the following in `flowParams` to turn the correction on
(passing both raises a `ValueError`):

| Key    | Shape    | Meaning                              | Intensity                 |
| ------ | -------- | ------------------------------------ | ------------------------- |
| `pRMS` | (M, 2)   | RMS pressure difference (Pa)         | `I = pRMS / (2 \|Δp̄\|)`    |
| `qRMS` | (M, 2)   | RMS flow rate (m³/s)                 | `I = qRMS / \|q_MP\|`      |

`M` is the number of openings. Column 0 is used when the opening's mean flow
goes into the room (Δp > 0 or q > 0). Column 1 is used when it goes out. To use
one value for both directions, repeat it in both columns.

The scaling `S(I)` has two branches that meet tangentially at a critical
intensity `I*`:

- mean-dominated (`I < I*`): `S = sqrt(1 - I²)`
- fluctuation-dominated (`I ≥ I*`): `S = 1 / (2 H_T)`

| Form        | Function                       | `H_T`                       | `I*`        |
| ----------- | ------------------------------ | --------------------------- | ----------- |
| Ventilation | `ventilationBlendedScaling_q`  | `I`                         | `1/sqrt(2)` |
| Pressure    | `ventilationBlendedScaling_p`  | `sqrt((3 sqrt(3) / 8) · I)` | `1/sqrt(3)` |

See the docstring of `pyafn/ventilation.py` for how the fluctuation measure `H`
is defined and where these tangent models come from.

### Example: network with pressure fluctuations

```python
fp = dict(flowParams)
fp["pRMS"] = np.array([[3.0, 3.0],
                       [1.0, 1.0],
                       [4.0, 4.0]])   # Pa, [inflow, outflow] per opening

result = pyafn.findOptimalP0(fp)
q = pyafn.flowField(result.x, fp)   # fluctuation-corrected flows
```

`createFlowParams` also accepts `pRMS=` and `qRMS=` keyword arguments.

> **Note:** for a single room with exactly two openings, `findOptimalP0` uses a
> closed-form solution when it applies, i.e. without `pRMS`/`qRMS` and with one
> `C_d` per opening. Otherwise it solves numerically. Pass
> `checkAnalytical=False` to always solve numerically.

### Applying the correction to a single opening

Outside a network, the `ReDecomp` functions apply the correction directly to a
modeled velocity or flow:

```python
u = np.array([0.5, 1.0, 2.0])

# From RMS flow/velocity: I = u_rms / |a * u|
pyafn.ventilationReDecomp_q(u, a=1.0, u_rms=0.6)

# From RMS pressure: u is converted to Δp with Cd and A_param, then I = p_rms / (2|Δp|)
pyafn.ventilationReDecomp_p(u, a=1.0, p_rms=1.0, A_param=1.0)
```

Both return `a * u * S(I)`.

## The `flowParams` dictionary

Build it with `pyafn.createFlowParams(...)`, which converts lists to numpy
arrays. `M` is the number of openings and `N` the number of rooms.

| Key     | Shape            | Required | Meaning |
| ------- | ---------------- | -------- | ------- |
| `C_d`   | (M,) or (M, 2)   | yes      | Discharge coefficients. With shape (M, 2), column 0 applies to inflow and column 1 to outflow. |
| `A`     | (M,)             | yes      | Opening areas (m²) |
| `rooms` | (M, N)           | yes      | Connectivity. +1 means positive flow through the opening enters that room; -1 means it leaves. An internal opening between two rooms has +1 in one column and -1 in the other. |
| `p_w`   | (M,)             | yes      | Wind pressure at each opening (Pa); use 0 for internal openings |
| `z`     | (M,)             | yes      | Opening height (m) |
| `delT`  | (M,) or (M, L)   | yes      | Indoor minus outdoor temperature (K). With shape (M, L), each column is one of L equal-height layers spanning `hr`. |
| `hr`    | scalar           | yes      | Room height (m) |
| `q`     | (M,)             | no       | Measured flows (m³/s), used by `getC` and `findOptimalP0AndC` |
| `pRMS`  | (M, 2)           | no       | RMS pressure, turns on the pressure-form correction |
| `qRMS`  | (M, 2)           | no       | RMS flow, turns on the ventilation-form correction |

Sign convention: positive flow is into the room. The driving pressure at each
opening is the wind plus buoyancy pressure minus the room pressure `p_0`.

## API overview

### `pyafn.flow`

- `createFlowParams(...)`: build a `flowParams` dictionary
- `flowFromP(C_d, A, delp)`: flow from pressure difference
- `pFromFlow(C_d, A, q)`: pressure difference from flow
- `CFromFlow(q, A, delp)`: discharge coefficient from flow and pressure
- `getWindBuoyantP(flowParams)`: wind plus buoyancy driving pressure per opening
- `flowField(p_0, flowParams)`: flow through every opening for room pressures `p_0`, with the fluctuation correction if `pRMS`/`qRMS` is present
- `getC(p_0, flowParams)`: discharge coefficients implied by measured `q`

### `pyafn.solver`

- `findOptimalP0(flowParams, checkAnalytical=True)`: room pressures that satisfy mass balance (returns a `scipy.optimize.OptimizeResult`)
- `qObjective(p_0, flowParams)`: sum of squared room flow imbalances
- `findOptimalP0AndC(flowParams, weight=0.1, disp=False)`: jointly fit room pressures and `C_d` to measured `q`; `result.x = [p0_1..p0_N, Cd_1..Cd_M]`
- `matchObjective(x, flowParams, weight)`: objective used by `findOptimalP0AndC`

### `pyafn.ventilation`

- `ventilationBlendedScaling_q(I, I_crit=1/sqrt(2))`: ventilation-form scaling `S(I)`
- `ventilationBlendedScaling_p(I, I_crit=1/sqrt(3))`: pressure-form scaling `S(I)`
- `ventilationReDecomp_q(u_model, a, u_rms, I_crit=...)`: apply the ventilation-form correction to a modeled flow
- `ventilationReDecomp_p(u_model, a, p_rms, A_param=1, I_crit=...)`: apply the pressure-form correction to a modeled flow
- `ventilationLowerScaling(H)`, `ventilationUpperScaling(I)`, `ventilationScalingSwitch(H, I, I_crit)`: the individual branches and the switch between them

## Calibration example

```python
fp = dict(flowParams)
fp["q"] = np.array([1.9, 0.6, -2.5])   # measured flows (m³/s)

result = pyafn.findOptimalP0AndC(fp, weight=0.1)
N = fp["rooms"].shape[1]
p_0, C_d = result.x[:N], result.x[N:]
```

More examples, including two-room networks, layered temperatures and
direction-dependent `C_d`, are in [examples.ipynb](examples.ipynb).

## Development

```bash
pip install -e ".[dev]"
```

## License

MIT License. See [LICENSE](LICENSE).

## Author

Nicholas Bachand
