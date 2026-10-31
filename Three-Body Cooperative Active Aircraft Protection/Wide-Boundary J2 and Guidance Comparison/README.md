# Three-Body Cooperative Active Aircraft Protection Guidance
Please go through the report for More information about 3 body engagement scenario.

# J2 Guidance Comparison

A comparison of two cooperative J2 guidance laws for a
planar aircraft–defender–attacker engagement.

## Architectures 

This codebase contains  two guidance architectures:

1. **2021 Baseline — Narrow-Boundary J2 (NB J2)**  
   Its reaching term is `M2_2021 * tanh(S / smc_eps)`.

2. **WB J2 — Wide-Boundary J2 Guidance**  
   Its reaching term is `M2_wb * sat(S / smc_eps_wb)`, where
   `sat(x) = clip(x, -1, 1)`.

The two laws share the same J2 cooperative-control formulation, engagement
model, vehicle parameters, attacker proportional-navigation law, numerical
integrator, and scenario definitions. The reaching law and its associated
boundary-width parameter are the intended architectural differences.

## What is included

- The five requested evaluation scenarios:
  - `ideal`
  - `lag`
  - `noise`
  - `noise+lag`
  - `tight_miss`
- Six retained performance submetrics:
  - Relative control effort
  - Miss distance
  - `int_a_d2` — integral of squared defender acceleration
  - Defender saturation fraction
  - Chattering total variation (TV)
  - Terminal acceleration ratio, `|a_d(final)| / |a_m(final)|`
- A single report entry point: `report.py`.
- A generated Markdown verdict and the requested plots.


## Requirements

- Python 3.9 or later
- NumPy
- Matplotlib

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
python report.py
```



## Output layout

```text
J2_Guidance_Comparison/
├── config.py
├── report.py
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── attacker.py
│   ├── dynamics.py
│   ├── guidance_2021.py
│   ├── guidance_wb_j2.py
│   ├── metrics.py
│   └── simulator.py
└── results/
    ├── VERDICT.md
    └── figures/
        ├── ideal/
        ├── lag/
        ├── noise/
        ├── noise+lag/
        └── tight_miss/
```

Each scenario's figure folder contains:

| Figure | Contents |
|---|---|
| `trajectory_xy.png` | X–Y trajectories of aircraft, attacker, and both defenders |
| `accelerations_vs_time.png` | Three stacked subplots: aircraft, defender, missile (attacker) acceleration |
| `surface_S_vs_time.png` | Sliding surface `S` in degrees per second |
| `relative_control_effort_vs_gamma_m0.png` | Relative control effort versus initial attacker heading |
| `int_a_d2_vs_gamma_m0.png` | Defender control cost versus initial attacker heading |
| `chattering_TV_vs_gamma_m0.png` | Chattering TV versus initial attacker heading |
| `miss_distance_vs_gamma_m0.png` | Miss distance versus initial attacker heading |
| `miss_distance_vs_range.png` | Miss distance versus initial aircraft–attacker range |

## Verdict conventions

The verdict compares WB J2 against the 2021 Baseline. A positive percentage
means WB J2 is better; a negative percentage means the 2021 Baseline is better.

- **Higher is better:** Relative control effort.
- **Lower is better:** Miss distance, `int_a_d2`, defender saturation,
  chattering TV, and terminal acceleration ratio.

For a lower-is-better metric:

```text
improvement (%) = (baseline - WB_J2) / abs(baseline) * 100
```

For the higher-is-better relative-control-effort metric:

```text
improvement (%) = (WB_J2 - baseline) / abs(baseline) * 100
```


## Configuration

Edit `config.py` to change common simulation settings or the five scenario
overrides. The architecture-specific reaching-law parameters are:

| Parameter | Applies to | Meaning |
|---|---|---|
| `M2_2021` | 2021 Baseline | NB J2 reaching gain |
| `smc_eps` | 2021 Baseline | Tanh boundary-layer width |
| `M2_wb` | WB J2 | WB J2 reaching gain |
| `smc_eps_wb` | WB J2 | Wide saturation boundary width |

Keep the two parameter sets explicit when running a controlled comparison.
Changing them changes the experiment and should be documented when reporting
results.

## Model and numerical notes

The engagement is a planar, constant-speed, point-mass model. The aircraft and
defender cooperate; the attacker uses proportional navigation against the
aircraft. Guidance commands are clipped to the configured acceleration
limits. Autopilot lag, when enabled, is represented by a first-order response.
LOS-rate noise is injected into the geometry supplied to the guidance law,
while the true geometry is used for propagation and event detection.
Integration uses forward Euler with step size `dt`.

`int_a_d2` is calculated from the actual defender acceleration history using
the trapezoidal rule. Defender saturation fraction is the proportion of
samples where the unsaturated defender command exceeds its configured
acceleration limit. Chattering TV is the sum of absolute sample-to-sample
changes in the commanded defender acceleration during the final second of
the engagement.

## Reproducibility

Noise uses the configured `noise_seed`, and both architectures receive the
same scenario overrides and noise seed. The simulator creates its own seeded
generator for each run. The report is deterministic for a fixed Python,
NumPy, Matplotlib, and configuration environment.
