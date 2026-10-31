"""Generate the complete two-architecture comparison report.

Run from the project root:
    python report.py

The heading and range sweeps are executed internally. Only figures and
results/VERDICT.md are written; no sweep progress or intermediate data files
are emitted.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, SCENARIOS
from src.metrics import compute_metrics
from src.simulator import ARCHITECTURES, run_simulation

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# The same heading/range points are used for both architectures.
GAMMA_VALUES = np.array([90, 110, 130, 150, 170, 190, 210, 230, 250, 270], dtype=float)
RANGE_VALUES = np.array([3000, 4000, 5000, 6000, 7000], dtype=float)

METRIC_LABELS = {
    "relative_control_effort": "Relative control effort",
    "miss_distance": "Miss distance (m)",
    "int_a_d2": "Defender control cost, integral a_d^2 dt",
    "defender_saturation": "Defender saturation fraction",
    "chattering_TV": "Chattering TV",
    "terminal_acc_ratio": "Terminal acceleration ratio (defender / missile)",
}
HIGHER_IS_BETTER = {"relative_control_effort"}
ARCH_STYLE = {
    "2021 Baseline": {"linestyle": "--"},
    "WB J2": {"linestyle": "-"},
}

def run_pair(overrides):
    """Run both architectures with identical scenario/configuration inputs."""
    simulations = {}
    metrics = {}
    for architecture in ARCHITECTURES:
        result = run_simulation(architecture, CONFIG, overrides)
        simulations[architecture] = result
        metrics[architecture] = compute_metrics(result)
    return simulations, metrics


def save_figure(fig, relative_path):
    path = FIGURES / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(scenario, simulations):
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    baseline = simulations["2021 Baseline"]["state"]
    ax.plot(baseline[:, 0], baseline[:, 1], color="0.25",
            linewidth=1.7, label="Aircraft")
    ax.plot(baseline[:, 6], baseline[:, 7], color="0.55",
            linewidth=1.7, label="Attacker")
    for architecture in ARCHITECTURES:
        state = simulations[architecture]["state"]
        ax.plot(state[:, 3], state[:, 4],
                label=f"Defender — {architecture}",
                linewidth=1.8, **ARCH_STYLE[architecture])
    ax.scatter([baseline[0, 0]], [baseline[0, 1]], marker="o",
               color="0.25", s=35, zorder=5)
    ax.scatter([baseline[0, 3]], [baseline[0, 4]], marker="s",
               color="0.25", s=35, zorder=5)
    ax.scatter([baseline[0, 6]], [baseline[0, 7]], marker="^",
               color="0.55", s=40, zorder=5)
    ax.set(title=f"Engagement trajectories — {scenario}",
           xlabel="X position (m)", ylabel="Y position (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, Path(scenario) / "trajectory_xy.png")


def plot_accelerations(scenario, simulations):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for architecture in ARCHITECTURES:
        result = simulations[architecture]
        style = ARCH_STYLE[architecture]
        axes[0].plot(result["t"], result["a_a_actual"],
                     label=architecture, linewidth=1.15, **style)
        axes[1].plot(result["t"], result["a_d_actual"],
                     label=architecture, linewidth=1.15, **style)
        axes[2].plot(result["t"], result["a_m"],
                     label=architecture, linewidth=1.15, **style)
    axes[0].set_ylabel("Aircraft accel. (m/s²)")
    axes[1].set_ylabel("Defender accel. (m/s²)")
    axes[2].set_ylabel("Missile (attacker) accel. (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"Vehicle accelerations — {scenario}")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, Path(scenario) / "accelerations_vs_time.png")


def plot_surface(scenario, simulations):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for architecture in ARCHITECTURES:
        result = simulations[architecture]
        ax.plot(result["t"], np.rad2deg(result["S"]),
                label=architecture, linewidth=1.25,
                **ARCH_STYLE[architecture])
    ax.set(title=f"Sliding surface — {scenario}",
           xlabel="Time (s)", ylabel="S (deg/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, Path(scenario) / "surface_S_vs_time.png")


def run_heading_sweep(scenario):
    metrics = {
        key: np.full((len(GAMMA_VALUES), len(ARCHITECTURES)), np.nan)
        for key in METRIC_LABELS
    }
    for i, gamma in enumerate(GAMMA_VALUES):
        overrides = dict(SCENARIOS[scenario])
        overrides["gamma_m0_deg"] = float(gamma)
        _, values = run_pair(overrides)
        for j, architecture in enumerate(ARCHITECTURES):
            for key in METRIC_LABELS:
                metrics[key][i, j] = values[architecture][key]
    return metrics


def run_range_sweep(scenario):
    metrics = {
        key: np.full((len(RANGE_VALUES), len(ARCHITECTURES)), np.nan)
        for key in METRIC_LABELS
    }
    for i, initial_range in enumerate(RANGE_VALUES):
        overrides = dict(SCENARIOS[scenario])
        overrides["r_am0"] = float(initial_range)
        _, values = run_pair(overrides)
        for j, architecture in enumerate(ARCHITECTURES):
            for key in METRIC_LABELS:
                metrics[key][i, j] = values[architecture][key]
    return metrics


def plot_heading_metric(scenario, key, values):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for j, architecture in enumerate(ARCHITECTURES):
        ax.plot(GAMMA_VALUES, values[key][:, j],
                label=architecture, linewidth=1.7,
                **ARCH_STYLE[architecture])
    ax.set(title=f"{METRIC_LABELS[key]} vs initial attacker heading — {scenario}",
           xlabel="Initial attacker heading, gamma_m0 (deg)",
           ylabel=METRIC_LABELS[key])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, Path(scenario) / f"{key}_vs_gamma_m0.png")


def plot_range_miss_distance(scenario, values):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for j, architecture in enumerate(ARCHITECTURES):
        ax.plot(RANGE_VALUES, values["miss_distance"][:, j],
                label=architecture, linewidth=1.7,
                **ARCH_STYLE[architecture])
    ax.set(title=f"Miss distance vs initial range — {scenario}",
           xlabel="Initial aircraft–attacker range (m)",
           ylabel="Miss distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, Path(scenario) / "miss_distance_vs_range.png")


def percent_improvement(baseline, candidate, higher_is_better=False):
    """Positive means WB J2 is better; negative means the baseline is better."""
    if not (np.isfinite(baseline) and np.isfinite(candidate)):
        return None
    if abs(baseline) < 1e-12:
        if abs(candidate) < 1e-12:
            return 0.0
        return None
    if higher_is_better:
        return (candidate - baseline) / abs(baseline) * 100.0
    return (baseline - candidate) / abs(baseline) * 100.0


def fmt_value(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.5g}"


def fmt_improvement(value):
    if value is None:
        return "n/a"
    if abs(value) < 0.005:
        return "0.00%"
    return f"{value:+.2f}%"


def outcome_text(simulations):
    return " | ".join(
        f"{name}: {simulations[name]['outcome']}"
        for name in ARCHITECTURES
    )


def write_verdict(nominal_metrics, nominal_simulations):
    lines = [
        "# J2 Guidance Architecture Comparison",
        "",
        "## Architectures",
        "",
        "- **2021 Baseline:** Narrow-Boundary J2 (NB J2), using the original "
        "tanh reaching term.",
        "- **WB J2:** Wide-Boundary J2 Guidance, using the saturation reaching "
        "term with the wider boundary layer.",
        "",
        "Both architectures use the same engagement model, vehicle parameters, "
        "attacker guidance, integration step, and scenario overrides. Only the "
        "defender/aircraft cooperative guidance reaching law differs.",
        "",
        "## How to read the comparison",
        "",
        "The percentage column is the improvement of WB J2 relative to the "
        "2021 Baseline. **Positive values favour WB J2; negative values favour "
        "the 2021 Baseline.** Relative control effort is higher-is-better; all "
        "other retained submetrics are lower-is-better.",
        "",
        "Relative control effort is defined as "
        "`integral(a_m^2 dt) / integral(a_d^2 dt)`. Defender saturation is "
        "the fraction of simulation samples for which the unsaturated defender "
        "command exceeds its acceleration limit. Chattering TV is the total "
        "variation of the defender acceleration command over the final "
        "one-second interval. Terminal acceleration ratio is "
        "`|a_d(final)| / |a_m(final)|`.",
        "",
        "## Per-scenario results",
        "",
    ]

    scenario_wins = {}
    all_improvements = {key: [] for key in METRIC_LABELS}
    total_wins = 0
    total_comparisons = 0

    for scenario in SCENARIOS:
        metrics = nominal_metrics[scenario]
        simulations = nominal_simulations[scenario]
        lines += [
            f"### {scenario}",
            "",
            f"Engagement outcomes: {outcome_text(simulations)}.",
            "",
            "| Submetric | 2021 Baseline | WB J2 | WB J2 improvement | Better |",
            "|---|---:|---:|---:|---|",
        ]
        wins = 0
        for key, label in METRIC_LABELS.items():
            baseline = metrics["2021 Baseline"][key]
            candidate = metrics["WB J2"][key]
            improvement = percent_improvement(
                baseline, candidate, key in HIGHER_IS_BETTER
            )
            if improvement is not None:
                all_improvements[key].append(improvement)
            better = (
                "WB J2" if improvement is not None and improvement > 1e-9
                else "2021 Baseline" if improvement is not None and improvement < -1e-9
                else "Tie" if improvement is not None
                else "n/a"
            )
            if better == "WB J2":
                wins += 1
            if improvement is not None:
                total_comparisons += 1
            lines.append(
                f"| {label} | {fmt_value(baseline)} | {fmt_value(candidate)} "
                f"| {fmt_improvement(improvement)} | {better} |"
            )
        scenario_wins[scenario] = wins
        total_wins += wins
        lines += [
            "",
            f"WB J2 improves **{wins} of {len(METRIC_LABELS)}** retained "
            "submetrics in this scenario.",
            "",
        ]

    lines += [
        "## Overall summary",
        "",
        "| Submetric | Mean WB J2 improvement across scenarios | Scenarios favouring WB J2 |",
        "|---|---:|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        vals = all_improvements[key]
        mean = float(np.mean(vals)) if vals else None
        count = sum(v > 1e-9 for v in vals)
        lines.append(
            f"| {label} | {fmt_improvement(mean)} | {count}/{len(SCENARIOS)} |"
        )

    lines += [
        "",
        f"Across the {len(SCENARIOS)} scenarios and {len(METRIC_LABELS)} "
        f"submetrics ({total_comparisons} valid comparisons), WB J2 improves "
        f"**{total_wins}** comparisons.",
        "",
        "This is a metric-by-metric numerical comparison, not a claim that one "
        "architecture dominates in every operating condition. A positive "
        "percentage denotes an improvement under the direction convention "
        "above; it does not imply statistical significance.",
        "",
        "## Figures",
        "",
        "Each scenario folder under `figures/` contains:",
        "",
        "- `trajectory_xy.png` — aircraft, attacker, and both defender trajectories.",
        "- `accelerations_vs_time.png` — aircraft, defender, and attacker "
        "accelerations in three stacked subplots.",
        "- `surface_S_vs_time.png` — sliding surface in degrees per second.",
        "- `relative_control_effort_vs_gamma_m0.png`.",
        "- `int_a_d2_vs_gamma_m0.png`.",
        "- `chattering_TV_vs_gamma_m0.png`.",
        "- `miss_distance_vs_gamma_m0.png`.",
        "- `miss_distance_vs_range.png`.",
        "",
        "The heading sweep varies the initial attacker heading over "
        f"{', '.join(str(int(v)) for v in GAMMA_VALUES)} degrees. The range "
        "sweep varies initial aircraft–attacker separation over "
        f"{', '.join(str(int(v)) for v in RANGE_VALUES)} metres. These sweeps "
        "are computed internally by `report.py`; no sweep tables or "
        "intermediate data files are emitted.",
        "",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    nominal_metrics = {}
    nominal_simulations = {}

    # Nominal scenario figures and verdict data.
    for scenario, overrides in SCENARIOS.items():
        simulations, metrics = run_pair(overrides)
        nominal_simulations[scenario] = simulations
        nominal_metrics[scenario] = metrics
        plot_trajectory(scenario, simulations)
        plot_accelerations(scenario, simulations)
        plot_surface(scenario, simulations)

    # Requested heading-sweep plots. Sweeps are intentionally silent.
    heading_plot_keys = (
        "relative_control_effort",
        "int_a_d2",
        "chattering_TV",
        "miss_distance",
    )
    for scenario in SCENARIOS:
        heading_data = run_heading_sweep(scenario)
        for key in heading_plot_keys:
            plot_heading_metric(scenario, key, heading_data)

    # Requested range-sweep miss-distance plots.
    for scenario in SCENARIOS:
        range_data = run_range_sweep(scenario)
        plot_range_miss_distance(scenario, range_data)

    write_verdict(nominal_metrics, nominal_simulations)
    print(f"Report complete: {RESULTS / 'VERDICT.md'}")
    print(f"Figures saved under: {FIGURES}")


if __name__ == "__main__":
    main()
