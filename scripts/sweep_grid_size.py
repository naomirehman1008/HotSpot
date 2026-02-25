#!/usr/bin/env python3
"""
Grid Size Sweep for HotSpot 3D Thermal Simulation

Sweeps grid resolution from 128x128 down to 2x2 on a 4-tier stack with
default parameters.  For each grid size, generates input files, runs
HotSpot (steady-state only), post-processes results, and records timing
and temperature data into a CSV.  Then plots results vs grid size.

Usage:
    python sweep_grid_size.py                  # full sweep
    python sweep_grid_size.py --test           # quick test (grid sizes 4, 2)
    python sweep_grid_size.py --plot-only      # re-plot from existing CSV
    python sweep_grid_size.py -o my_results/   # custom output dir
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import yaml

# matplotlib is optional — only needed for plotting, not for running the sweep
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATE_SCRIPT = os.path.join(SCRIPT_DIR, "generate_3d_stack.py")
POSTPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "postprocess_results.py")
HOTSPOT_BINARY = os.environ.get(
    "HOTSPOT", os.path.join(SCRIPT_DIR, "..", "hotspot"))

FULL_GRID_SIZES = [2, 4, 8, 16, 24, 32, 48, 64]
TEST_GRID_SIZES = [2, 4]

NUM_TIERS = 4

CSV_HEADER = (
    ["grid_size", "elapsed_s", "peak_temp_K"]
    + [f"tier_{t}_si_max_K" for t in range(NUM_TIERS)]
    + [f"tier_{t}_si_mean_K" for t in range(NUM_TIERS)]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_inputs(grid_size: int, sim_dir: str) -> None:
    """Call generate_3d_stack.py to create HotSpot input files."""
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "-n", str(NUM_TIERS),
        "--grid-size", str(grid_size),
        "-o", sim_dir,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def run_hotspot_steady(sim_dir: str) -> float:
    """Run HotSpot steady-state simulation and return elapsed seconds.

    HotSpot is run from inside sim_dir because the LCF references
    floorplan files by relative path.
    """
    hotspot = os.path.abspath(HOTSPOT_BINARY)
    outputs_dir = os.path.join(sim_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    cmd = [
        hotspot,
        "-c", "hotspot.config",
        "-p", "power.ptrace",
        "-grid_layer_file", "stack.lcf",
        "-model_type", "grid",
        "-detailed_3D", "on",
        "-steady_file", "outputs/steady.out",
        "-grid_steady_file", "outputs/grid_steady.out",
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=sim_dir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t0
    return elapsed


def run_postprocess(sim_dir: str) -> None:
    """Call postprocess_results.py to generate YAML."""
    cmd = [sys.executable, POSTPROCESS_SCRIPT, sim_dir]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def extract_results(sim_dir: str) -> dict:
    """Read the YAML output and extract tier summary data."""
    yaml_path = os.path.join(sim_dir, "outputs", "results.yaml")
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    tiers = data["tiers"]
    peak_temp = max(t["silicon"]["max_K"] for t in tiers)

    result = {"peak_temp_K": peak_temp}
    for t in tiers:
        tier_num = t["tier"]
        result[f"tier_{tier_num}_si_max_K"] = t["silicon"]["max_K"]
        result[f"tier_{tier_num}_si_mean_K"] = t["silicon"]["mean_K"]

    return result


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(grid_sizes: list[int], output_dir: str) -> str:
    """Run the full sweep and write CSV.  Returns path to CSV."""
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    rows: list[dict] = []

    print(f"Sweeping {len(grid_sizes)} grid sizes: {grid_sizes}")
    print(f"Output directory: {output_dir}")
    print(f"HotSpot binary: {HOTSPOT_BINARY}")
    print()

    for gs in grid_sizes:
        sim_dir = os.path.join(output_dir, f"grid_{gs}x{gs}")
        print(f"  [{gs:>3d}x{gs:<3d}] Generating inputs...", end="", flush=True)
        generate_inputs(gs, sim_dir)

        print(" Running HotSpot...", end="", flush=True)
        elapsed = run_hotspot_steady(sim_dir)

        print(f" ({elapsed:.2f}s) Post-processing...", end="", flush=True)
        run_postprocess(sim_dir)

        results = extract_results(sim_dir)
        row = {
            "grid_size": gs,
            "elapsed_s": round(elapsed, 4),
            **results,
        }
        rows.append(row)
        print(f" peak={results['peak_temp_K']:.2f} K")

    # Write CSV
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def read_csv(csv_path: str) -> dict[str, np.ndarray]:
    """Read sweep CSV into numpy arrays keyed by column name."""
    data: dict[str, list] = {col: [] for col in CSV_HEADER}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col in CSV_HEADER:
                data[col].append(float(row[col]))
    return {col: np.array(vals) for col, vals in data.items()}


def plot_results(csv_path: str, output_dir: str) -> None:
    """Generate plots from the sweep CSV."""
    data = read_csv(csv_path)
    gs = data["grid_size"]

    # --- Plot 1: Grid size vs time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gs, data["elapsed_s"], "o-", color="tab:blue", linewidth=2)
    ax.set_xlabel("Grid Size (NxN)")
    ax.set_ylabel("Elapsed Time (s)")
    ax.set_title("HotSpot Simulation Time vs Grid Resolution")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(gs)
    ax.set_xticklabels([f"{int(g)}" for g in gs], rotation=45)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "grid_size_vs_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {path}")

    # --- Plot 2: Grid size vs peak temp ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gs, data["peak_temp_K"], "s-", color="tab:red", linewidth=2)
    ax.set_xlabel("Grid Size (NxN)")
    ax.set_ylabel("Peak Temperature (K)")
    ax.set_title("Peak Temperature vs Grid Resolution")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([f"{int(g)}" for g in gs], rotation=45)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "grid_size_vs_peak_temp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {path}")

    # --- Plot 3: Per-tier silicon max temp vs grid size ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for t in range(NUM_TIERS):
        col = f"tier_{t}_si_max_K"
        if col in data:
            ax.plot(gs, data[col], "o-", color=colors[t], linewidth=2,
                    label=f"Tier {t}")
    ax.set_xlabel("Grid Size (NxN)")
    ax.set_ylabel("Silicon Max Temperature (K)")
    ax.set_title("Per-Tier Silicon Max Temperature vs Grid Resolution")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([f"{int(g)}" for g in gs], rotation=45)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "grid_size_vs_tier_temps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep HotSpot grid resolution and plot results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir", default="sweep_grid_size_results",
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: only run grid sizes [2, 4]",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip simulations, just re-plot from existing CSV",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    csv_path = os.path.join(output_dir, "sweep_results.csv")

    if not args.plot_only:
        grid_sizes = TEST_GRID_SIZES if args.test else FULL_GRID_SIZES
        csv_path = run_sweep(grid_sizes, output_dir)

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found at {csv_path}", file=sys.stderr)
        print("Run without --plot-only first.", file=sys.stderr)
        sys.exit(1)

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib not installed — skipping plots.")
        print("Install with: pip install matplotlib")
        print("Then re-run with --plot-only to generate plots from the CSV.")
    else:
        print("\nGenerating plots...")
        plot_results(csv_path, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
